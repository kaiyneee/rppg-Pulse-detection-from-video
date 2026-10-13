"""
Лаборатория оценщика: метод извлечения x объединение ROI x спектр x
отслеживание — на разработочной части UBFC, по набору spectral_dataset.

Сигналы пересчитываются из сохранённого RGB ТЕМИ ЖЕ функциями пайплайна
(rppg.signal.methods + preprocess_signal); для POS это проверено побитово.
Основное мерило — эталон по полному 10-секундному окну (ref_full), см.
offline_estimators.ref_spectral_full; промахи — |ошибка| > 5 уд/мин.

    PYTHONPATH=src venv/bin/python3 benchmark/estimator_lab.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import experiments as ex  # noqa: E402
import offline_estimators as oe  # noqa: E402
import spectral_dataset as sd  # noqa: E402
from rppg.config import PipelineConfig  # noqa: E402
from rppg.signal.methods import SignalWindow, get_method  # noqa: E402
from rppg.signal.preprocessing import preprocess_signal  # noqa: E402

FULL_CACHE = REPO_ROOT / "cache" / "ref_full.pkl"
CFG = PipelineConfig()
METHODS = {name: get_method(name) for name in ("pos", "chrom", "green", "pca")}


@lru_cache(maxsize=None)
def windows(rid: str):
    return sd.load_record(rid)


@lru_cache(maxsize=None)
def refs_full():
    return pickle.loads(FULL_CACHE.read_bytes())


def processed(w, roi: str, method: str) -> np.ndarray | None:
    rgb = w.rgb.get(roi)
    if rgb is None:
        return None
    win = SignalWindow(rgb_traces={roi: rgb}, fps=w.fps, landmark_trajectories=None,
                       valid_mask=w.valid[roi], hr_band_hz=w.band)
    try:
        raw = METHODS[method].extract(win, roi)
    except Exception:  # noqa: BLE001
        return None
    return preprocess_signal(raw, w.fps, w.band[0], w.band[1], order=CFG.filt.filter_order,
                             detrend_method=CFG.filt.detrend_method,
                             tarvainen_lambda=CFG.filt.tarvainen_lambda,
                             normalize_method=CFG.filt.normalize_method)


@lru_cache(maxsize=None)
def record_spectra(rid: str, method: str, spec: str) -> list[dict]:
    """[окно] -> {roi: спектр на BPM_GRID}"""
    fn = {"periodogram": oe.spec_periodogram, "welch8": oe.spec_pipeline_welch}[spec]
    out = []
    for w in windows(rid):
        d = {}
        for roi in ("forehead", "left_cheek", "right_cheek"):
            y = processed(w, roi, method)
            if y is None or len(y) < 32 or not np.all(np.isfinite(y)):
                continue
            f, p = fn(y, w.fps)
            d[roi] = oe.on_grid(f, p)
        out.append(d)
    return out


def peak_fraction(s: np.ndarray, mask: np.ndarray, half_bpm: float = 3.0) -> float:
    """Доля мощности полосы в окрестности главного пика — мера «чистоты» спектра."""
    sm = np.where(mask, s, 0.0)
    i = int(np.argmax(sm))
    step = oe.BPM_GRID[1] - oe.BPM_GRID[0]
    h = int(round(half_bpm / step))
    total = float(sm.sum())
    return float(sm[max(i - h, 0):i + h + 1].sum() / total) if total > 0 else 0.0


def combined(specs: list[dict], mask: np.ndarray, how: str) -> np.ndarray | None:
    """specs — список словарей {roi: спектр} (по методам). how:
    mean   — равное среднее нормированных спектров;
    pfw    — среднее с весами = доля мощности у главного пика (peak fraction)^2."""
    items = [s for d in specs for s in d.values()]
    if not items:
        return None
    norm = [oe.normalize(np.where(mask, s, 0.0)) for s in items]
    if how == "mean":
        return np.mean(norm, axis=0)
    if how == "pfw":
        wts = np.array([peak_fraction(s, mask) ** 2 for s in norm])
        if wts.sum() <= 0:
            return np.mean(norm, axis=0)
        return np.average(norm, axis=0, weights=wts)
    raise ValueError(how)


def run(records, methods: tuple[str, ...], spec: str, how: str, track_sigma: float | None):
    est = {}
    for rid in records:
        ws = windows(rid)
        mask = oe.band_mask(ws[0].band)
        per_method = [record_spectra(rid, m, spec) for m in methods]
        seq = []
        for t in range(len(ws)):
            c = combined([pm[t] for pm in per_method], mask, how)
            seq.append(c if c is not None else np.ones(len(oe.BPM_GRID)))
        if track_sigma:
            est[rid] = oe.viterbi_track(seq, mask, sigma_bpm=track_sigma)
        else:
            est[rid] = np.array([oe.argmax_bpm(s, mask) for s in seq])
    return est


def score(records, est):
    rf = refs_full()
    per_rec, out_n, tot = [], 0, 0
    for rid in records:
        e = []
        for w, b in zip(windows(rid), est[rid]):
            r = rf.get((rid, w.timestamp_ms))
            if not w.publishable or r is None or not np.isfinite(b):
                continue
            e.append(abs(b - r))
        if e:
            per_rec.append(np.mean(e))
            out_n += int(np.sum(np.array(e) > 5.0))
            tot += len(e)
    return float(np.mean(per_rec)), out_n / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()
    dev, test = ex.split()
    records = {"dev": dev, "test": test, "all": dev + test}[args.set]

    grid = [
        (("pos",), "periodogram", "mean", None),
        (("chrom",), "periodogram", "mean", None),
        (("green",), "periodogram", "mean", None),
        (("pca",), "periodogram", "mean", None),
        (("pos", "chrom"), "periodogram", "mean", None),
        (("pos", "chrom", "green"), "periodogram", "mean", None),
        (("pos",), "periodogram", "pfw", None),
        (("pos", "chrom"), "periodogram", "pfw", None),
        (("pos", "chrom"), "periodogram", "mean", 4.0),
        (("pos", "chrom"), "periodogram", "pfw", 4.0),
    ]
    print(f"набор: {args.set}, записей {len(records)}; мерило — эталон по полному окну")
    print(f"{'методы':22s}{'спектр':13s}{'объед.':8s}{'Витерби':9s}{'MAE':>8s}{'промахи>5':>11s}")
    for methods, spec, how, sigma in grid:
        mae, orate = score(records, run(records, methods, spec, how, sigma))
        print(f"{'+'.join(methods):22s}{spec:13s}{how:8s}{('σ=' + str(sigma)) if sigma else '—':9s}"
              f"{mae:>8.3f}{orate:>10.2%}", flush=True)


if __name__ == "__main__":
    main()
