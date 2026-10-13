"""
Сравнение офлайн-оценщиков на разработочной части UBFC.

Для каждого варианта считается MAE по записям (опубликованные окна —
флаги публикации штатного пайплайна, чтобы варианты сравнивались на одних
и тех же окнах) против ДВУХ эталонов: эталона статьи и независимого
эталона по ударам контактной волны на полном окне. Плюс доля грубых
промахов (|ошибка| > 5 уд/мин относительно эталона по ударам).

    PYTHONPATH=src venv/bin/python3 benchmark/offline_eval.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import experiments as ex  # noqa: E402
import offline_estimators as oe  # noqa: E402
import spectral_dataset as sd  # noqa: E402
import trace_cache as tc  # noqa: E402

BEATS_CACHE = REPO_ROOT / "cache" / "ref_beats.pkl"
FULL_CACHE = REPO_ROOT / "cache" / "ref_full.pkl"
ROIS = ("forehead", "left_cheek", "right_cheek")


def beats_refs(records: list[str]) -> dict:
    cache = pickle.loads(BEATS_CACHE.read_bytes()) if BEATS_CACHE.exists() else {}
    changed = False
    for rid in records:
        if rid in cache:
            continue
        trace = tc.load_trace(rid)
        windows = sd.load_record(rid)
        cache[rid] = {
            w.timestamp_ms: oe.ref_beats(trace.t_ref, trace.wave, w.timestamp_ms / 1000.0 - 10.0,
                                         w.timestamp_ms / 1000.0)
            for w in windows
        }
        changed = True
    if changed:
        BEATS_CACHE.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    return cache


SPECTRA = {
    "welch8": oe.spec_pipeline_welch,
    "periodogram": oe.spec_periodogram,
    "multitaper": lambda x, fps: oe.spec_multitaper(x, fps, nw=1.5, k=2),
}


def window_spectra(w, spec_name: str) -> dict[str, np.ndarray]:
    fn = SPECTRA[spec_name]
    out = {}
    for roi in ROIS:
        x = w.signals.get(roi)
        if x is None or len(x) < 32:
            continue
        f, p = fn(x, w.fps)
        out[roi] = oe.on_grid(f, p)
    return out


def combine(spectra: dict[str, np.ndarray], how: str, best_roi: str | None) -> np.ndarray | None:
    if not spectra:
        return None
    if how == "best":
        key = best_roi if best_roi in spectra else next(iter(spectra))
        return spectra[key]
    if how == "mean":
        return np.mean([oe.normalize(s) for s in spectra.values()], axis=0)
    raise ValueError(how)


def estimate_record(windows, spec_name, how, refine, track_sigma):
    mask = oe.band_mask(windows[0].band)
    specs = []
    for w in windows:
        s = combine(window_spectra(w, spec_name), how, w.best_roi)
        specs.append(s if s is not None else np.ones(len(oe.BPM_GRID)))
    if track_sigma:
        return oe.viterbi_track(specs, mask, sigma_bpm=track_sigma)
    return np.array([oe.argmax_bpm(s, mask, refine=refine) for s in specs])


def score(records, est_by_record, refs_full):
    """MAE по записям против эталона статьи (8 с) и эталона полного окна
    (10 с, основное мерило); промахи >5 — против эталона полного окна."""
    rows_paper, rows_full, out_n, out_tot = [], [], 0, 0
    for rid in records:
        windows = sd.load_record(rid)
        est = est_by_record[rid]
        ep, ef = [], []
        for w, b in zip(windows, est):
            if not w.publishable or not np.isfinite(b):
                continue
            if w.ref is not None:
                ep.append(abs(b - w.ref))
            rf = refs_full.get((rid, w.timestamp_ms))
            if rf is not None:
                ef.append(abs(b - rf))
        if ep:
            rows_paper.append(np.mean(ep))
        if ef:
            rows_full.append(np.mean(ef))
            out_n += int(np.sum(np.array(ef) > 5.0))
            out_tot += len(ef)
    return float(np.mean(rows_paper)), float(np.mean(rows_full)), out_n / max(out_tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()
    dev, test = ex.split()
    records = {"dev": dev, "test": test, "all": dev + test}[args.set]
    refs_f = pickle.loads(FULL_CACHE.read_bytes())

    variants = [("штатный пайплайн (после 2-го прохода)", None)]
    for spec in ("welch8", "periodogram", "multitaper"):
        for how in ("best", "mean"):
            variants.append((f"{spec:11s} {how:4s}", (spec, how, False, None)))
    for spec in ("welch8", "periodogram", "multitaper"):
        for sigma in (1.0, 2.0, 4.0):
            variants.append((f"{spec:11s} mean +Витерби σ={sigma:g}", (spec, "mean", False, sigma)))

    print(f"набор: {args.set}, записей {len(records)}")
    print(f"{'вариант':42s}{'MAE (эт. статьи, 8с)':>22s}{'MAE (полное окно)':>19s}{'промахи>5':>11s}")
    for label, spec in variants:
        est = {}
        for rid in records:
            windows = sd.load_record(rid)
            if spec is None:
                est[rid] = np.array([w.bpm for w in windows])
            else:
                est[rid] = estimate_record(windows, *spec)
        mp, mb, orate = score(records, est, refs_f)
        print(f"{label:42s}{mp:>22.3f}{mb:>19.3f}{orate:>10.2%}", flush=True)


if __name__ == "__main__":
    main()
