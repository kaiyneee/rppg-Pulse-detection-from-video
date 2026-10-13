"""
Лаборатория мелких участков кожи: даёт ли усреднение спектров по 11
участкам (см. patch_cache.py) выигрыш сверх трёх штатных ROI.

Окно участка повторяет буфер пайплайна: кадры с меткой в
[t_end - 10 с, t_end], пропуски заполняются fill_gaps, окно отбрасывается
по is_gap_acceptable. Упрощение относительно пайплайна: для участков нет
проверки стабильности числа валидных пикселей (_check_valid_pixel_count_
stability) — её состояние привязано к штатным ROI.

    PYTHONPATH=src venv/bin/python3 benchmark/patch_lab.py
"""

from __future__ import annotations

import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import estimator_lab as L  # noqa: E402
import experiments as ex  # noqa: E402
import offline_estimators as oe  # noqa: E402
from rppg.signal.methods import SignalWindow, get_method  # noqa: E402
from rppg.signal.preprocessing import fill_gaps, is_gap_acceptable, preprocess_signal  # noqa: E402

PATCH_DIR = REPO_ROOT / "cache" / "patches"
POS = get_method("pos")


@lru_cache(maxsize=None)
def patches(rid: str) -> dict:
    return pickle.loads((PATCH_DIR / f"{rid}.pkl").read_bytes())


@lru_cache(maxsize=None)
def patch_spectra(rid: str) -> list[dict]:
    """[окно] -> {участок: спектр на BPM_GRID}"""
    data = patches(rid)
    ts = np.array(data["timestamps_ms"])
    out = []
    for w in L.windows(rid):
        sel = (ts >= w.timestamp_ms - 10000) & (ts <= w.timestamp_ms)
        idx = np.where(sel)[0]
        if len(idx) < 32:
            out.append({})
            continue
        t_win = ts[idx]
        fps = (len(t_win) - 1) / ((t_win[-1] - t_win[0]) / 1000.0)
        d = {}
        for k, name in enumerate(data["names"]):
            valid = data["valid"][idx, k] & data["detected"][idx]
            if not is_gap_acceptable(valid):
                continue
            raw = data["rgb"][idx, k, :]
            rgb = np.stack([fill_gaps(raw[:, ch], valid) for ch in range(3)], axis=1)
            win = SignalWindow(rgb_traces={name: rgb}, fps=fps, landmark_trajectories=None,
                               valid_mask=valid, hr_band_hz=w.band)
            try:
                y = preprocess_signal(POS.extract(win, name), fps, w.band[0], w.band[1],
                                      order=L.CFG.filt.filter_order, detrend_method=L.CFG.filt.detrend_method,
                                      tarvainen_lambda=L.CFG.filt.tarvainen_lambda,
                                      normalize_method=L.CFG.filt.normalize_method)
            except Exception:  # noqa: BLE001
                continue
            if not np.all(np.isfinite(y)):
                continue
            f, p = oe.spec_periodogram(y, fps)
            d[name] = oe.on_grid(f, p)
        out.append(d)
    return out


def run(records, use_rois: bool, use_patches: bool, how: str = "mean"):
    est = {}
    for rid in records:
        ws = L.windows(rid)
        mask = oe.band_mask(ws[0].band)
        rois = L.record_spectra(rid, "pos", "periodogram") if use_rois else [{}] * len(ws)
        pts = patch_spectra(rid) if use_patches else [{}] * len(ws)
        seq = []
        for r, p in zip(rois, pts):
            c = L.combined([r, p], mask, how)
            seq.append(oe.argmax_bpm(c, mask) if c is not None else np.nan)
        est[rid] = np.array(seq)
    return est


def main() -> None:
    dev, _ = ex.split()
    print(f"разработочная часть, {len(dev)} записей; мерило — эталон по полному окну")
    for label, a, b, how in (("3 ROI (как в пайплайне, fft+mean)", True, False, "mean"),
                             ("11 участков", False, True, "mean"),
                             ("11 участков, веса по чистоте спектра", False, True, "pfw"),
                             ("3 ROI + 11 участков", True, True, "mean")):
        mae, orate = L.score(dev, run(dev, a, b, how))
        print(f"  {label:40s} MAE {mae:.3f}  промахи>5 {orate:.2%}", flush=True)


if __name__ == "__main__":
    main()
