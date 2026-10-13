"""
Набор «сигналы трёх ROI по каждому окну» для офлайн-экспериментов с
оценкой частоты, объединением ROI и отслеживанием пика.

Сигналы снимаются при воспроизведении ШТАТНОГО пайплайна (trace_cache),
обёрткой вокруг RPPGPipeline._estimate_color_roi: после её вызова в
per_roi_signal лежит ровно тот обработанный сигнал (POS -> детренд ->
полоса -> нормировка), по которому пайплайн оценивает частоту. Ничего не
пересчитывается заново — берутся промежуточные величины самого пайплайна.

Вместе с сигналами сохраняется итог пайплайна по окну (BPM после второго
прохода, публикация, выбранный ROI, компоненты SQI) и эталон по волне.

    PYTHONPATH=src venv/bin/python3 benchmark/spectral_dataset.py
"""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import trace_cache as tc  # noqa: E402
from rppg.signal.preprocessing import fill_gaps, is_gap_acceptable  # noqa: E402
import experiments as ex  # noqa: E402

OUT_DIR = REPO_ROOT / "cache" / "spectral"


@dataclass
class WindowData:
    timestamp_ms: int
    fps: float
    band: tuple[float, float]
    signals: dict[str, np.ndarray]          # обработанный сигнал по ROI
    raw: dict[str, np.ndarray]              # сырой выход POS по ROI (до детренда)
    bpm_first_pass: float
    bpm: float                              # после второго прохода
    publishable: bool
    best_roi: str | None
    per_roi_bpm: dict[str, float]
    sqi: dict[str, float] = field(default_factory=dict)
    ref: float | None = None
    rgb: dict[str, np.ndarray] = field(default_factory=dict)    # RGB по ROI после заполнения пропусков
    valid: dict[str, np.ndarray] = field(default_factory=dict)  # маска валидных кадров по ROI


def build_record(record_id: str) -> list[WindowData]:
    trace = tc.load_trace(record_id)
    stash: dict = {"signals": {}, "raw": {}, "rgb": {}, "valid": {}, "fps": None, "band": None}
    windows: list[WindowData] = []

    def setup(pipe):
        original = pipe._estimate_color_roi

        def wrapped(name, fps, band, timestamps_sec, per_roi_bpm, per_roi_signal, warnings, **kw):
            # RGB и маска — ровно в том виде, в каком их видит _estimate_color_roi
            raw_rgb = np.array(pipe._buf.rgb_by_roi[name], dtype=float)
            vmask = np.array(pipe._buf.valid_by_roi[name], dtype=bool)
            if is_gap_acceptable(vmask):
                stash["rgb"][name] = np.stack([fill_gaps(raw_rgb[:, ch], vmask) for ch in range(3)], axis=1)
                stash["valid"][name] = vmask
            original(name, fps, band, timestamps_sec, per_roi_bpm, per_roi_signal, warnings, **kw)
            if name in per_roi_signal:
                stash["signals"][name] = np.array(per_roi_signal[name], dtype=float)
            raw = kw.get("per_roi_raw_signal")
            if raw is not None and name in raw:
                stash["raw"][name] = np.array(raw[name], dtype=float)
            stash["fps"], stash["band"] = float(fps), tuple(band)

        pipe._estimate_color_roi = wrapped

    def on_result(pipe, res):
        sqi = res.debug.sqi if res.debug is not None else None
        windows.append(WindowData(
            timestamp_ms=res.timestamp_ms,
            fps=stash["fps"] if stash["fps"] is not None else trace.fps,
            band=stash["band"] if stash["band"] is not None else (0.7, 4.0),
            signals=dict(stash["signals"]), raw=dict(stash["raw"]),
            rgb=dict(stash["rgb"]), valid=dict(stash["valid"]),
            bpm_first_pass=float(res.bpm), bpm=float(res.bpm),
            publishable=bool(res.publishable),
            best_roi=res.debug.best_source if res.debug is not None else None,
            per_roi_bpm=dict(res.per_roi_bpm),
            sqi={} if sqi is None else {
                "overall": sqi.overall_score, "snr_db": sqi.spectral_snr_db,
                "roi_agreement": sqi.cross_roi_agreement, "landmark": sqi.landmark_stability,
                "temporal": sqi.temporal_consistency, "harmonic": sqi.harmonic_score,
            },
        ))
        stash["signals"], stash["raw"], stash["rgb"], stash["valid"] = {}, {}, {}, {}

    rows, _ = tc.replay(trace, setup=setup, on_result=on_result)
    # второй проход меняет bpm на месте в rows — переносим итог
    for w, r in zip(windows, rows):
        w.bpm = float(r.bpm)
        w.ref = ex.reference(trace, w.timestamp_ms)
    return windows


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "WindowData":
            return WindowData
        return super().find_class(module, name)


def load_record(record_id: str) -> list[WindowData]:
    with (OUT_DIR / f"{record_id}.pkl").open("rb") as fh:
        return _Unpickler(fh).load()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    for path in sorted(tc.CACHE_DIR.glob("*.pkl")):
        rid = path.stem
        w = build_record(rid)
        (OUT_DIR / f"{rid}.pkl").write_bytes(pickle.dumps(w, protocol=pickle.HIGHEST_PROTOCOL))
        n += len(w)
    ex.save_refs()
    print(f"окон: {n}, записей: {len(list(OUT_DIR.glob('*.pkl')))}, {time.time() - t0:.1f}с")


if __name__ == "__main__":
    main()
