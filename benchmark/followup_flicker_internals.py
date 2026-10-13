"""
Задача C: почему проверка мерцания молчит — сырые промежуточные величины.

Логируется то, что штатный пайплайн наружу не отдаёт: дисперсия СЫРОГО
фонового сигнала ДО порогового отсечения, сам порог, прошёл ли гейт, и
мощность на наложенной частоте ОТДЕЛЬНО для фона и для лица.

Как это сделано без правки quality.py и pipeline.py: метод
RPPGPipeline._estimate_background_signal оборачивается на время прогона.
Обёртка вызывает оригинал (его решение и есть гейт), а рядом ПОВТОРЯЕТ ту
же цепочку preprocess_signal -> compute_psd на тех же данных, но
безусловно — чтобы увидеть, что детектор увидел БЫ, если бы гейт его
пропустил. Логика самих функций не меняется, они только вызываются ещё
раз снаружи.

Мощность на частоте берётся линейной интерполяцией по сетке compute_psd.
И фоновый, и лицевой сигналы проходят один и тот же preprocess_signal с
z-score, поэтому мощности сравнимы между собой, но не абсолютны.

    PYTHONPATH=src venv/bin/python3 benchmark/followup_flicker_internals.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from rppg.config import PipelineConfig  # noqa: E402
from rppg.degrade import FLICKER_SPECS, FrameDegrader, prepare_record  # noqa: E402
from rppg.pipeline import RPPGPipeline  # noqa: E402
from rppg.signal.preprocessing import interpolate_missing, preprocess_signal  # noqa: E402
from rppg.signal.quality import compute_psd  # noqa: E402

import degenerate as dg  # noqa: E402
import eval_ubfc_subject as ev  # noqa: E402

CONDITIONS = ("B1w", "B1f", "B2f")
FACE_ROI = "forehead"  # самая крупная из трёх зон; мерцание по маске лица её покрывает


def _power_at(signal, fps: float, f_hz: float) -> float:
    if signal is None or len(signal) < 8:
        return float("nan")
    freqs, psd = compute_psd(np.asarray(signal, dtype=float), fps)
    if freqs is None or psd is None:
        return float("nan")
    return float(np.interp(f_hz, freqs, psd))


def _preprocess_like_pipeline(raw, valid_mask, fps, band, cfg):
    fixed, ok = interpolate_missing(np.asarray(raw, dtype=float), np.asarray(valid_mask, dtype=bool))
    if not ok:
        return None, float("nan")
    raw_std = float(np.std(fixed))
    processed = preprocess_signal(
        fixed, fps, band[0], band[1],
        order=cfg.filt.filter_order,
        detrend_method=cfg.filt.detrend_method,
        tarvainen_lambda=cfg.filt.tarvainen_lambda,
        normalize_method=cfg.filt.normalize_method,
    )
    return processed, raw_std


def run_pair(record_id: str, subject_dir: Path, condition: str, cfg_template: PipelineConfig) -> list[dict]:
    video_path = ev.find_video(subject_dir)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg_template.window.assumed_fps

    cfg = PipelineConfig()
    cfg.face.model_asset_path = str(ev.MODEL_PATH)
    cfg.window.assumed_fps = float(fps)
    threshold = cfg.quality.flicker_min_raw_std

    injected_hz = FLICKER_SPECS[condition][0]
    prep = prepare_record(str(video_path), condition, float(fps))
    degrader = FrameDegrader(condition, record_id, float(fps), prep, injected_hz=injected_hz)

    internals: dict[int, dict] = {}
    holder = {"ts": -1}
    original = RPPGPipeline._estimate_background_signal

    def traced(self, fps_inner, band):
        out = original(self, fps_inner, band)
        bg_valid = np.array(self._buf.background_valid, dtype=bool)
        bg_raw_std, bg_power = float("nan"), float("nan")
        if bg_valid.sum() >= 8:
            green = np.array(self._buf.background_rgb)[:, 1]
            processed, bg_raw_std = _preprocess_like_pipeline(green, bg_valid, fps_inner, band, self.cfg)
            bg_power = _power_at(processed, fps_inner, injected_hz)
        face_power = float("nan")
        roi_rgb = self._buf.rgb_by_roi.get(FACE_ROI)
        roi_valid = self._buf.valid_by_roi.get(FACE_ROI)
        if roi_rgb is not None and roi_valid is not None and np.array(roi_valid, dtype=bool).sum() >= 8:
            green_face = np.array([
                (v[1] if v is not None else np.nan) for v in roi_rgb
            ], dtype=float)
            processed_face, _ = _preprocess_like_pipeline(
                np.nan_to_num(green_face, nan=0.0), np.array(roi_valid, dtype=bool), fps_inner, band, self.cfg,
            )
            face_power = _power_at(processed_face, fps_inner, injected_hz)
        internals[holder["ts"]] = {
            "bg_region_dispersion_raw": bg_raw_std,
            "bg_dispersion_threshold": threshold,
            "bg_gate_passed": out is not None,
            "bg_power_at_injected_freq": bg_power,
            "face_power_at_injected_freq": face_power,
        }
        return out

    RPPGPipeline._estimate_background_signal = traced
    rows = []
    try:
        with RPPGPipeline(cfg, debug=True) as pipeline:
            frame_idx = 0
            results = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = int(frame_idx / fps * 1000)
                holder["ts"] = ts
                res = pipeline.process_frame(degrader(frame, frame_idx), ts)
                if res is not None:
                    results.append(res)
                frame_idx += 1
    finally:
        RPPGPipeline._estimate_background_signal = original
        cap.release()

    for idx, res in enumerate(results):
        got = internals.get(res.timestamp_ms)
        flag = bool(res.debug.sqi.flicker_suspected) if res.debug is not None else False
        row = {
            "record_id": record_id, "condition": condition, "window_idx": idx,
            "bg_region_dispersion_raw": "", "bg_dispersion_threshold": f"{threshold:.6g}",
            "bg_gate_passed": "", "bg_power_at_injected_freq": "",
            "face_power_at_injected_freq": "", "flicker_flag_final": "1" if flag else "0",
        }
        if got is not None:
            row.update({
                "bg_region_dispersion_raw": f"{got['bg_region_dispersion_raw']:.6g}",
                "bg_gate_passed": "1" if got["bg_gate_passed"] else "0",
                "bg_power_at_injected_freq": f"{got['bg_power_at_injected_freq']:.6g}",
                "face_power_at_injected_freq": f"{got['face_power_at_injected_freq']:.6g}",
            })
        rows.append(row)
    print(f"  {record_id:6s} {condition:4s} окон {len(rows):3d}, гейт пройден "
          f"{sum(1 for r in rows if r['bg_gate_passed'] == '1'):3d}, флагов "
          f"{sum(1 for r in rows if r['flicker_flag_final'] == '1'):3d}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="~/Downloads")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate" / "followup"))
    ap.add_argument("--records", default="", help="по умолчанию — все записи из task0_windows.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.records:
        wanted = [r.strip() for r in args.records.split(",") if r.strip()]
    else:
        task0 = out_dir.parent / "task0_windows.csv"
        with task0.open(encoding="utf-8") as fh:
            wanted = sorted({r["record_id"] for r in csv.DictReader(fh)}, key=dg._natural_key)

    by_id = dict(dg.discover_records(Path(args.data_dir).expanduser()))
    cfg = PipelineConfig()
    cfg.face.model_asset_path = str(ev.MODEL_PATH)

    all_rows = []
    for condition in CONDITIONS:
        print(f"--- {condition} ---", flush=True)
        for record_id in wanted:
            all_rows.extend(run_pair(record_id, by_id[record_id], condition, cfg))

    path = out_dir / "flicker_internals.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nвыгружено {len(all_rows)} строк в {path}")


if __name__ == "__main__":
    main()
