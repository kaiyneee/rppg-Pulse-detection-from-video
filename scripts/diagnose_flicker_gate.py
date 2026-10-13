"""
Диагностика задачи 0: почему детектор мерцания молчит на условиях B1w/B1f.

Детектор (quality.detect_illumination_flicker) видит фоновый сигнал только
если тот вообще дошёл до него: RPPGPipeline._estimate_background_signal
отбрасывает фон, у которого СКО сырого (до z-score) сигнала меньше
QualityConfig.flicker_min_raw_std. Скрипт логирует эту величину по окнам —
ОТДЕЛЬНО от того, совпала ли частота фона с лицевой.

Пайплайн не изменяется: метод оборачивается на время прогона здесь.

    PYTHONPATH=src venv/bin/python3 scripts/diagnose_flicker_gate.py --record 10 --condition B1w
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from rppg.config import PipelineConfig  # noqa: E402
from rppg.pipeline import RPPGPipeline  # noqa: E402
from rppg.signal.preprocessing import interpolate_missing  # noqa: E402
from rppg.degrade import FLICKER_SPECS, FrameDegrader, prepare_record  # noqa: E402

import degenerate as dg  # noqa: E402
import eval_ubfc_subject as ev  # noqa: E402

TRACE: list[dict] = []


def patch_background(pipeline_cls, qcfg_threshold: float):
    original = pipeline_cls._estimate_background_signal

    def traced(self, fps, band):
        valid = np.array(self._buf.background_valid, dtype=bool)
        raw_std = float("nan")
        bg_mean = float("nan")
        if valid.sum() >= 8:
            green = np.array(self._buf.background_rgb)[:, 1]
            fixed, ok = interpolate_missing(green, valid)
            if ok:
                raw_std = float(np.std(fixed))
                bg_mean = float(np.mean(fixed))
        out = original(self, fps, band)
        TRACE.append({
            "n_valid": int(valid.sum()),
            "bg_mean": bg_mean,
            "raw_std": raw_std,
            "passed_gate": out is not None,
            "threshold": qcfg_threshold,
        })
        return out

    pipeline_cls._estimate_background_signal = traced
    return original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default="10")
    ap.add_argument("--condition", default="B1w")
    ap.add_argument("--data-dir", default="~/Downloads")
    args = ap.parse_args()

    records = dict(dg.discover_records(Path(args.data_dir).expanduser()))
    subject_dir = records[args.record]
    video = ev.find_video(subject_dir)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cfg = PipelineConfig()
    cfg.face.model_asset_path = str(ev.MODEL_PATH)
    cfg.window.assumed_fps = float(fps)

    injected_hz = FLICKER_SPECS.get(args.condition, (None,))[0]
    prep = prepare_record(str(video), args.condition, float(fps))
    degrader = FrameDegrader(args.condition, args.record, float(fps), prep, injected_hz=injected_hz)

    original = patch_background(RPPGPipeline, cfg.quality.flicker_min_raw_std)
    rows = []
    try:
        with RPPGPipeline(cfg, debug=True) as pipeline:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = pipeline.process_frame(degrader(frame, idx), int(idx / fps * 1000))
                if res is not None:
                    rows.append(res)
                idx += 1
    finally:
        RPPGPipeline._estimate_background_signal = original
        cap.release()

    print(f"запись {args.record}, условие {args.condition}, наложено {degrader.injected_bpm or float('nan'):.1f} BPM")
    print(f"порог отбрасывания фона flicker_min_raw_std = {cfg.quality.flicker_min_raw_std} ур. яркости\n")
    passed = [t for t in TRACE if t["passed_gate"]]
    std = np.array([t["raw_std"] for t in TRACE if np.isfinite(t["raw_std"])])
    mean = np.array([t["bg_mean"] for t in TRACE if np.isfinite(t["bg_mean"])])
    print(f"окон с фоном: {len(TRACE)}, дошло до детектора: {len(passed)} ({len(passed)/max(len(TRACE),1):.1%})")
    if std.size:
        print(f"СКО сырого фона: медиана {np.median(std):.4f}, макс {std.max():.4f} "
              f"(нужно >= {cfg.quality.flicker_min_raw_std})")
        print(f"средняя яркость фонового патча: {np.median(mean):.1f} ур.")
        if injected_hz:
            amp = FLICKER_SPECS[args.condition][1]
            face_only = FLICKER_SPECS[args.condition][2]
            expected = 0.0 if face_only else amp * np.median(mean) / np.sqrt(2)
            print(f"ожидаемое СКО от наложенного мерцания на фоне: {expected:.4f} "
                  f"({'мерцание только по лицу — фон не модулирован' if face_only else 'весь кадр'})")

    n_pub = sum(1 for r in rows if r.publishable)
    flags = sum(1 for r in rows if r.debug is not None and r.debug.sqi.flicker_suspected)
    cands = sum(1 for r in rows if r.debug is not None and r.debug.sqi.flicker_candidate)
    no_debug = sum(1 for r in rows if r.debug is None)
    print(f"\nокон всего {len(rows)}: опубликовано {n_pub}, без debug {no_debug}, "
          f"кандидатов мерцания {cands}, подтверждённых флагов {flags}")


if __name__ == "__main__":
    main()
