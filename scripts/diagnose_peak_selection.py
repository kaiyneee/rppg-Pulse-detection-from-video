"""
Задача 50: сравнивает MAE при peak_selection="argmax" (умолчание) против
"whitened" (спектральное отбеливание, см. rppg.signal.frequency.whiten_psd)
на всех доступных испытуемых UBFC. НЕ пишет в ubfc_results.csv — это
эксперимент для выбора умолчания, не финальный прогон.

Использование:
    PYTHONPATH=src python3 scripts/diagnose_peak_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_ubfc_subject import find_video, find_ground_truth, load_reference_waveform, reference_median_for_window, MODEL_PATH
from rppg.config import PipelineConfig
from rppg.pipeline import RPPGPipeline

SUBJECT_DIRS = [
    "/Users/ylhjn/Downloads/5-gt",
    "/Users/ylhjn/Downloads/6-gt",
    "/Users/ylhjn/Downloads/7-gt",
    "/Users/ylhjn/Downloads/8-gt",
]


def run_variant(video_path: Path, t_ref, hr_ref, peak_selection: str) -> dict:
    config = PipelineConfig()
    config.face.model_asset_path = str(MODEL_PATH)
    config.frequency.peak_selection = peak_selection

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps

    errors = []
    frame_idx = 0
    with RPPGPipeline(config) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None and result.bpm == result.bpm:
                window_end_s = timestamp_ms / 1000.0
                ref = reference_median_for_window(t_ref, hr_ref, window_end_s, config.window.window_seconds)
                if ref is not None:
                    errors.append(result.bpm - ref)
            frame_idx += 1
    cap.release()
    errors = np.array(errors)
    return {
        "n": len(errors),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
    }


def main() -> None:
    print(f"{'испытуемый':<12s}{'MAE argmax':>12s}{'MAE whitened':>14s}{'разница':>10s}")
    for subject_dir_str in SUBJECT_DIRS:
        subject_dir = Path(subject_dir_str)
        if not subject_dir.exists():
            print(f"  [SKIP] {subject_dir} не найден")
            continue
        subject = subject_dir.name[:-3] if subject_dir.name.endswith("-gt") else subject_dir.name
        video_path = find_video(subject_dir)
        gt_path = find_ground_truth(subject_dir)
        # Задача 53 переименовала загрузчик: теперь он отдаёт и волну.
        # Этим диагностикам нужна прежняя ЧСС прибора — берём третий элемент.
        t_ref, _wave, hr_ref = load_reference_waveform(gt_path)

        argmax_result = run_variant(video_path, t_ref, hr_ref, "argmax")
        whitened_result = run_variant(video_path, t_ref, hr_ref, "whitened")

        diff = whitened_result["mae"] - argmax_result["mae"]
        print(f"{subject:<12s}{argmax_result['mae']:>12.2f}{whitened_result['mae']:>14.2f}{diff:>+10.2f}")


if __name__ == "__main__":
    main()
