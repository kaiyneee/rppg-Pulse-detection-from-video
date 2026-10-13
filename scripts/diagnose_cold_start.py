"""
Задача 49: проверяет гипотезу "первые окна хуже из-за краевых эффектов
фильтра на едва заполненном буфере" — не выдавать оценку, пока в буфере
не накопится window_seconds + 2*edge_trim_seconds (12с вместо 10),
сравнивает MAE (в целом и на первых 5 окнах) до/после на всех доступных
испытуемых. НЕ пишет в ubfc_results.csv — это эксперимент, не финальный
прогон (см. eval_ubfc_subject.py для накопительной оценки).

Использование:
    PYTHONPATH=src python3 scripts/diagnose_cold_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_ubfc_subject import find_video, find_ground_truth, load_reference_waveform, reference_median_for_window, MODEL_PATH
from rppg.config import PipelineConfig
from rppg.pipeline import RPPGPipeline
import cv2

SUBJECT_DIRS = [
    "/Users/ylhjn/Downloads/5-gt",
    "/Users/ylhjn/Downloads/6-gt",
    "/Users/ylhjn/Downloads/7-gt",
    "/Users/ylhjn/Downloads/8-gt",
]


def run_variant(video_path: Path, t_ref, hr_ref, skip_seconds: float) -> dict:
    """skip_seconds=0 -> базовое поведение как есть. skip_seconds>0 -> НЕ
    подавать в pipeline.process_frame первые skip_seconds видео (буфер их
    вообще никогда не увидит — window_seconds в WindowConfig НЕ трогается,
    т.к. кольцевой буфер физически не может хранить больше window_seconds
    данных: min_seconds_before_estimate > window_seconds просто никогда не
    выполняется и не даёт вообще ни одной оценки — проверено эмпирически).
    Абсолютные timestamp_ms НЕ сдвигаются, чтобы сверка с эталоном по
    реальному времени записи оставалась корректной."""
    config = PipelineConfig()
    config.face.model_asset_path = str(MODEL_PATH)

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
            t_s = frame_idx / fps
            frame_idx += 1
            if t_s < skip_seconds:
                continue
            timestamp_ms = int(t_s * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None and result.bpm == result.bpm:
                window_end_s = timestamp_ms / 1000.0
                ref = reference_median_for_window(t_ref, hr_ref, window_end_s, config.window.window_seconds)
                if ref is not None:
                    errors.append(result.bpm - ref)
    cap.release()
    errors = np.array(errors)
    return {
        "n": len(errors),
        "mae_all": float(np.mean(np.abs(errors))) if len(errors) else None,
        "mae_first5": float(np.mean(np.abs(errors[:5]))) if len(errors) >= 1 else None,
        "first5": errors[:5].tolist(),
    }


def main() -> None:
    print(f"{'испытуемый':<12s}{'MAE(0с) все':>14s}{'MAE(0с) п.5':>14s}{'MAE(2с) все':>14s}{'MAE(2с) п.5':>14s}")
    baseline_first5, experiment_first5 = [], []
    baseline_all, experiment_all = [], []
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

        baseline = run_variant(video_path, t_ref, hr_ref, skip_seconds=0.0)
        experiment = run_variant(video_path, t_ref, hr_ref, skip_seconds=2.0)

        baseline_first5.append(baseline["mae_first5"] if baseline["mae_first5"] is not None else np.nan)
        experiment_first5.append(experiment["mae_first5"] if experiment["mae_first5"] is not None else np.nan)
        baseline_all.append(baseline["mae_all"] if baseline["mae_all"] is not None else np.nan)
        experiment_all.append(experiment["mae_all"] if experiment["mae_all"] is not None else np.nan)

        def _f(x):
            return f"{x:.2f}" if x is not None else "—"
        print(f"{subject:<12s}{_f(baseline['mae_all']):>14s}{_f(baseline['mae_first5']):>14s}"
              f"{_f(experiment['mae_all']):>14s}{_f(experiment['mae_first5']):>14s}")
        print(f"    n окон (0с/2с): {baseline['n']}/{experiment['n']}")
        print(f"    первые 5 ошибок (0с): {[round(e, 1) for e in baseline['first5']]}")
        print(f"    первые 5 ошибок (2с): {[round(e, 1) for e in experiment['first5']]}")

    b5, e5 = np.nanmean(baseline_first5), np.nanmean(experiment_first5)
    print(f"\nСреднее MAE на первых 5 окнах по испытуемым: "
          f"0с={b5:.2f}, 2с={e5:.2f} "
          f"({'СНИЖЕНИЕ' if e5 < b5 else 'НЕ снижается'} в {b5 / max(e5, 1e-9):.2f}x)")
    worse = [
        s for s, b, e in zip(SUBJECT_DIRS, baseline_all, experiment_all)
        if not np.isnan(b) and not np.isnan(e) and e > b
    ]
    print(f"Общий MAE вырос у: {worse if worse else 'ни у кого'}")


if __name__ == "__main__":
    main()
