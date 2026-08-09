"""
Задача 23: мини-бенчмарк на фиксированном наборе из 6 видео (расширен с 4 —
добавлены "мужчина" и "вероника", два новых типа отказа, см. ниже) —
регрессионный тест, который не даст "починить" SQI просто ослабив всё подряд.

Прогоняет RPPGPipeline с ОДНИМ фиксированным PipelineConfig по фиксированному
списку видео и печатает/сохраняет сводную таблицу: число окон, покрытие (по
ЕДИНОМУ источнику истины sqi.is_reliable, см. задачу 16 — не пересчитывается
здесь отдельно), перцентили 5/25/50/75/95 всех компонент SQI.

Наблюдаемый на этих 6 видео порядок при min_spectral_snr_db=-3.0 (задача 22,
см. план/задачу 23), который любая последующая перекалибровка ОБЯЗАНА
сохранить — чёткий разрыв на две группы по покрытию:
  ХОРОШАЯ группа (26-51% покрытия, landmark_stability ~0.99-1.00):
    - "тётя" — 51.4%;
    - "девушка темнокожая вертикально/горизонтально" — 51.1%/26.4%.
  ПРОБЛЕМНАЯ группа (0-19% покрытия, landmark_stability ~0.76-0.81):
    - "вероника" (движение при хорошем свете) — 19.1%;
    - "мужчина" (разрывы трекинга, ~24% окон window_status=no_signal) — 14.3%;
    - "девочка" (движение в кадре) — ХУДШАЯ, 0.0%, ни одного publishable окна.
Минимум ХОРОШЕЙ группы (26.4%) строго выше максимума ПРОБЛЕМНОЙ (19.1%) —
см. tests/test_signal_pipeline.py::test_ranking_preserved, пропускается,
если видео недоступны локально.

Использование:
    PYTHONPATH=src python3 scripts/run_benchmark.py
    PYTHONPATH=src python3 scripts/run_benchmark.py --videos a.mp4 b.mp4 --out results.csv
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

from rppg.config import PipelineConfig
from rppg.pipeline import RPPGPipeline

# 6 видео (расширено в задаче 23 с исходных 4 — добавлены "мужчина" и
# "вероника") — не часть репозитория, путь настраивается через --videos
# для запуска в другом окружении.
DEFAULT_VIDEOS = [
    "/Users/ylhjn/Desktop/проекты/статья пульс по видо/видео тетя.mp4",
    "/Users/ylhjn/Desktop/проекты/статья пульс по видо/девушка темнокожая вертикально.mp4.mp4",
    "/Users/ylhjn/Desktop/проекты/статья пульс по видо/девушка темнокожая горизонтально.mp4",
    "/Users/ylhjn/Desktop/проекты/статья пульс по видо/мужчина важный.mp4",
    "/Users/ylhjn/Desktop/вероника .mov",
    "/Users/ylhjn/Desktop/проекты/статья пульс по видо/девочка.mp4",
]

PERCENTILES = [5, 25, 50, 75, 95]
SQI_COMPONENTS = [
    "spectral_snr_db",
    "harmonic_score",
    "cross_roi_agreement",
    "landmark_stability",
    "temporal_consistency",
]


def run_video(video_path: str, model_path: str, config: PipelineConfig) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps
    config.face.model_asset_path = model_path

    n_windows = 0
    n_publishable = 0
    component_values: dict[str, list[float]] = {c: [] for c in SQI_COMPONENTS}

    frame_idx = 0
    with RPPGPipeline(config, debug=True) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                n_windows += 1
                if result.publishable:  # задача 16: sqi.is_reliable, единственный источник истины
                    n_publishable += 1
                if result.debug is not None:
                    sqi = result.debug.sqi
                    component_values["spectral_snr_db"].append(sqi.spectral_snr_db)
                    component_values["harmonic_score"].append(sqi.harmonic_score)
                    component_values["cross_roi_agreement"].append(sqi.cross_roi_agreement)
                    component_values["landmark_stability"].append(sqi.landmark_stability)
                    component_values["temporal_consistency"].append(sqi.temporal_consistency)
            frame_idx += 1
    cap.release()

    percentiles: dict[str, list[float]] = {}
    for name, values in component_values.items():
        finite = np.array([v for v in values if np.isfinite(v)])
        percentiles[name] = list(np.percentile(finite, PERCENTILES)) if len(finite) else [float("nan")] * len(PERCENTILES)

    return {
        "n_windows": n_windows,
        "n_publishable": n_publishable,
        "coverage": (n_publishable / n_windows) if n_windows else float("nan"),
        "percentiles": percentiles,
    }


def print_summary(name: str, summary: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  окон: {summary['n_windows']}, publishable: {summary['n_publishable']} "
          f"({summary['coverage'] * 100:.1f}%)")
    header = f"  {'компонента':<24s}" + "".join(f"{p:>9d}%" for p in PERCENTILES)
    print(header)
    for comp in SQI_COMPONENTS:
        row = f"  {comp:<24s}" + "".join(f"{v:9.2f}" for v in summary["percentiles"][comp])
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--out", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        raise SystemExit(
            "Видео не найдены:\n" + "\n".join(f"  {v}" for v in missing) +
            "\n\nПередайте реальные пути через --videos."
        )

    results: dict[str, dict] = {}
    for video in videos:
        config = PipelineConfig()
        summary = run_video(str(video), args.model, config)
        results[video.stem] = summary
        print_summary(video.stem, summary)

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["video", "n_windows", "n_publishable", "coverage"]
        for comp in SQI_COMPONENTS:
            header += [f"{comp}_p{p}" for p in PERCENTILES]
        writer.writerow(header)
        for name, summary in results.items():
            row = [name, summary["n_windows"], summary["n_publishable"], f"{summary['coverage']:.4f}"]
            for comp in SQI_COMPONENTS:
                row += [f"{v:.3f}" for v in summary["percentiles"][comp]]
            writer.writerow(row)
    print(f"\nСводная таблица сохранена в {out_path}")


if __name__ == "__main__":
    main()
