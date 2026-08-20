"""
Задача 19: подбор полуширины узкой полосы вокруг пика в
quality.dominant_frequency_and_snr (см. QualityConfig.snr_peak_half_width_*).

Контекст: значение 0.15 Hz было назначено ДО того, как задача 5 сменила
nperseg в quality.compute_psd (128 -> len//2, ради реального усреднения
Уэлча) — при 30 fps это расширило главный лепесток окна примерно вдвое, и
эффект двух правок вместе никогда не измерялся. Плюс задача 13а добавила
2-ю гармонику в energию сигнала — три изменения, взаимодействие которых
неизвестно без прогона на реальных данных.

Этот скрипт НЕ выбирает "победителя" автоматически: расширение полосы
почти всегда поднимает средний SNR (в полосу попадает больше энергии, в
т.ч. шумовой) — выбор по "у кого SNR выше" был бы самообманом. Вместо этого
печатает таблицу "полуширина x видео -> медиана spectral_snr_db" и метрику
РАЗДЕЛЯЮЩЕЙ СПОСОБНОСТИ (разброс медиан между видео при данной полуширине:
чем он больше, тем лучше полоса отличает "хороший" сигнал от "плохого" —
если разброс схлопывается, полоса просто размывает всё до одинаково
"высокого" SNR, включая шум). Выбор конкретного значения — за человеком,
глядя на эту таблицу (см. задачу 19, критерий приёмки).

Использование:
    PYTHONPATH=src python3 scripts/sweep_snr_band.py
    PYTHONPATH=src python3 scripts/sweep_snr_band.py --videos a.mp4 b.mp4 --model models/face_landmarker.task
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.config import PipelineConfig
from rppg.pipeline import RPPGPipeline

# Задача 19: список видео ЕДИНЫЙ для всех сравнительных скриптов — импорт
# из run_benchmark.py (задача 23, 6 видео: тётя/темнокожая×2/мужчина/
# вероника/девочка), а не отдельная копия, которая могла бы разъехаться со
# списком в run_benchmark.py/test_ranking_preserved (как уже случилось
# однажды с путём к "вероника .mov" — см. git log).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_benchmark import DEFAULT_VIDEOS

FIXED_CANDIDATES_HZ = [0.10, 0.15, 0.20, 0.25]
ADAPTIVE_K_CANDIDATES = [1.0, 1.44, 2.0]


def _median_snr_for_video(video_path: str, model_path: str, config: PipelineConfig) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps
    config.face.model_asset_path = model_path

    snr_values: list[float] = []
    frame_idx = 0
    with RPPGPipeline(config, debug=True) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None and result.debug is not None:
                snr_values.append(result.debug.sqi.spectral_snr_db)
            frame_idx += 1
    cap.release()

    finite = np.array([v for v in snr_values if np.isfinite(v)])
    if len(finite) == 0:
        return float("nan")
    return float(np.median(finite))


def _variant_label(mode: str, value: float) -> str:
    return f"fixed={value:.2f}Hz" if mode == "fixed" else f"adaptive k={value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS, help="Пути к видео (по умолчанию — 6 видео из run_benchmark.DEFAULT_VIDEOS)")
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        raise SystemExit(
            "Видео не найдены:\n" + "\n".join(f"  {v}" for v in missing) +
            "\n\nПередайте реальные пути через --videos."
        )

    variants: list[tuple[str, float]] = (
        [("fixed", v) for v in FIXED_CANDIDATES_HZ] + [("adaptive", k) for k in ADAPTIVE_K_CANDIDATES]
    )

    # results[variant_label][video_name] = median spectral_snr_db
    results: dict[str, dict[str, float]] = {}
    for mode, value in variants:
        label = _variant_label(mode, value)
        results[label] = {}
        for video in videos:
            config = PipelineConfig()
            config.quality.snr_peak_half_width_mode = mode
            if mode == "fixed":
                config.quality.snr_peak_half_width_hz = value
            else:
                config.quality.snr_peak_half_width_k = value
            median_snr = _median_snr_for_video(str(video), args.model, config)
            results[label][video.stem] = median_snr
            print(f"  [{label:<16s}] {video.stem:<40s} медиана spectral_snr_db = {median_snr:.2f} dB")

    video_names = [v.stem for v in videos]
    header = f"{'вариант':<16s}" + "".join(f"{name[:18]:>20s}" for name in video_names) + f"{'разброс (max-min)':>20s}"
    print("\n=== Сводная таблица: полуширина x видео -> медиана spectral_snr_db ===")
    print(header)
    for mode, value in variants:
        label = _variant_label(mode, value)
        row_values = [results[label][name] for name in video_names]
        finite = [v for v in row_values if np.isfinite(v)]
        spread = (max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
        row = f"{label:<16s}" + "".join(f"{v:20.2f}" for v in row_values) + f"{spread:20.2f}"
        print(row)

    print(
        "\nВыбирайте вариант по разделяющей способности (разброс медиан между видео), "
        "а не по тому, у кого SNR выше в среднем — расширение полосы почти всегда "
        "поднимает SNR, в т.ч. на шуме (см. докстринг модуля)."
    )


if __name__ == "__main__":
    main()
