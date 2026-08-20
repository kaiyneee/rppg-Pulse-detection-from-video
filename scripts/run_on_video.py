"""
Офлайн-обработка видеофайла: извлекает временной ряд BPM/HRV/SQI и
сохраняет в CSV — удобно для последующей подачи в систему анализа ПТСР
или для построения графиков (см. benchmark/evaluate.py для сравнения
с референсным сигналом при наличии ground truth).

Конфиг эксперимента сохраняется РЯДОМ с результатом (--out result.csv ->
result.config.yaml) — п.42 требований: "каждая строчка таблицы в статье
должна быть привязана к конкретному конфигу", а не к неявному состоянию
кода/CLI-флагов на момент запуска.

Использование:
    python scripts/run_on_video.py --video path/to/video.mp4 --out result.csv
    python scripts/run_on_video.py --video path/to/video.mp4 --config my_experiment.yaml --out result.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.config_io import load_config, save_config
from rppg.pipeline import RPPGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн rPPG обработка видео")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Базовый YAML/JSON-конфиг (см. rppg.config_io) — --method/--freq, если заданы, переопределяют его.",
    )
    parser.add_argument("--method", type=str, default=None, choices=[m.value for m in ExtractionMethod])
    parser.add_argument("--freq", type=str, default=None, choices=[m.value for m in FrequencyMethod])
    parser.add_argument("--out", type=str, default="rppg_result.csv")
    parser.add_argument(
        "--log", type=str, default=None,
        help="Путь к JSONL-логу по окнам (timestamp/BPM по ROI/компоненты SQI/warnings, см. п.43)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Дозаписать в существующий --log вместо перезаписи (задача 26; по умолчанию файл "
             "перезаписывается — прошлый баг молча склеивал записи от разных прогонов/конфигов).",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else PipelineConfig()
    config.face.model_asset_path = args.model
    if args.method is not None:
        config.method = ExtractionMethod(args.method)
    if args.freq is not None:
        config.frequency_method = FrequencyMethod(args.freq)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps

    rows = []
    frame_idx = 0

    with RPPGPipeline(config, log_path=args.log, log_append=args.append) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                rows.append(result)
            frame_idx += 1

    cap.release()

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms", "bpm", "status",
            "sqi_score", "sqi_level", "sqi_spectral_snr_db",
            # Задача 25 (пересмотрено 2026-08-18): контракт с моделью ПТСР —
            # РОВНО ДВЕ колонки, не пять. bpm_aggregate — всегда реально
            # измеренное число (или NaN); bpm_aggregate_quality — одно
            # число [0,1], 0.0 означает "использовать нельзя". Устаревание
            # агрегат берёт на себя через ПЛАВНО убывающее quality (см.
            # RPPGPipeline._compute_aggregate) — отдельные last_valid_bpm/
            # last_valid_bpm_age_s/publishable для этого больше не нужны и
            # убраны из CSV (они дублировали более старый, ступенчатый
            # механизм устаревания — задача 8). window_status/n_windows/
            # spread_bpm, стоявшие за bpm_aggregate, остаются ТОЛЬКО в
            # JSONL-логе (см. structured_log.WindowLogRecord) — это
            # диагностика для отладки, не часть контракта потребителя.
            "bpm_aggregate", "bpm_aggregate_quality",
            "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "pnn20_pct",
            "lf_hf_ratio", "method_used", "frequency_method_used", "warnings",
        ])
        for r in rows:
            hrv = r.hrv
            writer.writerow([
                r.timestamp_ms, r.bpm, r.status,
                f"{r.sqi_score:.3f}", r.sqi_level, r.confidence_breakdown.get("spectral_snr_db", ""),
                r.bpm_aggregate, f"{r.bpm_aggregate_quality:.3f}",
                hrv.mean_hr_bpm if hrv else "", hrv.sdnn_ms if hrv else "",
                hrv.rmssd_ms if hrv else "", hrv.pnn50_pct if hrv else "",
                hrv.pnn20_pct if hrv else "", hrv.lf_hf_ratio if hrv else "",
                r.method_used, r.frequency_method_used, " ; ".join(r.warnings),
            ])

    config_out_path = out_path.with_suffix("").with_suffix(".config.yaml")
    save_config(config, config_out_path)

    n_publishable = sum(1 for r in rows if r.publishable)
    print(f"Обработано окон: {len(rows)}, из них publishable (SQI ok): {n_publishable}")
    print(f"Результат сохранён в {out_path}")
    print(f"Конфиг эксперимента сохранён в {config_out_path}")


if __name__ == "__main__":
    main()
