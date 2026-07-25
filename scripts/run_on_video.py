"""
Офлайн-обработка видеофайла: извлекает временной ряд BPM/HRV/SQI и
сохраняет в CSV — удобно для последующей подачи в систему анализа ПТСР
или для построения графиков (см. benchmark/evaluate.py для сравнения
с референсным сигналом при наличии ground truth).

Использование:
    python scripts/run_on_video.py --video path/to/video.mp4 --out result.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.pipeline import RPPGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн rPPG обработка видео")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--method", type=str, default="pos", choices=[m.value for m in ExtractionMethod])
    parser.add_argument("--freq", type=str, default="welch", choices=[m.value for m in FrequencyMethod])
    parser.add_argument("--out", type=str, default="rppg_result.csv")
    args = parser.parse_args()

    config = PipelineConfig()
    config.face.model_asset_path = args.model
    config.method = ExtractionMethod(args.method)
    config.frequency_method = FrequencyMethod(args.freq)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps

    rows = []
    frame_idx = 0

    with RPPGPipeline(config) as pipeline:
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

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms", "bpm", "publishable", "sqi_score", "sqi_level",
            "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "pnn20_pct",
            "lf_hf_ratio", "method_used", "frequency_method_used", "warnings",
        ])
        for r in rows:
            hrv = r.hrv
            writer.writerow([
                r.timestamp_ms, r.bpm, r.publishable, f"{r.sqi_score:.3f}", r.sqi_level,
                hrv.mean_hr_bpm if hrv else "", hrv.sdnn_ms if hrv else "",
                hrv.rmssd_ms if hrv else "", hrv.pnn50_pct if hrv else "",
                hrv.pnn20_pct if hrv else "", hrv.lf_hf_ratio if hrv else "",
                r.method_used, r.frequency_method_used, " ; ".join(r.warnings),
            ])

    n_publishable = sum(1 for r in rows if r.publishable)
    print(f"Обработано окон: {len(rows)}, из них publishable (SQI ok): {n_publishable}")
    print(f"Результат сохранён в {args.out}")


if __name__ == "__main__":
    main()
