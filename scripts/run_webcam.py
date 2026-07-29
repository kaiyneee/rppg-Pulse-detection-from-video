"""
Запуск пайплайна с веб-камеры в реальном времени.

Использование:
    python scripts/run_webcam.py --camera 0 --method pos --freq welch
    python scripts/run_webcam.py --camera 0 --log session.jsonl   # см. п.43

Перед первым запуском один раз скачайте модель Face Landmarker (версия
зафиксирована — п.41 требований, см. src/rppg/face/landmarker.py::MODEL_URL):
    mkdir -p models
    curl -L -o models/face_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.config_io import load_config
from rppg.pipeline import RPPGPipeline


def draw_overlay(frame, result) -> None:
    if result is None:
        cv2.putText(frame, "Buffering...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return

    color = (0, 200, 0) if result.publishable else (0, 165, 255)
    bpm_text = f"BPM: {result.bpm:.1f}" if not (result.bpm != result.bpm) else "BPM: --"
    cv2.putText(frame, bpm_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"SQI: {result.sqi_level} ({result.sqi_score:.2f})", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"method: {result.method_used}/{result.frequency_method_used}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if result.hrv is not None and result.publishable:
        cv2.putText(frame, f"SDNN: {result.hrv.sdnn_ms:.0f}ms  RMSSD: {result.hrv.rmssd_ms:.0f}ms  "
                            f"pNN50: {result.hrv.pnn50_pct:.0f}%",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    if not result.publishable:
        cv2.putText(frame, "LOW QUALITY - not sent to PTSD pipeline", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="rPPG realtime demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--config", type=str, default=None, help="Базовый YAML/JSON-конфиг, см. rppg.config_io")
    parser.add_argument("--method", type=str, default=None, choices=[m.value for m in ExtractionMethod])
    parser.add_argument("--freq", type=str, default=None, choices=[m.value for m in FrequencyMethod])
    parser.add_argument(
        "--log", type=str, default=None,
        help="Путь к JSONL-логу по окнам (timestamp/BPM по ROI/компоненты SQI/warnings, см. п.43)",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else PipelineConfig()
    config.face.model_asset_path = args.model
    if args.method is not None:
        config.method = ExtractionMethod(args.method)
    if args.freq is not None:
        config.frequency_method = FrequencyMethod(args.freq)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {args.camera}")

    start_ms = time.time() * 1000

    with RPPGPipeline(config, log_path=args.log) as pipeline:
        last_result = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(time.time() * 1000 - start_ms)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                last_result = result
                if result.warnings:
                    print(f"[{timestamp_ms/1000:.1f}s] " + " | ".join(result.warnings))

            draw_overlay(frame, last_result)
            cv2.imshow("rPPG (ESC to quit)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
