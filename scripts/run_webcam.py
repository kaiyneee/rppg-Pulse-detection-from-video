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
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.config_io import load_config
from rppg.pipeline import RPPGPipeline

# Сколько последних ПУБЛИКУЕМЫХ (SQI ok) оценок усредняем для отображаемого
# числа. Даже среди publishable=True окон отдельные оценки могут скакать —
# медиана нескольких последних сглаживает это для живого демо, не трогая
# сам pipeline/SQI (это чисто отображение, см. draw_overlay).
DISPLAY_SMOOTHING_WINDOW = 5
# Если дольше этого не было ни одной publishable-оценки — считаем
# сглаженное число устаревшим и не показываем его (человек мог отойти,
# сменить освещение и т.п.), а не морозим старое значение на экране.
STALE_AFTER_MS = 10_000


def draw_overlay(frame, result, smoothed_bpm: float | None) -> None:
    if result is None:
        cv2.putText(frame, "Buffering...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return

    # ВАЖНО: показываем число ТОЛЬКО когда SQI ему доверяет (smoothed_bpm
    # приходит из окон с publishable=True, см. main()). Раньше здесь
    # показывался result.bpm ЛЮБОГО окна (просто другим цветом, если
    # publishable=False) — визуально выглядело как "пульс скачет 50<->150",
    # хотя на самом деле система КОРРЕКТНО не доверяла этим числам, просто
    # всё равно их печатала. Не публикуемые, но реально существующие числа
    # ещё сильнее шумят, чем можно было бы подумать по (относительно
    # спокойному) SQI-скору — сам SQI не гарантирует, что НЕОПУБЛИКОВАННОЕ
    # число близко к истине, гарантия действует только для publishable=True.
    if smoothed_bpm is not None:
        cv2.putText(frame, f"BPM: {smoothed_bpm:.0f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "Измеряю пульс... (нужно больше света, лицо ближе, поменьше движения)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    color = (0, 200, 0) if result.publishable else (0, 165, 255)
    cv2.putText(frame, f"SQI: {result.sqi_level} ({result.sqi_score:.2f})", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"method: {result.method_used}/{result.frequency_method_used}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # best_effort_bpm — НЕ валидированное измерение (см. BestEffortConfig в
    # config.py): всегда реальное число в правдоподобном диапазоне,
    # ограниченное по скорости изменения, для сравнения с "доверенным"
    # BPM выше. Показано мелко и отдельным цветом, чтобы не подменять
    # собой честный зелёный/оранжевый индикатор.
    cv2.putText(frame, f"auxiliary (always-on): {result.best_effort_bpm:.0f}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 1)

    if result.hrv is not None and result.publishable:
        cv2.putText(frame, f"SDNN: {result.hrv.sdnn_ms:.0f}ms  RMSSD: {result.hrv.rmssd_ms:.0f}ms  "
                            f"pNN50: {result.hrv.pnn50_pct:.0f}%",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    if not result.publishable:
        cv2.putText(frame, "LOW QUALITY - not sent to PTSD pipeline", (20, 155),
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

    published_bpm_history: deque = deque(maxlen=DISPLAY_SMOOTHING_WINDOW)
    last_published_ms: int | None = None

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
                if result.publishable:
                    published_bpm_history.append(result.bpm)
                    last_published_ms = timestamp_ms
                if result.warnings:
                    print(f"[{timestamp_ms/1000:.1f}s] " + " | ".join(result.warnings))

            is_stale = last_published_ms is None or timestamp_ms - last_published_ms > STALE_AFTER_MS
            smoothed_bpm = None if is_stale or not published_bpm_history else float(np.median(published_bpm_history))

            draw_overlay(frame, last_result, smoothed_bpm)
            cv2.imshow("rPPG (ESC to quit)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
