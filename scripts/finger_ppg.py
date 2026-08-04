"""
Задача 3б: дешёвый локальный референс — контактный PPG с пальца.

Идея: палец, плотно прижатый к объективу телефона с включённым фонариком/
вспышкой, модулирует проходящий/отражённый свет пульсацией кровотока
напрямую (транссветовая фотоплетизмография — тот же физический принцип,
что и в пульсоксиметре), а не через отражение от кожи лица на расстоянии.
SNR такого сигнала обычно на порядок выше лицевого rPPG, а полученную
кривую BPM можно использовать как почти бесплатный референс "на себе" для
сравнения с результатом лицевого пайплайна (см. scripts/record_with_reference.py
и критерий приёмки задачи 3б).

ВАЖНО: этот файл ПЕРЕИСПОЛЬЗУЕТ тот же сигнальный код, что и лицевой
пайплайн (rppg.signal.preprocessing/methods/frequency/quality) — единственное
отличие от RPPGPipeline в том, что здесь НЕТ MediaPipe/ROI-детекции лица:
палец, прижатый к объективу, обычно заполняет кадр целиком и сам по себе
уже один большой ROI, поэтому вместо face/roi.py используется тривиальное
"среднее по центральному кропу кадра" (см. extract_frame_rgb).

Использование:
    python scripts/finger_ppg.py --video finger.mp4 --out finger_bpm.csv
    python scripts/finger_ppg.py --video finger.mp4 --method pos --freq welch
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.config import FilterConfig, ExtractionMethod, FrequencyMethod
from rppg.signal.preprocessing import preprocess_signal
from rppg.signal.methods import get_method, SignalWindow
from rppg.signal.frequency import estimate_hr
from rppg.signal.quality import dominant_frequency_and_snr


@dataclass
class FingerBPMSample:
    """Одна оценка BPM по скользящему окну контактного PPG пальца —
    структурно аналог WindowLogRecord лицевого пайплайна (см.
    structured_log.py), но БЕЗ SQI-гейтинга: контактный сигнал по
    построению высокого SNR, и цель этого файла — дать референс, а не
    воспроизвести весь гейтинг лицевого пайплайна."""

    timestamp_ms: int
    bpm: float
    snr_db: float


def extract_frame_rgb(frame_bgr: np.ndarray, center_crop_fraction: float = 0.6) -> np.ndarray:
    """Среднее R,G,B по ЦЕНТРАЛЬНОМУ кропу кадра, а не по кадру целиком —
    палец, прижатый к объективу, обычно заполняет кадр почти полностью, но
    края кадра у многих камер виньетируются или расфокусированы вплотную к
    объективу; центральный кроп безопаснее и не требует НИКАКОЙ ROI-детекции
    (в отличие от лицевого пайплайна, здесь она в принципе не нужна — палец
    и так один большой ROI)."""
    h, w = frame_bgr.shape[:2]
    ch, cw = max(1, int(h * center_crop_fraction)), max(1, int(w * center_crop_fraction))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = frame_bgr[y0:y0 + ch, x0:x0 + cw]
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    return mean_bgr[::-1].astype(np.float64)  # BGR -> RGB


def process_finger_video(
    video_path: str | Path,
    method_name: str = "green",
    frequency_method: str = "welch",
    window_seconds: float = 10.0,
    step_seconds: float = 1.0,
    band_hz: tuple[float, float] = (0.7, 4.0),
    filt: FilterConfig | None = None,
    center_crop_fraction: float = 0.6,
) -> list[FingerBPMSample]:
    """Скользящее окно по видео пальца -> список оценок BPM во времени.

    Структура намеренно повторяет RPPGPipeline._compute_estimate (буфер по
    РЕАЛЬНОМУ времени, тот же preprocess_signal/get_method/estimate_hr), но
    сильно упрощена: один "ROI" (весь кадр/центральный кроп), НЕТ
    occlusion-гейта (valid_mask всегда True — на контактном видео пальца
    нет такого явления, как MediaPipe не нашёл лицо), НЕТ SQI (см. докстринг
    класса FingerBPMSample)."""
    filt = filt or FilterConfig()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    method = get_method(method_name)
    rgb_buffer: deque = deque()
    ts_buffer_ms: deque = deque()
    samples: list[FingerBPMSample] = []
    last_estimate_ms: float | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = frame_idx / fps * 1000.0
        rgb_buffer.append(extract_frame_rgb(frame, center_crop_fraction))
        ts_buffer_ms.append(ts_ms)

        # Обрезаем буфер по РЕАЛЬНОМУ времени (не по числу кадров) — тот же
        # инвариант, что и _RingBuffer.trim_older_than в pipeline.py, чтобы
        # длина окна в секундах не зависела от фактического fps видео.
        cutoff = ts_ms - window_seconds * 1000.0
        while ts_buffer_ms and ts_buffer_ms[0] < cutoff:
            ts_buffer_ms.popleft()
            rgb_buffer.popleft()

        elapsed_s = (ts_buffer_ms[-1] - ts_buffer_ms[0]) / 1000.0 if len(ts_buffer_ms) > 1 else 0.0
        frame_idx += 1
        if elapsed_s < window_seconds:
            continue  # окно ещё не заполнено — см. задачу 6, тот же принцип: не оценивать по неполному окну
        if last_estimate_ms is not None and ts_ms - last_estimate_ms < step_seconds * 1000.0:
            continue
        last_estimate_ms = ts_ms

        rgb = np.array(rgb_buffer)
        valid_mask = np.ones(len(rgb), dtype=bool)
        window = SignalWindow(rgb_traces={"finger": rgb}, fps=fps, valid_mask=valid_mask, hr_band_hz=band_hz)
        raw_signal = method.extract(window, "finger")
        processed = preprocess_signal(
            raw_signal, fps, band_hz[0], band_hz[1],
            order=filt.filter_order, detrend_method=filt.detrend_method,
            tarvainen_lambda=filt.tarvainen_lambda, normalize_method=filt.normalize_method,
        )
        est = estimate_hr(processed, fps, band_hz, method=frequency_method)
        _, snr_db = dominant_frequency_and_snr(processed, fps, band_hz)
        samples.append(FingerBPMSample(timestamp_ms=int(ts_ms), bpm=est.bpm, snr_db=snr_db))

    cap.release()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=str, required=True, help="Видео пальца, прижатого к объективу+вспышке")
    parser.add_argument("--out", type=str, default=None, help="Путь к CSV (по умолчанию <video>_finger_bpm.csv)")
    parser.add_argument("--method", type=str, default="green", choices=[m.value for m in ExtractionMethod if m.value != "head_motion" and m.value != "auto"])
    parser.add_argument("--freq", type=str, default="welch", choices=[m.value for m in FrequencyMethod])
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--low-hz", type=float, default=0.7)
    parser.add_argument("--high-hz", type=float, default=4.0)
    args = parser.parse_args()

    samples = process_finger_video(
        args.video, method_name=args.method, frequency_method=args.freq,
        window_seconds=args.window_seconds, step_seconds=args.step_seconds,
        band_hz=(args.low_hz, args.high_hz),
    )
    if not samples:
        raise SystemExit(f"Не удалось посчитать ни одной оценки — видео короче {args.window_seconds}с?")

    out_path = Path(args.out) if args.out else Path(args.video).with_name(Path(args.video).stem + "_finger_bpm.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "bpm", "snr_db"])
        for s in samples:
            writer.writerow([s.timestamp_ms, f"{s.bpm:.2f}", f"{s.snr_db:.2f}"])

    bpm_values = np.array([s.bpm for s in samples])
    snr_values = np.array([s.snr_db for s in samples])
    print(f"Оценок: {len(samples)}")
    print(f"BPM: median={np.median(bpm_values):.1f}, std={np.std(bpm_values):.2f}")
    print(f"spectral_snr_db: median={np.median(snr_values):.1f} dB (для сравнения с лицевым — см. scripts/analyze_log.py)")
    print(f"Сохранено в {out_path}")


if __name__ == "__main__":
    main()
