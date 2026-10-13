"""
Проверка численной состоятельности наложения мерцания (см. docstring
rppg/degrade.py): выживает ли субразрядная модуляция квантование в uint8.

Проверяются ровно два утверждения из задания:
  1. поверх A2 (кадр с шумом сенсора) модуляция a=0.003 после квантования
     восстанавливается в среднем по ROI — шум работает дизерингом;
  2. поверх A1 (чистый замороженный кадр, без шума) она НЕ восстанавливается
     — квантование уничтожает её целиком.

Запуск:
    PYTHONPATH=src venv/bin/python3 scripts/check_degrade_dither.py --video ~/Downloads/8-gt/vid.avi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.degrade import RecordPrep, FrameDegrader, prepare_record  # noqa: E402


def modulation_amplitude(series: np.ndarray, fps: float, f_hz: float) -> float:
    """Амплитуда гармоники f_hz в ряде, в долях от постоянной составляющей
    (проекция на sin/cos — не argmax спектра: частота известна точно)."""
    t = np.arange(len(series)) / fps
    x = series - series.mean()
    c = 2.0 * np.mean(x * np.cos(2 * np.pi * f_hz * t))
    s = 2.0 * np.mean(x * np.sin(2 * np.pi * f_hz * t))
    return float(np.hypot(c, s) / series.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--amp", type=float, default=0.003)
    ap.add_argument("--freq-hz", type=float, default=1.2)
    args = ap.parse_args()

    video = str(Path(args.video).expanduser())
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    prep = prepare_record(video, "B1f", fps)
    print(f"σ шума по фактическому временно́му СКО записи: {prep.noise_sigma:.3f} ур. яркости")
    face = prep.face_mask[:, :, 0] > 0.5
    print(f"маска лица: {face.sum()} пикселей из {face.size}")

    for label, condition in (("A2 + мерцание (штатный путь)", "B1f"), ("A1 + мерцание (без шума)", "_nodither")):
        if condition == "_nodither":
            # тот же кадр, но БЕЗ шума сенсора — прямая проверка утверждения 2
            prep_nd = RecordPrep(
                median_frame=prep.median_frame, noise_sigma=0.0,
                face_mask=prep.face_mask, n_frames_seen=prep.n_frames_seen,
            )
            deg = FrameDegrader("B1f", "check", fps, prep_nd)
        else:
            deg = FrameDegrader("B1f", "check", fps, prep)
        deg.flicker_amp, deg.flicker_hz = args.amp, args.freq_hz

        means = np.array([deg(prep.median_frame, i)[face].mean() for i in range(args.n_frames)])
        amp = modulation_amplitude(means, fps, args.freq_hz)
        print(f"{label:32s} восстановленная амплитуда {amp * 100:.4f}% (наложено {args.amp * 100:.3f}%)")


if __name__ == "__main__":
    main()
