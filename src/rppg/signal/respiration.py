"""
Оценка частоты дыхания по амплитудной модуляции rPPG-сигнала
(Respiratory-Induced Amplitude Variation, RIAV) — пункт 18 требований.

Зачем это нужно:

  (а) бесплатный дополнительный признак для ПТСР-системы отдельно от
      HR/HRV — частота и вариабельность дыхания сами по себе релевантны
      физиологии острого стрессового ответа;
  (б) необходимый методологический контроль для HF-компоненты HRV: HF-
      мощность по построению отражает дыхательную модуляцию сердечного
      ритма (respiratory sinus arrhythmia), поэтому её величина зависит от
      частоты дыхания (Task Force, 1996; Grossman & Taylor, 2007) — без
      параллельной оценки частоты дыхания сравнение HF между записями или
      группами методологически некорректно. Именно на этот довод ссылается
      обоснование LF/HF в hrv/features.py (Schneider & Schwerdtfeger, 2020).

Метод: огибающая пульсовой волны (модуль аналитического сигнала по
Гильберту) несёт медленную амплитудную модуляцию на частоте дыхания —
доминирующая частота огибающей в полосе 0.15-0.4 Гц (9-24 дыханий/мин)
даёт оценку частоты дыхания напрямую из уже вычисленного пульсового
сигнала, без отдельного канала/сенсора.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch


def estimate_respiration_rate(
    pulse_signal: np.ndarray,
    fps: float,
    resp_band_hz: tuple[float, float] = (0.15, 0.4),
    min_seconds: float = 10.0,
) -> tuple[float | None, float | None]:
    """Возвращает (частота_дыхания_в_минуту, частота_Hz) или (None, None).

    None, если сигнал короче min_seconds: полоса 0.15-0.4 Гц соответствует
    периодам 2.5-6.7с, и для устойчивой локализации пика в спектре огибающей
    нужно хотя бы несколько полных циклов дыхания.
    """
    signal = np.asarray(pulse_signal, dtype=float)
    if len(signal) < int(round(min_seconds * fps)) or np.std(signal) < 1e-10:
        return None, None

    envelope = np.abs(hilbert(signal))
    envelope = envelope - np.mean(envelope)

    nperseg = min(len(envelope), max(64, int(fps * 20)))
    freqs, psd = welch(envelope, fs=fps, nperseg=nperseg)

    mask = (freqs >= resp_band_hz[0]) & (freqs <= resp_band_hz[1])
    if not mask.any():
        return None, None

    band_freqs, band_psd = freqs[mask], psd[mask]
    peak_freq = float(band_freqs[np.argmax(band_psd)])
    return peak_freq * 60.0, peak_freq
