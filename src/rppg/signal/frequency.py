"""
Оценка частоты пульса (пункт "Анализ частоты" ТЗ): FFT, Welch PSD, Lomb-Scargle.

Обоснование выбора метода по умолчанию (Welch) — см. docs/research_report.md,
раздел 4.5. Коротко о компромиссах:

* FFT на всём окне — самый простой и "нервный" вариант: единственная
  реализация без усреднения, высокая дисперсия оценки пика при коротких
  окнах и шуме. Это то, что использовал оригинальный репозиторий.
* Welch PSD — усредняет спектр по перекрывающимся сегментам окна, снижая
  дисперсию оценки пика ценой частотного разрешения. Для окон 8-15 секунд
  на 25-30 fps это выгодный компромисс, поэтому выбран как метод по
  умолчанию.
* Lomb-Scargle — единственный из трёх методов, корректно определённый на
  НЕРАВНОМЕРНО дискретизированных данных. Это не "ещё один вариант FFT",
  а единственно корректный инструмент в конкретном сценарии: после удаления
  кадров с окклюзией/потерей трекинга (пункт "устойчивость к окклюзии")
  сигнал перестаёт быть равномерно дискретизированным по времени, и
  FFT/Welch формально требуют предварительной интерполяции на равномерную
  сетку (что само по себе вносит артефакты на границах пропусков).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from scipy.signal import lombscargle


@dataclass
class FrequencyEstimate:
    bpm: float
    freq_hz: float
    method: str
    freqs_hz: np.ndarray
    power: np.ndarray


def _band_mask(freqs: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    return (freqs >= band_hz[0]) & (freqs <= band_hz[1])


def estimate_hr_fft(signal: np.ndarray, fps: float, band_hz: tuple[float, float]) -> FrequencyEstimate:
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    # Zero-padding до следующей степени двойкиx4 повышает интерполяционную
    # точность пика (частотное "разрешение отображения", не физическое).
    n_fft = max(1024, int(2 ** np.ceil(np.log2(n))) * 4)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * window, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    mask = _band_mask(freqs, band_hz)
    if not mask.any():
        return FrequencyEstimate(np.nan, np.nan, "fft", freqs, spectrum)

    band_freqs, band_power = freqs[mask], spectrum[mask]
    peak_freq = float(band_freqs[np.argmax(band_power)])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "fft", band_freqs, band_power)


def estimate_hr_welch(signal: np.ndarray, fps: float, band_hz: tuple[float, float]) -> FrequencyEstimate:
    signal = np.asarray(signal, dtype=float)
    nperseg = min(len(signal), max(64, int(fps * 8)))
    # nfft отвязан от nperseg и явно завышен (zero-padding) — nperseg управляет
    # усреднением/дисперсией оценки, а nfft отвечает за разрешение сетки для
    # локализации пика. Без этого шаг сетки может быть ~7-8 BPM при
    # типичных nperseg, что слишком грубо для локализации пика ЧСС.
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft)

    mask = _band_mask(freqs, band_hz)
    if not mask.any():
        return FrequencyEstimate(np.nan, np.nan, "welch", freqs, psd)

    band_freqs, band_psd = freqs[mask], psd[mask]
    peak_freq = float(band_freqs[np.argmax(band_psd)])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "welch", band_freqs, band_psd)


def estimate_hr_lombscargle(
    timestamps_sec: np.ndarray,
    signal: np.ndarray,
    band_hz: tuple[float, float],
    n_freqs: int = 512,
) -> FrequencyEstimate:
    """
    Lomb-Scargle периодограмма для неравномерно дискретизированного сигнала.

    timestamps_sec — реальные метки времени валидных (не интерполированных)
    отсчётов; их количество должно совпадать с len(signal).
    """
    timestamps_sec = np.asarray(timestamps_sec, dtype=float)
    signal = np.asarray(signal, dtype=float) - np.mean(signal)

    freqs_hz = np.linspace(band_hz[0], band_hz[1], n_freqs)
    angular_freqs = 2 * np.pi * freqs_hz
    power = lombscargle(timestamps_sec, signal, angular_freqs, normalize=True)

    peak_freq = float(freqs_hz[np.argmax(power)])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "lomb_scargle", freqs_hz, power)


ESTIMATORS = {
    "fft": estimate_hr_fft,
    "welch": estimate_hr_welch,
    "lomb_scargle": estimate_hr_lombscargle,
}


def estimate_hr(
    signal: np.ndarray,
    fps: float,
    band_hz: tuple[float, float],
    method: str = "welch",
    timestamps_sec: np.ndarray | None = None,
) -> FrequencyEstimate:
    if method == "lomb_scargle":
        if timestamps_sec is None:
            timestamps_sec = np.arange(len(signal)) / fps
        return estimate_hr_lombscargle(timestamps_sec, signal, band_hz)
    if method == "fft":
        return estimate_hr_fft(signal, fps, band_hz)
    if method == "welch":
        return estimate_hr_welch(signal, fps, band_hz)
    raise ValueError(f"Unknown frequency method: {method}")
