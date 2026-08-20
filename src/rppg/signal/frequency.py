"""
Оценка частоты пульса (пункт "Анализ частоты" ТЗ): FFT, Welch PSD,
Lomb-Scargle, Harmonic Sum.

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
* Harmonic Sum — НЕ метод по умолчанию (см. PipelineConfig.frequency_method,
  включать после сравнения scripts/compare_frequency_methods.py): ищет пик
  не по самой мощной точке PSD, а по сумме мощностей на f и её первых
  гармониках (см. estimate_hr_harmonic_sum) — устойчивее к тому, что
  случайный шумовой/двигательный пик на 2f/f/2 окажется выше истинного пика
  ровно на f.

Выбор ПИКА (не только МЕТОДА расчёта спектра) может дополнительно учитывать
непрерывность оценки во времени — см. _select_peak_with_continuity и
FrequencyConfig.continuity_* — если в рабочей полосе несколько локальных
максимумов сопоставимой мощности, предпочитается тот, что ближе к недавним
оценкам, а не автоматически глобальный максимум.
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


def _select_peak_with_continuity(
    freqs: np.ndarray,
    power: np.ndarray,
    recent_median_hz: float | None,
    continuity_power_tolerance: float,
) -> int:
    """Индекс выбранного пика в (freqs, power) — обычный argmax, ЕСЛИ:
    recent_median_hz не задан, либо в полосе нет альтернативного локального
    максимума сопоставимой мощности. В противном случае — среди локальных
    максимумов, чья мощность отличается от глобального не больше чем на
    continuity_power_tolerance (масштабно-независимая относительная доля,
    см. FrequencyConfig), выбирается ближайший к recent_median_hz.

    ВАЖНО: это НЕ подмена измерения историей — кандидат ОБЯЗАН реально
    существовать как локальный максимум мощности в ЭТОМ окне; при
    единственном локальном максимуме (типичный случай для чистого сигнала)
    поведение НЕ отличается от обычного argmax."""
    power = np.asarray(power, dtype=float)
    if len(power) < 3 or recent_median_hz is None:
        return int(np.argmax(power))

    is_local_max = np.zeros(len(power), dtype=bool)
    is_local_max[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    global_idx = int(np.argmax(power))
    is_local_max[global_idx] = True  # глобальный максимум всегда кандидат, даже на краю сетки

    candidate_idx = np.where(is_local_max)[0]
    if len(candidate_idx) < 2:
        return global_idx

    global_power = float(power[global_idx])
    tolerance_abs = continuity_power_tolerance * abs(global_power)
    within_tolerance = candidate_idx[power[candidate_idx] >= global_power - tolerance_abs]
    if len(within_tolerance) < 2:
        return global_idx

    freq_diffs = np.abs(freqs[within_tolerance] - recent_median_hz)
    return int(within_tolerance[int(np.argmin(freq_diffs))])


def estimate_hr_fft(
    signal: np.ndarray,
    fps: float,
    band_hz: tuple[float, float],
    recent_median_hz: float | None = None,
    continuity_power_tolerance: float = 0.15,
) -> FrequencyEstimate:
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
    peak_idx = _select_peak_with_continuity(band_freqs, band_power, recent_median_hz, continuity_power_tolerance)
    peak_freq = float(band_freqs[peak_idx])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "fft", band_freqs, band_power)


def estimate_hr_welch(
    signal: np.ndarray,
    fps: float,
    band_hz: tuple[float, float],
    recent_median_hz: float | None = None,
    continuity_power_tolerance: float = 0.15,
) -> FrequencyEstimate:
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
    peak_idx = _select_peak_with_continuity(band_freqs, band_psd, recent_median_hz, continuity_power_tolerance)
    peak_freq = float(band_freqs[peak_idx])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "welch", band_freqs, band_psd)


def estimate_hr_lombscargle(
    timestamps_sec: np.ndarray,
    signal: np.ndarray,
    band_hz: tuple[float, float],
    n_freqs: int = 512,
    recent_median_hz: float | None = None,
    continuity_power_tolerance: float = 0.15,
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

    peak_idx = _select_peak_with_continuity(freqs_hz, power, recent_median_hz, continuity_power_tolerance)
    peak_freq = float(freqs_hz[peak_idx])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "lomb_scargle", freqs_hz, power)


def estimate_hr_harmonic_sum(
    signal: np.ndarray,
    fps: float,
    band_hz: tuple[float, float],
    n_harmonics: int = 3,
    recent_median_hz: float | None = None,
    continuity_power_tolerance: float = 0.15,
) -> FrequencyEstimate:
    """
    Harmonic sum: вместо argmax по самому мощному пику Welch-PSD ищет
    частоту f в рабочей полосе, максимизирующую

        S(f) = Σ_{k=1..n_harmonics} log P(k·f)

    — сумму (в ЛОГ-масштабе) мощности на f И на её первых гармониках 2f,
    3f, ... Пульсовая волна физиологически несинусоидальна (резкий
    систолический подъём, пологий диастолический спад — см. также
    quality.include_second_harmonic_in_snr): реальная частота пульса несёт
    согласованную энергию не только на f, но и на кратных, а классическая
    ошибка "argmax по одному пику" (случайный шумовой/двигательный пик на
    2f или f/2 оказался выше истинного пика на f) устойчивее ловится
    суммой по нескольким гармоникам сразу — ложный пик редко имеет
    согласованные гармоники именно на СВОИХ кратных.

    Лог, а не произведение мощностей (классический harmonic product
    spectrum): произведение обнуляется целиком, если хотя бы ОДНА
    гармоника провалилась в почти нулевую мощность (провал/шум между
    сегментами PSD) — сумма логарифмов эквивалентна произведению по
    определению (log(a·b) = log a + log b), но один слабый гармоник
    штрафует S(f) АДДИТИВНО, а не обнуляет весь продукт.

    n_harmonics=3 по умолчанию (f, 2f, 3f): дальше третьей гармоники
    энергия несинусоидальной пульсовой волны в норме уже пренебрежимо
    мала. Гармоника k·f, лежащая ВЫШЕ частоты Найквиста (fps/2) физически
    недоступна в PSD — не участвует в сумме ДЛЯ ТЕХ f, для которых она
    недоступна (не обрезаем весь диапазон кандидатов f сразу по самой
    строгой гармонике k=n_harmonics, чтобы не терять k=1 там, где k=3 уже
    недоступна): для такого f её вклад в сумму просто 0 (нейтрально, не
    штраф и не бонус), а не NaN/ошибка.
    """
    signal = np.asarray(signal, dtype=float)
    nperseg = min(len(signal), max(64, int(fps * 8)))
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft)

    mask = _band_mask(freqs, band_hz)
    if not mask.any():
        return FrequencyEstimate(np.nan, np.nan, "harmonic_sum", freqs, psd)

    band_freqs = freqs[mask]
    nyquist = fps / 2.0
    log_psd = np.log(psd + 1e-15)  # eps: PSD теоретически строго > 0, но защищаемся от log(0)

    harmonic_sum = np.zeros_like(band_freqs)
    for k in range(1, n_harmonics + 1):
        harmonic_freqs = k * band_freqs
        available = harmonic_freqs <= nyquist
        contribution = np.interp(harmonic_freqs, freqs, log_psd)
        harmonic_sum = harmonic_sum + np.where(available, contribution, 0.0)

    peak_idx = _select_peak_with_continuity(band_freqs, harmonic_sum, recent_median_hz, continuity_power_tolerance)
    peak_freq = float(band_freqs[peak_idx])
    return FrequencyEstimate(peak_freq * 60.0, peak_freq, "harmonic_sum", band_freqs, harmonic_sum)


ESTIMATORS = {
    "fft": estimate_hr_fft,
    "welch": estimate_hr_welch,
    "lomb_scargle": estimate_hr_lombscargle,
    "harmonic_sum": estimate_hr_harmonic_sum,
}


def estimate_hr(
    signal: np.ndarray,
    fps: float,
    band_hz: tuple[float, float],
    method: str = "welch",
    timestamps_sec: np.ndarray | None = None,
    recent_median_hz: float | None = None,
    continuity_power_tolerance: float = 0.15,
) -> FrequencyEstimate:
    if method == "lomb_scargle":
        if timestamps_sec is None:
            timestamps_sec = np.arange(len(signal)) / fps
        return estimate_hr_lombscargle(
            timestamps_sec, signal, band_hz,
            recent_median_hz=recent_median_hz, continuity_power_tolerance=continuity_power_tolerance,
        )
    if method == "fft":
        return estimate_hr_fft(
            signal, fps, band_hz,
            recent_median_hz=recent_median_hz, continuity_power_tolerance=continuity_power_tolerance,
        )
    if method == "welch":
        return estimate_hr_welch(
            signal, fps, band_hz,
            recent_median_hz=recent_median_hz, continuity_power_tolerance=continuity_power_tolerance,
        )
    if method == "harmonic_sum":
        return estimate_hr_harmonic_sum(
            signal, fps, band_hz,
            recent_median_hz=recent_median_hz, continuity_power_tolerance=continuity_power_tolerance,
        )
    raise ValueError(f"Unknown frequency method: {method}")
