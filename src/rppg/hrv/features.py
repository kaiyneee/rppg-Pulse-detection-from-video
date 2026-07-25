"""
Признаки для ПТСР-системы (пункт "Признаки для ПТСР" ТЗ):
Mean HR, HRV (сводно), RMSSD, SDNN, pNN50, Pulse Signal Quality Score.

ВАЖНАЯ МЕТОДОЛОГИЧЕСКАЯ ОГОВОРКА (перенесена из docs/research_report.md,
раздел 7, чтобы быть видимой прямо в коде, а не только в тексте статьи):

    Показатели вычисляются из межпульсовых интервалов (Pulse-to-Pulse
    Intervals, PPI), а не из R-R интервалов ЭКГ. В литературе это принято
    называть Pulse Rate Variability (PRV). PRV — хорошая, но не идеальная
    аппроксимация классической HRV: у здоровых людей в покое согласие с
    ЭКГ высокое, но расходится при движении, сосудистых нарушениях и
    аритмиях (Schäfer & Vagedes, 2013, Int. J. Cardiology — обзор точности
    PRV относительно ЭКГ). При заявке точности в статье об этом необходимо
    явно писать и, где возможно, валидировать против синхронной ЭКГ/PPG-
    референса (см. экспериментальный протокол).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, welch


@dataclass
class HRVFeatures:
    mean_hr_bpm: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float
    pnn20_pct: float
    mean_ibi_ms: float
    cv_ibi_pct: float
    n_beats: int
    lf_power: float | None = None
    hf_power: float | None = None
    lf_hf_ratio: float | None = None
    pulse_signal_quality_score: float = 0.0
    warnings: list[str] = field(default_factory=list)


def detect_pulse_peaks(
    signal: np.ndarray, fps: float, min_hr_bpm: float = 40.0, max_hr_bpm: float = 220.0
) -> np.ndarray:
    """Детекция систолических пиков пульсовой волны.

    distance/prominence подобраны из физиологических границ ЧСС, а не
    произвольно — так минимизируются как пропуски слабых пиков, так и
    ложные срабатывания на дикротической выемке пульсовой волны.
    """
    signal = np.asarray(signal, dtype=float)
    min_distance = int(round(fps * 60.0 / max_hr_bpm))
    prominence = 0.25 * np.std(signal) if np.std(signal) > 1e-8 else None

    peaks, _ = find_peaks(signal, distance=max(1, min_distance), prominence=prominence)
    return peaks


def compute_ibi_ms(peak_indices: np.ndarray, fps: float) -> np.ndarray:
    if len(peak_indices) < 2:
        return np.array([])
    return np.diff(peak_indices) / fps * 1000.0


def _remove_ectopic_like_outliers(ibi_ms: np.ndarray, max_relative_change: float = 0.3) -> np.ndarray:
    """Простой фильтр физиологически неправдоподобных скачков IBI
    (аналог "artifact correction" в HRV-литературе, напр. Kubios): скачок
    интервала более чем на max_relative_change от соседнего почти всегда
    означает пропущенный/ложный пик детектора, а не реальную аритмию —
    для промышленной ЭКГ такие случаи размечают отдельно, но для
    rPPG-прототипа безопаснее консервативно отбросить выброс.
    """
    if len(ibi_ms) < 3:
        return ibi_ms
    keep = np.ones(len(ibi_ms), dtype=bool)
    for i in range(1, len(ibi_ms) - 1):
        neighborhood = (ibi_ms[i - 1] + ibi_ms[i + 1]) / 2.0
        if neighborhood > 0 and abs(ibi_ms[i] - neighborhood) / neighborhood > max_relative_change:
            keep[i] = False
    return ibi_ms[keep]


def hrv_time_domain(
    ibi_ms: np.ndarray, pnn_threshold_ms: float = 50.0, pnn20_threshold_ms: float = 20.0
) -> dict:
    if len(ibi_ms) < 2:
        return dict(
            mean_hr_bpm=np.nan, sdnn_ms=np.nan, rmssd_ms=np.nan,
            pnn50_pct=np.nan, pnn20_pct=np.nan, mean_ibi_ms=np.nan,
            cv_ibi_pct=np.nan, n_beats=len(ibi_ms) + 1 if len(ibi_ms) else 0,
        )

    diffs = np.diff(ibi_ms)
    mean_ibi = float(np.mean(ibi_ms))
    sdnn = float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > pnn_threshold_ms) * 100) if len(diffs) > 0 else 0.0
    pnn20 = float(np.mean(np.abs(diffs) > pnn20_threshold_ms) * 100) if len(diffs) > 0 else 0.0
    cv_ibi = float(sdnn / mean_ibi * 100) if mean_ibi > 0 else 0.0
    mean_hr = float(60000.0 / mean_ibi) if mean_ibi > 0 else np.nan

    return dict(
        mean_hr_bpm=mean_hr, sdnn_ms=sdnn, rmssd_ms=rmssd,
        pnn50_pct=pnn50, pnn20_pct=pnn20, mean_ibi_ms=mean_ibi,
        cv_ibi_pct=cv_ibi, n_beats=len(ibi_ms) + 1,
    )


def hrv_frequency_domain(ibi_ms: np.ndarray, resample_hz: float = 4.0) -> dict:
    """
    LF/HF — бонусные признаки сверх обязательного списка ТЗ, добавлены
    потому что именно LF/HF (баланс симпатической/парасимпатической
    активности) — один из показателей с наиболее устойчивой ассоциацией с
    ПТСР в мета-анализе Schneider & Schwerdtfeger (2020, Psychological
    Medicine): более высокое отношение LF/HF в группе ПТСР по сравнению с
    контролем. См. docs/research_report.md, раздел 7.

    Требует равномерной передискретизации ряда IBI (сам ряд IBI по
    построению неравномерен по времени — интервал между "отсчётами"
    равен самому IBI).
    """
    if len(ibi_ms) < 4:
        return dict(lf_power=None, hf_power=None, lf_hf_ratio=None)

    beat_times_s = np.cumsum(ibi_ms) / 1000.0
    beat_times_s -= beat_times_s[0]
    ibi_centered = ibi_ms - np.mean(ibi_ms)

    duration = beat_times_s[-1]
    if duration < 20:  # частотный HRV на коротких окнах статистически ненадёжен
        return dict(lf_power=None, hf_power=None, lf_hf_ratio=None)

    uniform_t = np.arange(0, duration, 1.0 / resample_hz)
    uniform_ibi = np.interp(uniform_t, beat_times_s, ibi_centered)

    freqs, psd = welch(uniform_ibi, fs=resample_hz, nperseg=min(len(uniform_ibi), 256))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-8 else None

    return dict(lf_power=lf_power, hf_power=hf_power, lf_hf_ratio=lf_hf_ratio)


def extract_hrv_features(
    pulse_signal: np.ndarray,
    fps: float,
    pulse_signal_quality_score: float = 0.0,
    pnn_threshold_ms: float = 50.0,
    pnn20_threshold_ms: float = 20.0,
    compute_frequency_domain: bool = True,
) -> HRVFeatures:
    warnings: list[str] = []

    peaks = detect_pulse_peaks(pulse_signal, fps)
    ibi_ms = compute_ibi_ms(peaks, fps)
    ibi_clean = _remove_ectopic_like_outliers(ibi_ms)

    if len(ibi_clean) < len(ibi_ms):
        warnings.append(
            f"Удалено {len(ibi_ms) - len(ibi_clean)} физиологически неправдоподобных "
            "межпульсовых интервалов (вероятный артефакт детектора пиков)."
        )
    if len(peaks) < 3:
        warnings.append("Слишком мало обнаруженных пульсовых пиков для надёжного HRV.")

    td = hrv_time_domain(ibi_clean, pnn_threshold_ms, pnn20_threshold_ms)

    fd = dict(lf_power=None, hf_power=None, lf_hf_ratio=None)
    if compute_frequency_domain:
        fd = hrv_frequency_domain(ibi_clean)

    return HRVFeatures(
        mean_hr_bpm=td["mean_hr_bpm"],
        sdnn_ms=td["sdnn_ms"],
        rmssd_ms=td["rmssd_ms"],
        pnn50_pct=td["pnn50_pct"],
        pnn20_pct=td["pnn20_pct"],
        mean_ibi_ms=td["mean_ibi_ms"],
        cv_ibi_pct=td["cv_ibi_pct"],
        n_beats=td["n_beats"],
        lf_power=fd["lf_power"],
        hf_power=fd["hf_power"],
        lf_hf_ratio=fd["lf_hf_ratio"],
        pulse_signal_quality_score=pulse_signal_quality_score,
        warnings=warnings,
    )
