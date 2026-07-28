"""
Signal Quality Index (пункт "Оценка качества сигнала" ТЗ).

Если итоговый score ниже порога — BPM/HRV НЕ передаются в систему ПТСР
(см. pipeline.py::RPPGPipeline._gate_output), а выдаётся предупреждение.
Это прямая реализация требования ТЗ: "не передавать значение BPM в систему
ПТСР [при низком качестве]".

Компоненты индекса:

1. spectral_snr       — насколько энергия сигнала сконцентрирована вокруг
                         предполагаемой частоты пульса (в противовес шуму,
                         размазанному по всей полосе).
2. cross_roi_agreement — независимые оценки BPM с лба/левой/правой щеки
                         должны совпадать; расхождение обычно означает, что
                         один из ROI зашумлён (тень, движение, макияж, окклюзия).
3. landmark_stability  — по вариации позы головы/дрожанию landmark-точек
                         между кадрами (proxy для устойчивости к движению
                         и наличия окклюзии).

Итоговый score — взвешенное среднее трёх нормированных к [0,1] компонент.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

from rppg.config import QualityLevel


def _band_mask(freqs: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    return (freqs >= band_hz[0]) & (freqs <= band_hz[1])


def dominant_frequency_and_snr(
    signal: np.ndarray, fps: float, band_hz: tuple[float, float]
) -> tuple[float, float]:
    """
    Возвращает (доминирующая_частота_Hz, spectral_snr_dB).

    spectral_snr — отношение энергии в узкой полосе (+-0.15 Hz) вокруг пика
    к энергии всей рабочей полосы band_hz, в дБ. Эта же величина используется
    как критерий периодичности для выбора PCA/ICA/head-motion компоненты
    (select_best_component_by_periodicity ниже) — общий инвариант: "хорошая"
    пульсовая компонента должна быть узкополосной.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 8 or np.std(signal) < 1e-10:
        return 0.0, -np.inf

    nperseg = min(len(signal), 256)
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, nfft=nfft)
    mask = _band_mask(freqs, band_hz)
    if not mask.any():
        return 0.0, -np.inf

    band_freqs, band_psd = freqs[mask], psd[mask]
    peak_idx = int(np.argmax(band_psd))
    peak_freq = float(band_freqs[peak_idx])

    narrow_mask = np.abs(band_freqs - peak_freq) <= 0.15
    signal_power = np.sum(band_psd[narrow_mask])
    total_power = np.sum(band_psd)
    noise_power = max(total_power - signal_power, 1e-12)

    snr_db = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else 60.0
    return peak_freq, float(snr_db)


def select_best_component_by_periodicity(
    components: np.ndarray, fps: float, band_hz: tuple[float, float]
) -> int:
    """
    components: (n_components, T). Возвращает индекс компоненты с максимальным
    spectral_snr в рабочей полосе — то есть "наиболее пульсоподобной".

    Это прямое обобщение критерия отбора компоненты из Balakrishnan et al. 2013
    ("choose the component that best corresponds to heartbeats based on its
    temporal frequency spectrum"), переиспользуемое для PCA/ICA(цвет) и
    head-motion(движение) — см. methods.py.
    """
    best_idx = 0
    best_snr = -np.inf
    for i in range(components.shape[0]):
        _, snr = dominant_frequency_and_snr(components[i], fps, band_hz)
        if snr > best_snr:
            best_snr = snr
            best_idx = i
    return best_idx


def cross_roi_agreement_score(bpm_by_roi: dict[str, float], max_diff: float = 8.0) -> float:
    """1.0 = все ROI согласны, 0.0 = расхождение >= max_diff BPM."""
    values = [v for v in bpm_by_roi.values() if v is not None and not np.isnan(v)]
    if len(values) < 2:
        return 0.5  # недостаточно ROI для перекрёстной проверки — нейтральная оценка
    spread = max(values) - min(values)
    return float(np.clip(1.0 - spread / max_diff, 0.0, 1.0))


def landmark_stability_score(landmark_trajectories: np.ndarray | None) -> float:
    """
    Оценка устойчивости позы/трекинга по кадр-к-кадру дрожанию стабильных
    landmark-точек (нос, углы глаз и т.п.). Высокая частота больших скачков
    типична при резких движениях головы или частичной потере трекинга из-за
    окклюзии.
    """
    if landmark_trajectories is None or landmark_trajectories.shape[0] < 3:
        return 0.5
    diffs = np.diff(landmark_trajectories, axis=0)  # (T-1, N, 2)
    frame_jitter = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (T-1,)
    # Нормируем по медиане, чтобы не зависеть от разрешения кадра.
    median_jitter = np.median(frame_jitter) + 1e-6
    relative_jitter = frame_jitter / median_jitter
    instability = np.mean(relative_jitter > 4.0)  # доля "выбросов" движения
    return float(np.clip(1.0 - instability, 0.0, 1.0))


@dataclass
class SQIResult:
    overall_score: float
    level: QualityLevel
    is_reliable: bool
    spectral_snr_db: float
    cross_roi_agreement: float
    landmark_stability: float
    warnings: list[str] = field(default_factory=list)


def assess_quality(
    spectral_snr_db: float,
    bpm_by_roi: dict[str, float],
    landmark_trajectories: np.ndarray | None,
    min_spectral_snr_db: float,
    max_cross_roi_bpm_diff: float,
    min_landmark_stability: float,
    min_overall_score_to_publish: float,
) -> SQIResult:
    warnings: list[str] = []

    snr_score = float(np.clip((spectral_snr_db - (-5)) / (min_spectral_snr_db - (-5) + 10), 0.0, 1.0))
    if spectral_snr_db < min_spectral_snr_db:
        warnings.append(
            f"Низкий spectral SNR ({spectral_snr_db:.1f} dB < {min_spectral_snr_db} dB): "
            "сигнал зашумлён или пульсовая компонента не выделяется."
        )

    roi_score = cross_roi_agreement_score(bpm_by_roi, max_diff=max_cross_roi_bpm_diff)
    if roi_score < 0.5:
        warnings.append(
            "Оценки BPM с разных ROI расходятся — вероятна частичная окклюзия "
            "или локальная засветка/тень на одной из зон лица."
        )

    stability_score = landmark_stability_score(landmark_trajectories)
    if stability_score < min_landmark_stability:
        warnings.append(
            "Высокая нестабильность landmark-точек — вероятны резкие движения "
            "головы или сбои трекинга."
        )

    overall = float(np.clip(0.5 * snr_score + 0.3 * roi_score + 0.2 * stability_score, 0.0, 1.0))

    if overall >= 0.75:
        level = QualityLevel.HIGH
    elif overall >= min_overall_score_to_publish:
        level = QualityLevel.MEDIUM
    else:
        level = QualityLevel.LOW

    # Взвешенное overall само по себе не гейт: SNR — единственный компонент,
    # который непосредственно измеряет "есть ли вообще пульсовая
    # составляющая в сигнале"; roi/stability оценивают согласованность и
    # трекинг и остаются высокими даже на чистом шуме. Поэтому публикация
    # требует ОБА условия отдельно, а не только их взвешенное среднее —
    # иначе snr_score=0 (чистый шум) при roi=stab=1.0 даёт overall=0.5 и
    # проходит порог "низкое качество -> не публиковать" из ТЗ.
    is_reliable = (overall >= min_overall_score_to_publish) and (spectral_snr_db >= min_spectral_snr_db)

    if not is_reliable:
        warnings.append(
            "SQI ниже порога публикации — BPM/HRV НЕ передаются в систему ПТСР "
            "за этот интервал (см. QualityConfig.min_overall_score_to_publish)."
        )

    return SQIResult(
        overall_score=overall,
        level=level,
        is_reliable=is_reliable,
        spectral_snr_db=spectral_snr_db,
        cross_roi_agreement=roi_score,
        landmark_stability=stability_score,
        warnings=warnings,
    )
