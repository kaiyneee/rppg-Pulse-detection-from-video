"""
Взвешенное объединение (fusion) пульсовых сигналов из нескольких ROI и
модальностей (п.34 требований) — альтернатива argmax-выбору одного
"лучшего" источника (см. pipeline.py::RPPGPipeline._compute_estimate,
best_roi = max(per_roi_signal, key=spectral_snr)).

Мотивация: argmax отбрасывает ВСЮ информацию из источников, кроме одного
"победителя" по spectral SNR, даже если остальные источники частично
согласуются и несут НЕЗАВИСИМЫЙ (не дублирующий) шум, который при
объединении статистически усредняется — классический аргумент в пользу
sensor fusion. Взвешенное объединение по SQI (вес ~ качество каждого
источника) сохраняет эту информацию, вместо того чтобы игнорировать
2 источника из 3-4 на каждом окне.

РИСК, который проверяется ЭКСПЕРИМЕНТАЛЬНО (см.
scripts/compare_fusion_vs_argmax.py), а не считается данностью: color-rPPG
сигнал (поглощение света гемоглобином, de Haan & Jeanne 2013 и др.) и
head-motion сигнал (баллистокардиографическая отдача, Balakrishnan et al.
2013) физически измеряют РАЗНЫЕ явления и не гарантированно синфазны между
собой — наивное суммирование БЕЗ выравнивания по знаку/фазе рискует
деструктивной интерференцией и может дать оценку ХУЖЕ, чем argmax
отдельного лучшего источника. Именно поэтому здесь есть явный шаг
выравнивания (_align_sign_and_lag) перед суммированием.
"""

from __future__ import annotations

import numpy as np


def snr_db_to_weight(snr_db: float, floor_db: float = -5.0, ceil_db: float = 15.0) -> float:
    """Монотонное отображение spectral SNR (дБ) в неотрицательный вес
    [0,1]. Та же кусочно-линейная идея, что и snr_score в
    quality.assess_quality (чем выше SNR, тем больше вес), но не завязана
    на конкретный QualityConfig-порог публикации — fusion работает даже
    когда ни один источник по отдельности не проходит порог гейтинга."""
    if not np.isfinite(snr_db):
        return 0.0
    return float(np.clip((snr_db - floor_db) / (ceil_db - floor_db), 0.0, 1.0))


def _align_sign_and_lag(
    signal: np.ndarray, reference: np.ndarray, max_lag_samples: int
) -> tuple[np.ndarray, int, float]:
    """
    Выравнивает signal относительно reference по ЗНАКУ и ЦЕЛОЧИСЛЕННОМУ
    лагу через кросс-корреляцию, ограниченную окном +-max_lag_samples (см.
    модульный docstring про физическую несинфазность color/motion сигналов).

    Возвращает (выровненный_signal, применённый_лаг_в_отсчётах,
    коэффициент_корреляции_на_этом_лаге _после_ учёта знака, т.е. всегда
    >= 0). Лаг применяется через np.roll (циклический сдвиг) — при
    max_lag_seconds << длины окна (по умолчанию 0.3с при окне 10с)
    краевые артефакты циклического сдвига пренебрежимо малы относительно
    длины окна.
    """
    n = len(signal)
    max_lag_samples = max(0, min(max_lag_samples, n - 1))
    ref_c = reference - reference.mean()
    sig_c = signal - signal.mean()

    xcorr = np.correlate(ref_c, sig_c, mode="full")
    # np.correlate(a, v, 'full')[k] = sum_n a[n] * v[n - lag], lag = k - (len(v)-1)
    # -> положительный lag означает "v нужно сдвинуть ВПРАВО (np.roll(v, lag))",
    # чтобы совпасть с a. Проверено юнит-тестом test_align_sign_and_lag_recovers_known_shift.
    lags = np.arange(-(n - 1), n)
    mask = np.abs(lags) <= max_lag_samples
    xcorr_w, lags_w = xcorr[mask], lags[mask]

    best_idx = int(np.argmax(np.abs(xcorr_w)))
    best_lag = int(lags_w[best_idx])

    aligned = np.roll(signal.astype(float), best_lag)
    # Коэффициент корреляции пересчитывается НАПРЯМУЮ на уже выровненных (по
    # лагу) массивах через np.corrcoef, а НЕ как xcorr_w[best_idx]/norm:
    # xcorr в mode="full" — это ЛИНЕЙНАЯ кросс-корреляция (сумма только по
    # перекрывающимся после сдвига отсчётам), тогда как norm=||ref||*||sig||
    # берёт нормы ПОЛНОЙ длины n — числитель и знаменатель относятся к
    # разным по размеру суммам, что систематически ЗАНИЖАЕТ полученный
    # коэффициент (проверено юнит-тестом: на идеально совпадающих после
    # выравнивания массивах старая формула давала ~0.98, а не ~1.0).
    corr_coef = float(np.corrcoef(aligned, reference)[0, 1])
    if np.isnan(corr_coef):
        corr_coef = 0.0
    if corr_coef < 0:
        aligned = -aligned
        corr_coef = -corr_coef

    return aligned, best_lag, corr_coef


def fuse_signals_by_sqi(
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
    fps: float = 30.0,
    max_lag_seconds: float = 0.3,
) -> tuple[np.ndarray, dict[str, dict]]:
    """
    Взвешенное объединение НЕСКОЛЬКИХ независимых оценок пульсового сигнала
    (например, 3 цветовых ROI + 1 канал head-motion) в ОДИН сигнал —
    альтернатива argmax-выбору (п.34 требований).

    signals: name -> 1D сигнал ОДИНАКОВОЙ длины, ожидается уже
    z-score-нормированным и полосно-отфильтрованным (preprocess_signal) —
    fusion не занимается собственной нормализацией амплитуды.
    weights: name -> неотрицательный вес (например, snr_db_to_weight от
    spectral SNR каждого источника, см. quality.dominant_frequency_and_snr).
    Нормируются к сумме 1 здесь; вызывающий код передаёт "сырые" веса.

    Источник с МАКСИМАЛЬНЫМ весом используется как REFERENCE для
    выравнивания фазы/знака остальных — не произвольный первый ключ
    словаря, чтобы результат не зависел от порядка перечисления источников
    и чтобы выравнивание опиралось на предположительно самый чистый сигнал.

    Возвращает (fused_signal, diagnostics), где diagnostics[name] =
    {"weight", "lag_samples", "corr_with_reference"} — для логирования и
    для scripts/compare_fusion_vs_argmax.py.
    """
    if not signals:
        raise ValueError("fuse_signals_by_sqi: пустой словарь сигналов")
    if set(signals.keys()) != set(weights.keys()):
        raise ValueError("signals и weights должны иметь одинаковые ключи")

    names = list(signals.keys())
    lengths = {len(signals[n]) for n in names}
    if len(lengths) != 1:
        raise ValueError(f"все сигналы должны быть одной длины, получено {lengths}")

    total_weight = sum(max(w, 0.0) for w in weights.values())
    if total_weight <= 1e-12:
        # Все веса нулевые (например, все источники ниже SNR floor) —
        # равномерные веса как безопасный fallback, а не деление на 0.
        norm_weights = {n: 1.0 / len(names) for n in names}
    else:
        norm_weights = {n: max(weights[n], 0.0) / total_weight for n in names}

    reference_name = max(norm_weights, key=norm_weights.get)
    reference_signal = np.asarray(signals[reference_name], dtype=float)
    max_lag_samples = max(1, int(round(max_lag_seconds * fps)))

    fused = np.zeros(len(reference_signal), dtype=float)
    diagnostics: dict[str, dict] = {}
    for name in names:
        if name == reference_name:
            aligned, lag, corr = reference_signal, 0, 1.0
        else:
            aligned, lag, corr = _align_sign_and_lag(
                np.asarray(signals[name], dtype=float), reference_signal, max_lag_samples
            )
        fused += norm_weights[name] * aligned
        diagnostics[name] = {"weight": norm_weights[name], "lag_samples": lag, "corr_with_reference": corr}

    return fused, diagnostics
