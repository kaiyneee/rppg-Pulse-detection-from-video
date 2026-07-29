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

ВТОРАЯ МЕТОДОЛОГИЧЕСКАЯ ОГОВОРКА — окна BPM и HRV РАЗНЫЕ (см. config.py
HRVConfig, RPPGPipeline._update_ibi_log/_maybe_compute_hrv). BPM оценивается
по короткому (по умолчанию 10с) скользящему окну спектральным методом — это
корректно, спектральной оценке частоты не нужны минуты данных. HRV time/
frequency-domain метрики (SDNN, RMSSD, LF/HF) по определению Task Force
(1996) требуют существенно более длинных рядов IBI (минуты, не секунды) —
поэтому пайплайн копит IBI в отдельном накопителе поверх перекрывающихся
BPM-окон и публикует HRV значительно реже, чем BPM.
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
    # Доля IBI, отбракованных ectopic_artifact_mask как физиологически
    # неправдоподобные (см. п.16/17 требований).
    artifact_fraction: float = 0.0
    # False, если artifact_fraction превысил порог (HRVConfig.max_artifact_fraction)
    # или данных ещё недостаточно — потребитель обязан проверять этот флаг,
    # а не наличие чисел в полях (тот же принцип, что и PTSDPulseFeatures.publishable).
    publishable: bool = True
    warnings: list[str] = field(default_factory=list)


def detect_pulse_peaks(
    signal: np.ndarray, fps: float, min_hr_bpm: float = 40.0, max_hr_bpm: float = 220.0
) -> np.ndarray:
    """Детекция систолических пиков пульсовой волны (целочисленные индексы
    отсчётов — суб-сэмпловое уточнение делает refine_peaks_subsample).

    distance/prominence подобраны из физиологических границ ЧСС, а не
    произвольно — так минимизируются как пропуски слабых пиков, так и
    ложные срабатывания на дикротической выемке пульсовой волны.
    """
    signal = np.asarray(signal, dtype=float)
    min_distance = int(round(fps * 60.0 / max_hr_bpm))
    prominence = 0.25 * np.std(signal) if np.std(signal) > 1e-8 else None

    peaks, _ = find_peaks(signal, distance=max(1, min_distance), prominence=prominence)
    return peaks


def refine_peaks_subsample(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Параболическая суб-сэмпловая интерполяция положения пика по трём
    отсчётам вокруг него (y[i-1], y[i], y[i+1]) -> дробный индекс.

    Обязательна для HRV: на 30 fps шаг индекса пика равен 33.3 мс, а RMSSD
    у здоровых в покое — 20-50 мс, то есть шум квантования кадров сопоставим
    с самой измеряемой величиной. Без суб-сэмпловой локализации RMSSD/pNN50
    в значительной мере отражают частоту кадров камеры, а не физиологию.
    """
    signal = np.asarray(signal, dtype=float)
    refined = np.asarray(peaks, dtype=float).copy()
    n = len(signal)
    for i, p in enumerate(peaks):
        p = int(p)
        if p <= 0 or p >= n - 1:
            continue
        y0, y1, y2 = signal[p - 1], signal[p], signal[p + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) < 1e-12:
            continue
        offset = 0.5 * (y0 - y2) / denom
        refined[i] = p + float(np.clip(offset, -0.5, 0.5))
    return refined


def compute_ibi_ms(peak_positions: np.ndarray, fps: float) -> np.ndarray:
    """peak_positions — индексы отсчётов, целые или дробные (после
    refine_peaks_subsample)."""
    if len(peak_positions) < 2:
        return np.array([])
    return np.diff(peak_positions) / fps * 1000.0


def ectopic_artifact_mask(ibi_ms: np.ndarray, max_relative_change: float = 0.4) -> np.ndarray:
    """Булева маска физиологически правдоподобных IBI (True = валиден).

    В отличие от старого подхода (полное удаление выброса из ряда), маска
    НЕ удаляет интервалы: удаление "сшивает" ряд, и np.diff() на этом сшитом
    ряде считает ложную разность через дыру, там где реального соседства
    двух ударов не было. Правильная обработка — маскировать интервал как
    невалидный и явно исключать из diff() только пары, где хотя бы один
    сосед невалиден (см. hrv_time_domain).

    Порог поднят с прежних 30% до 40%: дыхательная синусовая аритмия (RSA)
    у молодых испытуемых в покое даёт межбитовый размах того же порядка, и
    более жёсткий фиксированный порог систематически подрезал реальную
    физиологическую вариабельность — то есть систематически занижал именно
    RMSSD/SDNN, которые предполагается сравнивать между группами.
    """
    ibi_ms = np.asarray(ibi_ms, dtype=float)
    n = len(ibi_ms)
    valid = np.ones(n, dtype=bool)
    if n < 3:
        return valid
    for i in range(1, n - 1):
        neighborhood = (ibi_ms[i - 1] + ibi_ms[i + 1]) / 2.0
        if neighborhood > 0 and abs(ibi_ms[i] - neighborhood) / neighborhood > max_relative_change:
            valid[i] = False
    return valid


def hrv_time_domain(
    ibi_ms: np.ndarray,
    valid_mask: np.ndarray | None = None,
    pnn_threshold_ms: float = 50.0,
    pnn20_threshold_ms: float = 20.0,
) -> dict:
    """valid_mask — см. ectopic_artifact_mask. mean/SDNN считаются по
    валидному подмножеству (эквивалентно старому удалению — порядок для них
    не важен); RMSSD/pNN* — только по диффам МЕЖДУ СОСЕДНИМИ валидными
    интервалами, чтобы не "перепрыгивать" через замаскированную дыру."""
    ibi_ms = np.asarray(ibi_ms, dtype=float)
    if valid_mask is None:
        valid_mask = np.ones(len(ibi_ms), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_ibi = ibi_ms[valid_mask]
    if len(valid_ibi) < 2:
        return dict(
            mean_hr_bpm=np.nan, sdnn_ms=np.nan, rmssd_ms=np.nan,
            pnn50_pct=np.nan, pnn20_pct=np.nan, mean_ibi_ms=np.nan,
            cv_ibi_pct=np.nan, n_beats=len(ibi_ms) + 1 if len(ibi_ms) else 0,
        )

    mean_ibi = float(np.mean(valid_ibi))
    sdnn = float(np.std(valid_ibi, ddof=1))
    mean_hr = float(60000.0 / mean_ibi) if mean_ibi > 0 else np.nan
    cv_ibi = float(sdnn / mean_ibi * 100) if mean_ibi > 0 else 0.0

    pair_valid = valid_mask[:-1] & valid_mask[1:] if len(valid_mask) > 1 else np.array([], dtype=bool)
    diffs = np.diff(ibi_ms)[pair_valid]
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > pnn_threshold_ms) * 100) if len(diffs) > 0 else 0.0
    pnn20 = float(np.mean(np.abs(diffs) > pnn20_threshold_ms) * 100) if len(diffs) > 0 else 0.0

    return dict(
        mean_hr_bpm=mean_hr, sdnn_ms=sdnn, rmssd_ms=rmssd,
        pnn50_pct=pnn50, pnn20_pct=pnn20, mean_ibi_ms=mean_ibi,
        cv_ibi_pct=cv_ibi, n_beats=len(ibi_ms) + 1,
    )


def hrv_frequency_domain(
    ibi_ms: np.ndarray,
    resample_hz: float = 4.0,
    lf_min_duration_seconds: float = 120.0,
    hf_min_duration_seconds: float = 60.0,
) -> dict:
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

    Минимальная длительность задаётся ОТДЕЛЬНО для LF и HF, а не одним
    общим порогом: LF-полоса 0.04-0.15 Гц соответствует периодам 6.7-25с,
    и чтобы такой период вообще был виден в спектре (нужно хотя бы
    несколько полных периодов), требуется порядка 2 минут данных; для
    HF 0.15-0.4 Гц (периоды 2.5-6.7с) достаточно ~1 минуты. Старый общий
    порог duration<20с был мягче ЛЮБОГО периода LF-полосы и на 10-секундном
    BPM-окне обнулял LF/HF ещё до этой проверки — см. HRVConfig и
    RPPGPipeline._maybe_compute_hrv, где IBI для этой функции берутся уже
    из минутного+ накопителя, а не из одного короткого окна.
    """
    if len(ibi_ms) < 4:
        return dict(lf_power=None, hf_power=None, lf_hf_ratio=None)

    beat_times_s = np.cumsum(ibi_ms) / 1000.0
    beat_times_s -= beat_times_s[0]
    ibi_centered = ibi_ms - np.mean(ibi_ms)

    duration = beat_times_s[-1]
    if duration < hf_min_duration_seconds:
        return dict(lf_power=None, hf_power=None, lf_hf_ratio=None)

    uniform_t = np.arange(0, duration, 1.0 / resample_hz)
    uniform_ibi = np.interp(uniform_t, beat_times_s, ibi_centered)

    freqs, psd = welch(uniform_ibi, fs=resample_hz, nperseg=min(len(uniform_ibi), 256))
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0

    lf_power = None
    lf_hf_ratio = None
    if duration >= lf_min_duration_seconds:
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        lf_hf_ratio = lf_power / hf_power if hf_power > 1e-8 else None

    return dict(lf_power=lf_power, hf_power=hf_power, lf_hf_ratio=lf_hf_ratio)


def _trim_edges(signal: np.ndarray, fps: float, edge_seconds: float, warnings: list[str]) -> np.ndarray:
    """Отбрасывает ~edge_seconds с каждого края перед пиковым анализом.

    Причины (не влияют на оценку BPM по спектру, но искажают именно HRV):
    у POS первые/последние (ws-1) отсчётов overlap-add собраны из меньшего
    числа слагаемых окна -> заниженная амплитуда; плюс краевые переходные
    процессы filtfilt. Заниженные/искажённые края дают пропущенные или
    ложные пики find_peaks на границах окна.
    """
    trim = int(round(edge_seconds * fps))
    if trim <= 0 or len(signal) <= 2 * trim + int(round(fps * 2)):
        if trim > 0:
            warnings.append(
                "Окно слишком короткое для обрезки краёв перед пиковым анализом — "
                "точность HRV на границах окна может быть снижена."
            )
        return signal
    return signal[trim: len(signal) - trim]


def _finalize_hrv_features(
    ibi_ms: np.ndarray,
    pulse_signal_quality_score: float,
    pnn_threshold_ms: float,
    pnn20_threshold_ms: float,
    compute_frequency_domain: bool,
    max_artifact_fraction: float,
    ectopic_max_relative_change: float,
    lf_min_duration_seconds: float,
    hf_min_duration_seconds: float,
    warnings: list[str],
) -> HRVFeatures:
    ibi_ms = np.asarray(ibi_ms, dtype=float)
    valid_mask = ectopic_artifact_mask(ibi_ms, max_relative_change=ectopic_max_relative_change)
    n_total = len(ibi_ms)
    n_rejected = int((~valid_mask).sum())
    artifact_fraction = (n_rejected / n_total) if n_total > 0 else 0.0

    if n_rejected > 0:
        warnings.append(
            f"Отбраковано {n_rejected}/{n_total} межпульсовых интервалов "
            f"({artifact_fraction * 100:.1f}%) как физиологически неправдоподобные "
            "(вероятный артефакт детектора пиков); замаскированы, а не удалены "
            "из ряда (см. ectopic_artifact_mask)."
        )

    # Стандарт HRV-литературы (Kubios, neurokit2): при высокой доле
    # отбракованных интервалов метрики HRV за окно не публикуются.
    publishable = n_total >= 2 and artifact_fraction <= max_artifact_fraction
    if not publishable:
        if n_total < 2:
            warnings.append("Недостаточно обнаруженных межпульсовых интервалов для HRV.")
        else:
            warnings.append(
                f"Доля отбракованных интервалов ({artifact_fraction * 100:.1f}%) превышает "
                f"порог {max_artifact_fraction * 100:.0f}% — HRV за этот интервал не публикуется."
            )

    td = hrv_time_domain(ibi_ms, valid_mask, pnn_threshold_ms, pnn20_threshold_ms)

    fd = dict(lf_power=None, hf_power=None, lf_hf_ratio=None)
    if compute_frequency_domain and publishable:
        fd = hrv_frequency_domain(
            ibi_ms[valid_mask],
            lf_min_duration_seconds=lf_min_duration_seconds,
            hf_min_duration_seconds=hf_min_duration_seconds,
        )

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
        artifact_fraction=artifact_fraction,
        publishable=publishable,
        warnings=warnings,
    )


def extract_hrv_features(
    pulse_signal: np.ndarray,
    fps: float,
    pulse_signal_quality_score: float = 0.0,
    pnn_threshold_ms: float = 50.0,
    pnn20_threshold_ms: float = 20.0,
    compute_frequency_domain: bool = True,
    edge_trim_seconds: float = 1.0,
    max_artifact_fraction: float = 0.05,
    ectopic_max_relative_change: float = 0.4,
    lf_min_duration_seconds: float = 120.0,
    hf_min_duration_seconds: float = 60.0,
) -> HRVFeatures:
    """HRV из ОДНОГО непрерывного окна пульсового сигнала (детектирует пики
    сам). Для короткого (секунды-десятки секунд) окна LF/HF почти всегда
    останутся None по построению (см. hrv_frequency_domain) — это ожидаемо
    и корректно, а не баг: полноценный HRV в пайплайне считается по
    накопителю IBI поверх многих окон, см. extract_hrv_features_from_ibi и
    RPPGPipeline._maybe_compute_hrv.
    """
    warnings: list[str] = []

    trimmed_signal = _trim_edges(pulse_signal, fps, edge_trim_seconds, warnings)
    peaks = detect_pulse_peaks(trimmed_signal, fps)
    if len(peaks) < 3:
        warnings.append("Слишком мало обнаруженных пульсовых пиков для надёжного HRV.")

    peaks_sub = refine_peaks_subsample(trimmed_signal, peaks)
    ibi_ms = compute_ibi_ms(peaks_sub, fps)

    return _finalize_hrv_features(
        ibi_ms, pulse_signal_quality_score, pnn_threshold_ms, pnn20_threshold_ms,
        compute_frequency_domain, max_artifact_fraction, ectopic_max_relative_change,
        lf_min_duration_seconds, hf_min_duration_seconds, warnings,
    )


def extract_hrv_features_from_ibi(
    ibi_ms: np.ndarray,
    pulse_signal_quality_score: float = 0.0,
    pnn_threshold_ms: float = 50.0,
    pnn20_threshold_ms: float = 20.0,
    compute_frequency_domain: bool = True,
    max_artifact_fraction: float = 0.05,
    ectopic_max_relative_change: float = 0.4,
    lf_min_duration_seconds: float = 120.0,
    hf_min_duration_seconds: float = 60.0,
) -> HRVFeatures:
    """Как extract_hrv_features, но принимает уже готовый ряд IBI вместо
    сырого сигнала — путь, которым пользуется пайплайн: там IBI копится
    инкрементально поверх множества перекрывающихся BPM-окон
    (RPPGPipeline._update_ibi_log), а не детектируется заново из одного
    короткого окна."""
    warnings: list[str] = []
    return _finalize_hrv_features(
        ibi_ms, pulse_signal_quality_score, pnn_threshold_ms, pnn20_threshold_ms,
        compute_frequency_domain, max_artifact_fraction, ectopic_max_relative_change,
        lf_min_duration_seconds, hf_min_duration_seconds, warnings,
    )
