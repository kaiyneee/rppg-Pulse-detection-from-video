"""
Signal Quality Index (пункт "Оценка качества сигнала" ТЗ).

Если итоговый score ниже порога — BPM/HRV НЕ передаются в систему ПТСР
(см. pipeline.py::RPPGPipeline._gate_output), а выдаётся предупреждение.
Это прямая реализация требования ТЗ: "не передавать значение BPM в систему
ПТСР [при низком качестве]".

Компоненты индекса:

1. spectral_snr        — насколько энергия сигнала сконцентрирована вокруг
                          предполагаемой частоты пульса (в противовес шуму,
                          размазанному по всей полосе). Дополнительно
                          штрафуется harmonic_plausibility (см. ниже).
2. cross_roi_agreement  — независимые оценки BPM с лба/левой/правой щеки
                          должны совпадать; расхождение обычно означает, что
                          один из ROI зашумлён (тень, движение, макияж,
                          окклюзия).
3. landmark_stability   — по вариации позы головы/дрожанию landmark-точек
                          между кадрами, НОРМИРОВАННОЙ на межзрачковое
                          (межугловое, canthus-to-canthus) расстояние —
                          абсолютная, а не самоотносительная нормировка (см.
                          docstring landmark_stability_score).
4. temporal_consistency — устойчивость BPM между соседними перекрывающимися
                          окнами (см. docstring temporal_consistency_score).
                          Ловит коррелированные артефакты (мерцание
                          освещения, доминирующая гармоника движения), к
                          которым слепо cross_roi_agreement: все три ROI
                          берутся из одного видеопотока одним методом и
                          ошибутся согласованно, дав agreement=1.0.

Дополнительные (не входят в веса overall напрямую, но гейтят публикацию):

* harmonic_plausibility — энергия на 2f и f/2 относительно энергии на f;
  если сопоставима — вероятна путаница первой/второй гармоники или
  двигательный артефакт на кратной частоте (см. п.21 требований).
  Используется как штрафующий множитель для spectral SNR.
* illumination_flicker  — фоновый ROI (заведомо вне лица) сверяется с
  лицевым: узкий стабильный пик на СОВПАДАЮЩЕЙ частоте на фоне почти
  наверняка означает мерцание освещения (50/60 Гц люминесцентных/
  светодиодных источников при 30 fps бьётся прямо в полосу пульса), а не
  реальный пульс — у стен нет кровоснабжения. Жёсткий гейт публикации.

Итоговый score — взвешенное среднее нормированных к [0,1] компонент 1-4.
Публикация требует ОБА условия отдельно: overall >= порога И
spectral_snr_db >= порога И нет подозрения на мерцание — см. комментарий
внутри assess_quality о том, почему одного взвешенного overall недостаточно.

Веса компонент и большинство порогов (см. QualityConfig) НАЗНАЧЕНЫ по
разумным инженерным соображениям, а не откалиброваны на валидационной
выборке методом "покрытие vs ошибка" — это отдельная задача (п.24/30
требований), для которой нужен реальный или синтетический с известной
разметкой датасет. Каждый такой параметр помечен TODO(калибровка) в
QualityConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

from rppg.config import QualityConfig, QualityLevel


def _band_mask(freqs: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    return (freqs >= band_hz[0]) & (freqs <= band_hz[1])


def compute_psd(signal: np.ndarray, fps: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Welch PSD на ПОЛНОМ спектре (без обрезки по band_hz) — общий шаг для
    dominant_frequency_and_snr (там результат маскируется по band_hz) и
    harmonic_plausibility (там нужен доступ к 2f/f/2, которые часто лежат
    ВНЕ band_hz). Публичная (не приватная) — переиспользуется отладочным
    оверлеем scripts/run_webcam.py --debug для отрисовки того же самого
    спектра, который реально видит assess_quality, а не пересчитанного
    заново с другими параметрами Welch.

    ИСПРАВЛЕНИЕ (задача 5, п.4): раньше nperseg=min(len(signal), 256) при
    типичных len(signal)=300 (10с окно @ 30fps) давал nperseg=256 и, с
    noverlap по умолчанию (nperseg//2=128), РОВНО ОДИН сегмент
    (1 + (300-256)//128 = 1) — то есть усреднения Уэлча ФАКТИЧЕСКИ НЕ
    происходило, и получившаяся оценка PSD (а значит и spectral_snr_db, и
    порог на нём) имела дисперсию одиночной периодограммы, а не Welch.
    nperseg = len(signal)//2 (с explicit noverlap=nperseg//2) даёт 2-3
    перекрывающихся сегмента на типичном 10-секундном окне ценой частотного
    разрешения (короче сегмент -> шире главный лепесток) — для оценки SNR
    это выгодный обмен: ниже дисперсия оценки, стабильнее порог гейтинга."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 8 or np.std(signal) < 1e-10:
        return None, None
    nperseg = min(len(signal), max(64, len(signal) // 2))
    noverlap = nperseg // 2
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return freqs, psd


def dominant_frequency_and_snr(
    signal: np.ndarray, fps: float, band_hz: tuple[float, float], include_second_harmonic: bool = False,
) -> tuple[float, float]:
    """
    Возвращает (доминирующая_частота_Hz, spectral_snr_dB).

    spectral_snr — отношение энергии в узкой полосе (+-0.15 Hz) вокруг пика
    к энергии всей рабочей полосы band_hz, в дБ. Эта же величина используется
    как критерий периодичности для выбора PCA/ICA/head-motion компоненты
    (select_best_component_by_periodicity ниже) — общий инвариант: "хорошая"
    пульсовая компонента должна быть узкополосной.

    include_second_harmonic (задача 5, п.2): по умолчанию False — сохраняет
    исходное определение (энергия только на f). Wang et al. (2016, POS)
    определяют SNR ИНАЧЕ: в энергию сигнала входит и узкая полоса вокруг 2-й
    гармоники 2f — систолический подъём делает пульсовую волну
    несинусоидальной, и 2-я гармоника несёт реальную физиологическую
    энергию, а не шум. Наше определение по умолчанию строже литературного
    (не засчитывает 2f как "сигнал"), что даёт систематически более низкий
    snr_db при том же реальном качестве сигнала. include_second_harmonic=True
    включает альтернативный (Wang et al.) режим — ТОЛЬКО для сравнения
    "наших" порогов с литературными числами (см. scripts/analyze_log.py,
    задача 5), не используется по умолчанию нигде в пайплайне. См. также
    benchmark/evaluate.py::reference_relative_snr_db — то же определение с
    2-й гармоникой, но относительно ИЗВЕСТНОЙ референсной частоты (для
    валидации на датасете с ground truth), а не найденного здесь пика (для
    онлайн-гейтинга без ground truth).
    """
    freqs, psd = compute_psd(signal, fps)
    if freqs is None:
        return 0.0, -np.inf

    mask = _band_mask(freqs, band_hz)
    if not mask.any():
        return 0.0, -np.inf

    band_freqs, band_psd = freqs[mask], psd[mask]
    peak_idx = int(np.argmax(band_psd))
    peak_freq = float(band_freqs[peak_idx])

    narrow_mask = np.abs(band_freqs - peak_freq) <= 0.15
    signal_power = np.sum(band_psd[narrow_mask])
    if include_second_harmonic:
        second_harmonic_mask = np.abs(band_freqs - 2.0 * peak_freq) <= 0.15
        signal_power = signal_power + np.sum(band_psd[second_harmonic_mask])
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


def landmark_stability_score(
    landmark_trajectories: np.ndarray | None,
    interocular_distances_px: np.ndarray | None = None,
    jitter_threshold_ipd_fraction: float = 0.03,
    self_median_fallback_multiplier: float = 4.0,
) -> float:
    """
    Оценка устойчивости позы/трекинга по кадр-к-кадру дрожанию стабильных
    landmark-точек (нос, углы глаз и т.п.). Высокая частота больших скачков
    типична при резких движениях головы или частичной потере трекинга из-за
    окклюзии.

    Нормировка — АБСОЛЮТНАЯ, на межзрачковое (межугловое, canthus-to-canthus)
    расстояние в пикселях, а НЕ на собственную медиану джиттера в этом же
    окне (п.19 требований). Нормировка на собственную медиану делает метрику
    инвариантной к общему уровню движения: человек, равномерно и СИЛЬНО
    раскачивающийся всё окно, получил бы стабильность 1.0, потому что весь
    его джиттер был бы "типичным" относительно самого себя — метрика не
    может обнаружить то, чего она никогда не видела иначе. Нормировка на
    IPD даёт инвариантность к разрешению кадра и расстоянию до камеры
    (крупный план и мелкий план одного и того же движения дают одинаковый
    относительный джиттер), но НЕ к амплитуде движения — сильное раскачивание
    останется сильным относительно масштаба лица и будет корректно снижать
    оценку.
    """
    if landmark_trajectories is None or landmark_trajectories.shape[0] < 3:
        return 0.5
    diffs = np.diff(landmark_trajectories, axis=0)  # (T-1, N, 2)
    frame_jitter = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (T-1,)

    ref_distance = None
    if interocular_distances_px is not None:
        ipd = np.asarray(interocular_distances_px, dtype=float)
        ipd = ipd[np.isfinite(ipd) & (ipd > 1e-6)]
        if len(ipd) > 0:
            ref_distance = float(np.median(ipd))

    if ref_distance is not None:
        relative_jitter = frame_jitter / ref_distance
        threshold = jitter_threshold_ipd_fraction
    else:
        # Нет данных о масштабе лица (например, синтетические юнит-тесты без
        # реальных landmark-координат, или полностью потерянный трекинг) —
        # осознанный fallback на старую самоотносительную нормировку, а не
        # основной путь для реальных данных пайплайна. ВАЖНО: это другая
        # физическая величина (безразмерное отношение к собственной медиане,
        # обычно ~O(1)), а не доля IPD (обычно << 1) — поэтому у неё
        # ОТДЕЛЬНЫЙ, кратно больший порог (self_median_fallback_multiplier),
        # а не jitter_threshold_ipd_fraction: спутать их превратило бы
        # fallback в "почти всегда нестабильно" на любом реальном джиттере.
        relative_jitter = frame_jitter / (float(np.median(frame_jitter)) + 1e-6)
        threshold = self_median_fallback_multiplier

    instability = np.mean(relative_jitter > threshold)
    return float(np.clip(1.0 - instability, 0.0, 1.0))


def temporal_consistency_score(
    recent_bpm_history: list[float] | None,
    max_expected_change_bpm: float = 6.0,
) -> float:
    """
    Устойчивость оценки BPM между соседними перекрывающимися окнами (п.20
    требований).

    Зачем это отдельная, четвёртая компонента, а не то же самое, что
    cross_roi_agreement: cross_roi_agreement сравнивает три ROI ВНУТРИ
    одного окна, но все три берутся из одного и того же видеопотока одним
    и тем же методом извлечения — при мерцании освещения или доминирующей
    гармонике общего движения (например, дыхания или дрожи камеры) все три
    ROI ошибутся СОГЛАСОВАННО и дадут ложный agreement=1.0.
    temporal_consistency вместо этого сравнивает независимые по времени
    оценки (разные, хоть и перекрывающиеся, окна) — реальный физиологический
    пульс не может измениться на много BPM за step_seconds (обычно 1с) между
    соседними оценками, а нестабильный источник (артефакт, потеря трекинга,
    смена доминирующей частоты в спектре) обычно даёт заметно менее
    консистентную серию оценок.
    """
    if recent_bpm_history is None:
        return 0.5
    values = [v for v in recent_bpm_history if v is not None and not np.isnan(v)]
    if len(values) < 3:
        return 0.5  # недостаточно истории (начало записи) — нейтральная оценка
    diffs = np.abs(np.diff(values))
    mean_diff = float(np.mean(diffs))
    return float(np.clip(1.0 - mean_diff / max_expected_change_bpm, 0.0, 1.0))


def _local_bump_ratio(
    freqs: np.ndarray, psd: np.ndarray, center_hz: float, tol_hz: float, floor_half_width_hz: float,
) -> tuple[float, bool]:
    """Проверяет, есть ли на center_hz РЕАЛЬНЫЙ локальный горб спектра, а не
    просто точка на монотонно убывающем склоне 1/f-шума (задача 4).

    Сравнивает среднюю мощность в узкой полосе вокруг center_hz (±tol_hz) с
    мощностью в ДВУХ непосредственно прилегающих полосах — слева и справа
    (каждая шириной floor_half_width_hz, сразу за пределами ±tol_hz).
    В детрендированном rPPG-сигнале спектральная плотность круто падает с
    частотой (остаточный дрейф освещения/дыхание/микродвижения) — на чисто
    убывающем склоне мощность СЛЕВА (на более низкой частоте) всегда выше,
    чем на center_hz, поэтому условие "выше ОБОИХ соседей" истинно только
    на настоящем локальном горбе (реальной гармонике/движении на этой
    частоте), а не на любой точке склона 1/f. Отдельно возвращает
    (мощность_на_center / медиана_мощности_в_обеих_соседних_полосах) —
    оценку "во сколько раз пик выступает над локальным шумовым фоном".

    Возвращает (0.0, False), если по краю сетки не хватает данных для
    одной из соседних полос (например, center_hz слишком близко к 0 Hz).
    """
    peak_mask = np.abs(freqs - center_hz) <= tol_hz
    if not peak_mask.any():
        return 0.0, False
    p_peak = float(np.mean(psd[peak_mask]))

    left_mask = (freqs < center_hz - tol_hz) & (freqs >= center_hz - tol_hz - floor_half_width_hz)
    right_mask = (freqs > center_hz + tol_hz) & (freqs <= center_hz + tol_hz + floor_half_width_hz)
    if not left_mask.any() or not right_mask.any():
        return 0.0, False
    p_left = float(np.mean(psd[left_mask]))
    p_right = float(np.mean(psd[right_mask]))

    floor = float(np.median(np.concatenate([psd[left_mask], psd[right_mask]])))
    if floor <= 1e-15:
        return 0.0, False

    is_local_bump = p_peak > p_left and p_peak > p_right
    return p_peak / floor, is_local_bump


def harmonic_plausibility(
    raw_signal_detrended: np.ndarray | None,
    fps: float,
    peak_freq_hz: float,
    band_hz: tuple[float, float],
    significance_ratio: float = 3.0,
    tol_hz: float = 0.15,
    floor_half_width_hz: float = 0.3,
    max_penalty: float = 0.4,
) -> tuple[float, list[str]]:
    """
    Проверка на гармоники/субгармоники (п.21 требований) — классический
    failure mode rPPG: детектируется вторая гармоника истинного пульса
    (обнаруженная "частота" на самом деле 2x от настоящей) или движение с
    частотой, кратной пульсовой полосе, создаёт ложный пик.

    ВАЖНО: raw_signal_detrended должен быть ТОЛЬКО детрендирован, но НЕ
    пропущен через узкополосный bandpass в физиологическую полосу (обычно
    0.7-4.0 Hz) — иначе 2f и f/2 систематически обрезаны самим фильтром ещё
    до того, как их энергию можно измерить, и проверка вырождается в
    no-op на значительной части физиологического диапазона (f/2 < 0.7 Hz
    для любого f < 1.4 Hz, т.е. < 84 BPM; 2f > 4.0 Hz для любого f > 2.0 Hz,
    т.е. > 120 BPM). См. pipeline.py::_compute_estimate.

    ИСПРАВЛЕНИЕ (задача 4): раньше сравнивалась ДОЛЯ мощности p_2f/p_f (или
    p_half/p_f) с фиксированным порогом 0.7. Это ломалось именно там, где
    сигнал ЧИЩЕ всего: в детрендированном rPPG-сигнале спектральная
    плотность круто падает с частотой (остаточный дрейф освещения/дыхание/
    микродвижения — форма 1/f, "красный шум"). При пульсе 60-72 BPM
    субгармоника f/2 попадает в 0.5-0.6 Hz, где этой низкочастотной энергии
    заведомо больше, чем на самой пульсовой частоте f — p_half/p_f > 0.7
    срабатывало почти всегда НЕ из-за реальной субгармоники, а просто
    потому что рядом с DC мощность 1/f-шума всегда выше. Вместо доли
    мощности от пика на f теперь используется два независимых условия
    (см. _local_bump_ratio), которые 1/f-спад по построению не может
    удовлетворить:
      1) мощность на 2f/f/2 минимум в significance_ratio раз выше ЛОКАЛЬНОГО
         шумового фона рядом с этой же частотой (а не глобальной оценки по
         всей полосе, которая для 1/f-спектра плохо описывает локальный
         уровень шума и на низких, и на высоких частотах сразу);
      2) это настоящий ЛОКАЛЬНЫЙ МАКСИМУМ (горб), а не точка на монотонно
         убывающем склоне — на чистом 1/f-спаде мощность слева ВСЕГДА выше,
         чем в кандидатной точке, поэтому спад никогда не пройдёт этот тест.

    Субгармоника (f/2) проверяется, только если f/2 >= band_hz[0] (нижний
    край рабочей полосы) — иначе кандидатная частота лежит в зоне, где
    остаточный дрейф освещения физически доминирует над чем угодно, и
    проверка не может быть информативной ни при каком её исходе (обычно
    это BPM < ~84, т.е. f < 1.4 Hz -> f/2 < 0.7 Hz при типичной полосе
    0.7-4.0 Hz). В этом случае функция возвращает нейтральный score=1.0 по
    субгармонике молча (не как штраф и не как предупреждение об аномалии,
    см. assess_quality: сам факт пропуска проверки логируется отдельным
    булевым полем SQIResult, а не текстом в warnings, иначе на любой
    записи в состоянии покоя (< 84 BPM) эта нейтральная заметка забила бы
    собой топ warnings в scripts/analyze_log.py).

    Максимальный штраф ОГРАНИЧЕН max_penalty (по умолчанию 0.4, то есть
    score не падает ниже 0.6) — раньше множитель мог упасть до 0.4, что при
    и без того жёстком SNR-гейте (см. min_spectral_snr_db) почти
    гарантировало отказ в публикации за одно только гармоническое
    подозрение, даже не самое уверенное.

    Возвращает (score, warnings): score in [0,1], где 1.0 = гармоническая
    структура спектра чистая (заметной конкурирующей энергии на 2f/f/2 нет),
    ниже — тем более вероятна путаница. score используется как штрафующий
    множитель для snr_score в assess_quality: гармоническая путаница
    напрямую подрывает уверенность в том, что обнаруженная частота — это
    сам пульс, а не его кратная/дольная копия.
    """
    warnings: list[str] = []
    if raw_signal_detrended is None or peak_freq_hz <= 0:
        return 1.0, warnings

    freqs, psd = compute_psd(raw_signal_detrended, fps)
    if freqs is None:
        return 1.0, warnings

    score = 1.0

    ratio_2f, is_bump_2f = _local_bump_ratio(freqs, psd, 2.0 * peak_freq_hz, tol_hz, floor_half_width_hz)
    if is_bump_2f and ratio_2f >= significance_ratio:
        excess = min((ratio_2f - significance_ratio) / significance_ratio, 1.0)  # 0 на пороге -> 1 при 2x порога и выше
        score = min(score, 1.0 - max_penalty * excess)
        warnings.append(
            f"Локальный горб спектра на предполагаемой 2-й гармонике "
            f"(~{2 * peak_freq_hz * 60:.0f} BPM) в {ratio_2f:.1f}x выше локального шумового фона "
            f"(порог {significance_ratio:.1f}x) — возможна путаница истинной частоты пульса "
            "и её 2-й гармоники."
        )

    subharmonic_hz = peak_freq_hz / 2.0
    if subharmonic_hz >= band_hz[0]:
        ratio_half, is_bump_half = _local_bump_ratio(freqs, psd, subharmonic_hz, tol_hz, floor_half_width_hz)
        if is_bump_half and ratio_half >= significance_ratio:
            excess = min((ratio_half - significance_ratio) / significance_ratio, 1.0)
            score = min(score, 1.0 - max_penalty * excess)
            warnings.append(
                f"Локальный горб спектра на предполагаемой субгармонике f/2 "
                f"(~{peak_freq_hz * 30:.0f} BPM) в {ratio_half:.1f}x выше локального шумового фона "
                f"(порог {significance_ratio:.1f}x) — возможен двигательный/иной артефакт на "
                "частоте вдвое ниже обнаруженной."
            )
    # else: f/2 ниже рабочей полосы -> проверка неинформативна, см. докстринг
    # выше. Молча пропускается (не штраф, не warning) — см. assess_quality
    # про sqi_subharmonic_check_informative.

    return float(np.clip(score, 0.0, 1.0)), warnings


def detect_illumination_flicker(
    background_signal: np.ndarray | None,
    fps: float,
    face_peak_freq_hz: float,
    band_hz: tuple[float, float],
    freq_tolerance_hz: float = 0.15,
    min_background_snr_db: float = 3.0,
) -> tuple[bool, float, float]:
    """
    Дешёвый детектор мерцания освещения (п.22 требований). Люминесцентное/
    светодиодное освещение с частотой сети 50/60 Гц при частоте кадров ~30
    даёт биения (алиасинг), которые могут попасть прямо в физиологическую
    полосу пульса — визуально неотличимо от реального сигнала внутри одного
    ROI на лице.

    Лучший дешёвый детектор — фоновый ROI ЗАВЕДОМО ВНЕ лица (например,
    стена за головой, см. face/roi.py::build_background_roi_mask): если фон
    сам по себе даёт узкий стабильный пик на частоте, совпадающей с
    "пульсовой" частотой, обнаруженной на лице, это почти наверняка не
    пульс — у стен нет кровоснабжения, а значит, узкополосная периодичность
    в её яркости объясняется только внешним источником (мерцание, PWM
    подсветки и т.п.), который в равной мере модулирует и лицо.

    ИЗМЕНЕНИЕ (задача 7): раньше эта функция сама возвращала ФИНАЛЬНОЕ
    flicker_suspected по ОДНОМУ окну — жёсткий гейт срабатывал от единственного
    случайного совпадения. Два риска с этим: (а) фоновый ROI расположен по
    bounding box'у лица с небольшим отступом (см. face/roi.py::
    build_background_roi_mask) — если он задевает волосы/плечи/одежду,
    которые двигаются ВМЕСТЕ с человеком, совпадение по построению
    коррелирует с лицом; (б) вызывающий код (см. assess_quality) нормирует
    background_signal через preprocess_signal с z-score ДО того, как эта
    функция его увидит — z-score растягивает ЛЮБОЙ, даже почти постоянный,
    сигнал к единичной дисперсии, после чего усиленный шум квантования может
    случайно дать узкополосный на вид пик где угодно в рабочей полосе.

    Поэтому эта функция теперь возвращает только ОДНОКОННЫЙ "кандидат"
    (candidate_this_window, bg_freq_hz, bg_snr_db) — реальное решение
    flicker_suspected принимает assess_quality, требуя, чтобы кандидат
    подтвердился НЕСКОЛЬКО последовательных окон подряд (мерцание — стабильное
    во времени явление, см. SQIInputs.recent_flicker_candidates и
    QualityConfig.flicker_min_consecutive_windows), а не одно случайное окно.
    bg_freq_hz/bg_snr_db возвращаются ВСЕГДА (даже когда кандидат=False), а
    не только при совпадении — п.4 задачи 7: "логировать частоту и SNR
    фонового пика в каждое окно", чтобы scripts/analyze_log.py могло
    показать, насколько обоснованно срабатывает детектор, а не только факт
    срабатывания.
    """
    if background_signal is None or face_peak_freq_hz <= 0:
        return False, 0.0, float("-inf")

    bg_freq, bg_snr_db = dominant_frequency_and_snr(background_signal, fps, band_hz)
    if bg_snr_db < min_background_snr_db:
        return False, bg_freq, bg_snr_db  # фон сам по себе не узкополосен — ничего подозрительного

    candidate = abs(bg_freq - face_peak_freq_hz) <= freq_tolerance_hz
    return candidate, bg_freq, bg_snr_db


@dataclass
class SQIInputs:
    """Входные измерения для assess_quality за одно окно оценки."""

    spectral_snr_db: float
    peak_freq_hz: float
    fps: float
    band_hz: tuple[float, float]
    bpm_by_roi: dict[str, float]
    landmark_trajectories: np.ndarray | None = None
    interocular_distances_px: np.ndarray | None = None
    recent_bpm_history: list[float] | None = None
    # Детрендированный (НЕ узкополосно отфильтрованный) сигнал лучшего ROI —
    # только для harmonic_plausibility, см. её docstring.
    harmonic_check_signal: np.ndarray | None = None
    # Сигнал фонового ROI (вне лица), preprocess_signal-обработанный тем же
    # способом, что и лицевые ROI — для detect_illumination_flicker. None,
    # если фон недоступен ИЛИ его сырая (до z-score) дисперсия неотличима
    # от шума квантования — см. RPPGPipeline._estimate_background_signal.
    background_signal: np.ndarray | None = None
    # Задача 7: candidate-флаги detect_illumination_flicker с ПРЕДЫДУЩИХ
    # окон (самый свежий — последним, ТЕКУЩЕЕ окно сюда не входит) — для
    # требования устойчивости во времени (мерцание — стабильное явление,
    # разовое совпадение гораздо вероятнее случайный шумовой пик). См.
    # QualityConfig.flicker_min_consecutive_windows.
    recent_flicker_candidates: list[bool] | None = None


@dataclass
class SQIResult:
    overall_score: float
    level: QualityLevel
    is_reliable: bool
    spectral_snr_db: float
    cross_roi_agreement: float
    landmark_stability: float
    temporal_consistency: float
    harmonic_score: float
    # Финальное, персистентное решение (см. assess_quality) — жёсткий гейт
    # публикации. flicker_candidate ниже — сырое ОДНОконное совпадение ДО
    # требования устойчивости, используется RPPGPipeline только чтобы
    # обновить свою историю recent_flicker_candidates для следующего окна.
    flicker_suspected: bool
    flicker_candidate: bool = False
    flicker_background_freq_hz: float = 0.0
    flicker_background_snr_db: float = float("-inf")
    # Задача 4, п.3: False, если f/2 < band_hz[0] и проверка субгармоники
    # была ПРОПУЩЕНА как физически неинформативная (а не "пройдена чисто").
    # Отдельное булево поле, а не текст в warnings — иначе на любой записи
    # в покое (< ~84 BPM, обычный случай) эта нейтральная заметка забила бы
    # собой топ warnings в scripts/analyze_log.py на каждом окне.
    subharmonic_check_informative: bool = True
    warnings: list[str] = field(default_factory=list)


def assess_quality(inputs: SQIInputs, qcfg: QualityConfig) -> SQIResult:
    warnings: list[str] = []

    snr_score = float(
        np.clip((inputs.spectral_snr_db - (-5)) / (qcfg.min_spectral_snr_db - (-5) + 10), 0.0, 1.0)
    )
    if inputs.spectral_snr_db < qcfg.min_spectral_snr_db:
        warnings.append(
            f"Низкий spectral SNR ({inputs.spectral_snr_db:.1f} dB < {qcfg.min_spectral_snr_db} dB): "
            "сигнал зашумлён или пульсовая компонента не выделяется."
        )

    harmonic_score, harmonic_warnings = harmonic_plausibility(
        inputs.harmonic_check_signal,
        inputs.fps,
        inputs.peak_freq_hz,
        inputs.band_hz,
        significance_ratio=qcfg.harmonic_floor_ratio_threshold,
    )
    subharmonic_informative = (inputs.peak_freq_hz / 2.0) >= inputs.band_hz[0] if inputs.peak_freq_hz > 0 else True
    warnings.extend(harmonic_warnings)
    # Гармоническая путаница напрямую подрывает доверие к тому, что
    # обнаруженная частота — это сам пульс, а не его кратная/дольная копия,
    # поэтому штрафует именно spectral SNR, а не вводится как отдельный вес
    # в overall (у неё и так нет "своего" физического смысла качества сигнала
    # отдельно от того, ту ли частоту мы вообще нашли).
    snr_score *= harmonic_score

    roi_score = cross_roi_agreement_score(inputs.bpm_by_roi, max_diff=qcfg.max_cross_roi_bpm_diff)
    if roi_score < 0.5:
        warnings.append(
            "Оценки BPM с разных ROI расходятся — вероятна частичная окклюзия "
            "или локальная засветка/тень на одной из зон лица."
        )

    stability_score = landmark_stability_score(
        inputs.landmark_trajectories,
        interocular_distances_px=inputs.interocular_distances_px,
        jitter_threshold_ipd_fraction=qcfg.jitter_threshold_ipd_fraction,
    )
    if stability_score < qcfg.min_landmark_stability:
        warnings.append(
            "Высокая нестабильность landmark-точек относительно масштаба лица — "
            "вероятны резкие движения головы или сбои трекинга."
        )

    temporal_score = temporal_consistency_score(
        inputs.recent_bpm_history, max_expected_change_bpm=qcfg.max_expected_bpm_change_per_step
    )
    if temporal_score < 0.5:
        warnings.append(
            "BPM нестабилен между соседними перекрывающимися окнами — подозрение "
            "на коррелированный артефакт (мерцание освещения, доминирующая "
            "гармоника движения), который согласие ROI внутри одного окна не "
            "может обнаружить."
        )

    flicker_candidate, bg_freq_hz, bg_snr_db = detect_illumination_flicker(
        inputs.background_signal,
        inputs.fps,
        inputs.peak_freq_hz,
        inputs.band_hz,
        freq_tolerance_hz=qcfg.flicker_freq_tolerance_hz,
        min_background_snr_db=qcfg.flicker_min_background_snr_db,
    )
    # Задача 7: устойчивость во времени — мерцание физически стабильно,
    # разовое совпадение на одном окне гораздо правдоподобнее объясняется
    # случайным шумовым пиком (усиленным z-score нормализацией фона), чем
    # реальным мерцанием. Требуем, чтобы кандидат подтвердился в
    # flicker_min_consecutive_windows окон ПОДРЯД (включая текущее), иначе
    # это остаётся незавершённым подозрением (см. ниже), а не жёстким гейтом.
    candidate_streak = (list(inputs.recent_flicker_candidates or []) + [flicker_candidate])[
        -qcfg.flicker_min_consecutive_windows:
    ]
    flicker_suspected = (
        len(candidate_streak) >= qcfg.flicker_min_consecutive_windows and all(candidate_streak)
    )
    if flicker_suspected:
        warnings.append(
            f"Мерцание освещения ПОДТВЕРЖДЕНО {qcfg.flicker_min_consecutive_windows} окнами подряд: "
            f"фоновый ROI (вне лица) даёт узкий стабильный пик на {bg_freq_hz * 60:.1f} BPM "
            f"(SNR {bg_snr_db:.1f} дБ), совпадающий с частотой на лице "
            f"({inputs.peak_freq_hz * 60:.1f} BPM). Результат за это окно не публикуется."
        )
    elif flicker_candidate:
        warnings.append(
            f"Разовое совпадение фона с частотой лица на {bg_freq_hz * 60:.1f} BPM "
            f"(SNR {bg_snr_db:.1f} дБ) — ПОКА не подтверждено несколькими окнами подряд, "
            "публикация не заблокирована этим одним совпадением."
        )

    overall = float(
        np.clip(
            qcfg.weight_spectral_snr * snr_score
            + qcfg.weight_cross_roi * roi_score
            + qcfg.weight_landmark_stability * stability_score
            + qcfg.weight_temporal_consistency * temporal_score,
            0.0,
            1.0,
        )
    )

    if overall >= 0.75:
        level = QualityLevel.HIGH
    elif overall >= qcfg.min_overall_score_to_publish:
        level = QualityLevel.MEDIUM
    else:
        level = QualityLevel.LOW

    # Взвешенное overall само по себе не единственный гейт: SNR (со штрафом
    # за гармоники) — единственный компонент, который непосредственно
    # измеряет "есть ли вообще пульсовая составляющая в сигнале"; roi/
    # stability/temporal оценивают согласованность и трекинг и остаются
    # высокими даже на чистом шуме или синхронном артефакте (см. flicker).
    # Поэтому публикация требует ВСЕ условия одновременно, а не только
    # взвешенное среднее — иначе, например, snr_score=0 (чистый шум) при
    # остальных компонентах =1.0 даёт overall=0.6 и проходит порог
    # "низкое качество -> не публиковать" из ТЗ. Аналогично, мерцание
    # освещения может дать высокий overall (все компоненты "согласны" —
    # именно потому, что артефакт синхронный), но is_reliable всё равно
    # должно быть False.
    is_reliable = (
        (overall >= qcfg.min_overall_score_to_publish)
        and (inputs.spectral_snr_db >= qcfg.min_spectral_snr_db)
        and not flicker_suspected
    )

    if not is_reliable:
        warnings.append(
            "SQI ниже порога публикации — BPM/HRV НЕ передаются в систему ПТСР "
            "за этот интервал (см. QualityConfig.min_overall_score_to_publish)."
        )

    return SQIResult(
        overall_score=overall,
        level=level,
        is_reliable=is_reliable,
        spectral_snr_db=inputs.spectral_snr_db,
        cross_roi_agreement=roi_score,
        landmark_stability=stability_score,
        temporal_consistency=temporal_score,
        harmonic_score=harmonic_score,
        flicker_suspected=flicker_suspected,
        flicker_candidate=flicker_candidate,
        flicker_background_freq_hz=bg_freq_hz,
        flicker_background_snr_db=bg_snr_db,
        subharmonic_check_informative=subharmonic_informative,
        warnings=warnings,
    )
