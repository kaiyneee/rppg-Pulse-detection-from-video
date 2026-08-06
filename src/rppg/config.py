"""
Централизованная конфигурация rPPG-пайплайна.

Все "магические числа" исходного репозитория (0.75-3 Hz полоса, PCA=5 компонент,
захардкоженный fps и т.д.) вынесены сюда как явные, документированные и
переопределяемые параметры. Это отдельная научная претензия к оригинальному
коду: воспроизводимость эксперимента невозможна, если параметры разбросаны
по телу функций.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExtractionMethod(str, Enum):
    """Методы извлечения rPPG-сигнала (пункт "Извлечение сигнала" ТЗ)."""

    GREEN = "green"
    CHROM = "chrom"
    POS = "pos"
    PCA = "pca"
    ICA = "ica"
    HEAD_MOTION = "head_motion"
    # Адаптация к разным камерам/освещению: КАЖДОЕ окно оценивает все 5
    # цветовых методов выше по SNR и использует победителя (см.
    # RPPGPipeline._select_auto_method в pipeline.py). Разные камеры/условия
    # объективно благоприятствуют разным классическим методам — единого
    # метода, лучшего всегда и везде, в rPPG-литературе нет.
    AUTO = "auto"


class FrequencyMethod(str, Enum):
    """Методы оценки доминирующей частоты (пункт "Анализ частоты" ТЗ)."""

    FFT = "fft"
    WELCH = "welch"
    LOMB_SCARGLE = "lomb_scargle"


class ROIName(str, Enum):
    FOREHEAD = "forehead"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"


class QualityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FaceModelConfig:
    """Настройки MediaPipe Face Landmarker (Tasks API)."""

    # Скачивается один раз, см. docs/research_report.md -> "Установка модели".
    model_asset_path: str = "models/face_landmarker.task"
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_face_blendshapes: bool = False
    # Нужна для оценки позы головы (yaw/pitch/roll) -> устойчивость к движению.
    output_facial_transformation_matrixes: bool = True
    # 'GPU' ускоряет инференс на системах с поддержкой; 'CPU' — переносимый вариант.
    delegate: str = "CPU"


@dataclass
class ROIConfig:
    """Настройки динамических ROI (лоб / левая щека / правая щека)."""

    enabled_rois: tuple[ROIName, ...] = (
        ROIName.FOREHEAD,
        ROIName.LEFT_CHEEK,
        ROIName.RIGHT_CHEEK,
    )
    # Сжатие полигона к центроиду, чтобы не захватывать волосы/фон/уши
    # при повороте головы. 1.0 = без сжатия.
    shrink_factor: float = 0.90
    # Минимальная доля валидных (не окклюдированных) пикселей в ROI,
    # иначе ROI за этот кадр помечается невалидным.
    min_valid_pixel_fraction: float = 0.6


@dataclass
class FilterConfig:
    """Полосовой фильтр и препроцессинг сигнала."""

    # 0.7-4.0 Hz -> 42-240 BPM: с запасом покрывает физиологический диапазон
    # (включая тахикардию при остром стрессовом ответе, что важно для ПТСР-контекста).
    low_hz: float = 0.7
    high_hz: float = 4.0
    filter_order: int = 4
    detrend_method: str = "tarvainen"  # "linear" | "tarvainen" | "none"
    # п.36 требований: 300 в исходном коде было взято из HRV-литературы БЕЗ
    # проверки, что оно означает на fs=30 Гц (видео) — та же λ имеет РАЗНУЮ
    # частоту среза на разной fs (см. signal.preprocessing.tarvainen_cutoff_hz).
    # Стандартная HRV-практика (Kubios/PhysioData Toolbox: RR-тахограмма
    # ресэмплируется на 4 Гц, λ=500 -> cutoff~=0.04 Hz, см. docstring
    # tarvainen_frequency_response) при переносе λ=300 БЕЗ пересчёта на
    # fs=30 Гц даёт cutoff=0.344 Hz — почти в 10 раз выше, чем предполагалось
    # в литературе, и уже заметно (2.4%) затрагивает нижний край пульсовой
    # полосы (0.7 Hz). λ=3542 подобрано так, чтобы cutoff=0.1 Hz — период
    # 10с, РОВНО WindowConfig.window_seconds: тренды медленнее одного окна
    # анализа считаются дрейфом и удаляются, всё быстрее — нет (при этом на
    # 0.7 Hz затухание падает до 0.02%, т.е. пульсовая полоса становится ещё
    # прозрачнее). Полный вывод и АЧХ — scripts/analyze_tarvainen_lambda.py.
    tarvainen_lambda: float = 3542.0
    normalize_method: str = "zscore"  # "zscore" | "minmax" | "none"


@dataclass
class WindowConfig:
    """Скользящее окно анализа."""

    window_seconds: float = 10.0
    step_seconds: float = 1.0
    assumed_fps: float = 30.0
    # п.6 требований (задача 6): было 4.0 при window_seconds=10.0 — первые
    # ~6с оценки считались по окну ВДВОЕ короче проектного. Эти шумные
    # "прогревочные" оценки попадали в RPPGPipeline._recent_bpm_history
    # (роняя temporal_consistency на следующие несколько шагов) и в
    # _update_ibi_log (засоряя накопитель IBI мусорными пиками на все
    # HRVConfig.target_accumulation_seconds вперёд).
    #
    # None -> см. __post_init__: по умолчанию РАВЕН window_seconds, то есть
    # первая оценка в принципе не считается, пока окно не заполнено
    # полностью — это полностью устраняет "прогревочные" окна на уровне
    # инварианта (elapsed_s >= min_seconds_before_estimate == window_seconds
    # означает, что буфер уже содержит window_seconds секунд кадров), а не
    # только помечает их флагом постфактум. Можно явно переопределить более
    # низким значением ради быстрого time-to-first-estimate, но тогда нужно
    # самостоятельно исключать частичные окна из истории (см. pipeline.py).
    min_seconds_before_estimate: float | None = None

    def __post_init__(self) -> None:
        if self.min_seconds_before_estimate is None:
            self.min_seconds_before_estimate = self.window_seconds


@dataclass
class QualityConfig:
    """Пороги Signal Quality Index (пункт "Оценка качества сигнала" ТЗ).

    ВАЖНО про пороги ниже, помеченные TODO(калибровка): они назначены по
    разумным инженерным соображениям (см. комментарий у каждого), а НЕ
    подобраны на валидационной выборке по кривой "покрытие vs ошибка"
    (п.24/30 требований — для этого нужен размеченный датасет с истинным
    BPM, которого в этой среде нет). Значения по умолчанию — стартовая
    точка; перед использованием в статье/публикации их нужно откалибровать
    и показать саму калибровочную кривую, а не просто финальные числа.
    """

    # Задача 13а: ПО УМОЛЧАНИЮ True — energия сигнала в dominant_frequency_and_snr
    # включает узкую полосу вокруг 2-й гармоники 2f, а не только вокруг f
    # (см. её докстринг). Это определение Wang et al. (2016, POS) — то, с
    # чем сравнивается литература, и то, с чем сравниваются наши пороги
    # ниже. ДО задачи 13а было эквивалентно False: наше определение SNR
    # было строже литературного (не засчитывало 2f как "сигнал"), что
    # систематически занижало snr_db при том же реальном качестве сигнала
    # — одна из вероятных причин заниженного покрытия при пороге 3.0 дБ
    # (см. min_spectral_snr_db ниже). False оставлен только для
    # воспроизведения старого поведения/сравнения.
    include_second_harmonic_in_snr: bool = True

    # TODO(калибровка) — задача 5/10: это ЕДИНСТВЕННЫЙ порог, который сейчас
    # практически целиком определяет is_reliable (см. assess_quality в
    # quality.py): overall >= min_overall_score_to_publish достигается почти
    # автоматически, когда остальные 3 компоненты (суммарный вес 0.6)
    # приличные, поэтому решение о публикации на практике сводится к
    # spectral_snr_db >= это число. 3.0 дБ означает "67% мощности рабочей
    # полосы 0.7-4.0 Гц должно приходиться на полосу ±0.15 Гц вокруг пика"
    # (9% ширины полосы) — очень высокая планка для 10-секундного окна. С
    # задачи 13а include_second_harmonic_in_snr=True по умолчанию — теперь
    # ЭТО определение совпадает со стандартным литературным (Wang et al.,
    # 2016), а не строже его, как было раньше. НЕ менять это число наугад —
    # см. scripts/analyze_log.py (задача 1) для реального распределения
    # spectral_snr_db на конкретной камере/условиях и
    # scripts/calibrate_thresholds.py (задача 10) для итоговой калибровки по
    # кривой "покрытие vs ошибка".
    min_spectral_snr_db: float = 3.0
    # TODO(калибровка) — задача 13б: раньше cross_roi_agreement_score
    # сравнивал РАЗМАХ (max-min) с этим порогом — одно шумное ROI (например,
    # засвеченная/затенённая щека) утягивало max или min и обнуляло всю
    # компоненту, даже если ОСТАЛЬНЫЕ ROI прекрасно согласны между собой. С
    # задачи 13б используется МЕДИАННОЕ АБСОЛЮТНОЕ ОТКЛОНЕНИЕ (MAD) от
    # медианы — устойчивая статистика, которую один выброс сдвигает
    # незначительно (см. cross_roi_agreement_score). MAD для тех же данных
    # обычно заметно МЕНЬШЕ, чем range, поэтому порог поднят с прежних 8.0,
    # чтобы не стать СТРОЖЕ, чем раньше, по чистой случайности смены
    # статистики — точное значение ещё предстоит откалибровать (задача 10).
    max_cross_roi_bpm_diff: float = 15.0
    # TODO(калибровка) — задача 13б: минимальный СОБСТВЕННЫЙ spectral SNR
    # (см. dominant_frequency_and_snr), чтобы ROI вообще участвовал в
    # cross_roi_agreement_score. Сравнивать оценку заведомо шумного ROI
    # (SNR глубоко отрицательный — там нет пульсовой компоненты вообще) с
    # оценкой хорошего ROI бессмысленно: их "согласие" или "несогласие" —
    # чистая случайность, не сигнал о качестве. 0.0 дБ — половина порога
    # публикации min_spectral_snr_db, сознательно мягче него: цель этого
    # фильтра — отсечь ЯВНО безнадёжные ROI, а не предвосхищать основной
    # SNR-гейт ещё раз здесь.
    cross_roi_min_component_snr_db: float = 0.0
    min_landmark_stability: float = 0.5
    # TODO(калибровка) — см. комментарий у min_spectral_snr_db выше: пока
    # spectral_snr_db остаётся узким местом, этот порог редко оказывается
    # ограничивающим сам по себе. Итоговый BPM передаётся в систему ПТСР
    # только если score >= порога.
    min_overall_score_to_publish: float = 0.5

    # --- п.19: абсолютная нормировка landmark_stability на IPD ---
    # Доля межзрачкового (межуглового) расстояния, которую джиттер
    # landmark-точки за один кадр должен превысить, чтобы считаться
    # "выбросом" движения. TODO(калибровка): подобрать по реальным записям
    # с известной разметкой движения головы.
    jitter_threshold_ipd_fraction: float = 0.03

    # --- п.20: temporal consistency между соседними окнами ---
    # Ожидаемое максимальное изменение BPM за один шаг скользящего окна
    # (WindowConfig.step_seconds) у РЕАЛЬНОГО физиологического сигнала;
    # используется как масштаб для temporal_consistency_score.
    # TODO(калибровка).
    max_expected_bpm_change_per_step: float = 6.0

    # --- п.21: гармоники/субгармоники (см. задача 4) ---
    # ПЕРЕИМЕНОВАНО и переопределено относительно исходного
    # harmonic_ratio_threshold=0.7 (доля мощности p_2f/p_f или p_half/p_f):
    # на реальном детрендированном rPPG-сигнале (спектр 1/f, круто падающий
    # к низким частотам) эта доля почти всегда превышала 0.7 у субгармоники
    # НЕ из-за реальной путаницы, а просто потому что рядом с DC мощность
    # 1/f-шума заведомо выше, чем на самой пульсовой частоте (см. docstring
    # harmonic_plausibility в quality.py). Новое определение: во сколько раз
    # пик на 2f/f/2 должен превышать ЛОКАЛЬНЫЙ шумовой фон рядом с этой же
    # частотой (и одновременно быть настоящим локальным максимумом, а не
    # точкой на склоне) — величина, которую монотонный 1/f-спад в принципе
    # не может случайно удовлетворить. TODO(калибровка) на реальных данных
    # с разметкой ложных гармонических срабатываний (см. задача 10).
    harmonic_floor_ratio_threshold: float = 3.0

    # --- п.22: мерцание освещения (см. задача 7 про ложные срабатывания) ---
    # Допуск (Hz) для совпадения частоты фонового и лицевого пика.
    flicker_freq_tolerance_hz: float = 0.15
    # Минимальный spectral SNR фонового ROI, чтобы его пик вообще
    # рассматривался как "узкий и стабильный" (а не случайный шумовой максимум).
    # TODO(калибровка).
    flicker_min_background_snr_db: float = 3.0
    # Минимальное СКО сырого (ДО z-score нормализации) фонового сигнала в
    # единицах медианной яркости пикселя (0-255) — ниже этого фон считается
    # "нет данных", а не анализируется. preprocess_signal растягивает ЛЮБОЙ
    # сигнал к единичной дисперсии через z-score; без этой проверки почти
    # постоянный фон (ровная стена + шум квантования камеры) после
    # нормализации выглядит как обычный сигнал, и усиленный шум может
    # случайно дать узкополосный на вид пик где угодно в рабочей полосе —
    # ложное срабатывание детектора мерцания на ровном мониторе тишины.
    # TODO(калибровка) на реальных записях с известным уровнем шума камеры.
    flicker_min_raw_std: float = 0.5
    # Сколько ПОСЛЕДОВАТЕЛЬНЫХ окон подряд фоновый пик должен совпадать с
    # лицевым, чтобы флаг flicker_suspected действительно заблокировал
    # публикацию (см. quality.assess_quality). Мерцание освещения — стабильное
    # во времени явление; разовое совпадение на одном окне гораздо вероятнее
    # объясняется случайным шумовым пиком (см. flicker_min_raw_std выше), чем
    # реальным мерцанием. TODO(калибровка).
    flicker_min_consecutive_windows: int = 3

    # --- Веса компонент overall score (п.20: temporal_consistency — 4-я
    # компонента). Должны суммироваться в 1.0. spectral_snr остаётся
    # доминирующим весом, т.к. это единственный компонент, напрямую
    # измеряющий наличие пульсовой составляющей (см. quality.assess_quality).
    weight_spectral_snr: float = 0.40
    weight_cross_roi: float = 0.25
    weight_landmark_stability: float = 0.15
    weight_temporal_consistency: float = 0.20


@dataclass
class HRVConfig:
    """Пороги HRV — окно HRV отдельно от окна BPM (см. hrv/features.py и
    RPPGPipeline._update_ibi_log/_maybe_compute_hrv, п.14 требований):
    BPM оценивается спектрально по короткому скользящему окну (WindowConfig),
    а HRV time/frequency-domain метрикам по Task Force (1996) нужны
    существенно более длинные ряды IBI, поэтому они копятся отдельно."""

    pnn_threshold_ms: float = 50.0
    pnn20_threshold_ms: float = 20.0
    compute_frequency_domain: bool = True
    edge_trim_seconds: float = 1.0
    # Порог доли отбракованных (ectopic_artifact_mask) интервалов, выше
    # которого HRV за окно не публикуется (п.17, стандарт Kubios/neurokit2).
    max_artifact_fraction: float = 0.05
    # Порог "скачка" IBI относительно соседей для маскирования как
    # физиологически неправдоподобного (п.16).
    ectopic_max_relative_change: float = 0.4
    # Минимум накопленного ряда IBI перед первой публикацией HRV и
    # предпочтительная длина (Task Force, 1996 — 5 минут; п.14).
    min_accumulation_seconds: float = 120.0
    target_accumulation_seconds: float = 300.0
    # HRV пересчитывается не на каждом BPM-шаге, а раз в это число секунд.
    step_seconds: float = 60.0
    # Минимальная длительность ряда IBI для LF/HF (п.15): LF 0.04-0.15 Гц
    # -> периоды 6.7-25с, HF 0.15-0.4 Гц -> периоды 2.5-6.7с.
    lf_min_duration_seconds: float = 120.0
    hf_min_duration_seconds: float = 60.0


@dataclass
class AccelerationConfig:
    use_numba: bool = True
    use_onnx_for_learned_method: bool = False
    onnx_model_path: str | None = None


@dataclass
class FusionConfig:
    """SQI-взвешенное объединение сигналов нескольких ROI/модальностей
    вместо argmax-выбора одного "лучшего" по SNR (п.34 требований, см.
    signal/fusion.py). Выключено по умолчанию — включение НЕ меняет
    поведение существующих конфигов до явного opt-in, т.к. выигрыш от
    fusion относительно argmax нужно доказывать экспериментально
    (см. scripts/compare_fusion_vs_argmax.py), а не считать данностью."""

    enabled: bool = False
    # Помимо выбранного цветового метода (self.method), дополнительно
    # считать head-motion канал НАРЯДУ с ним (а не вместо) — без этого
    # fusion работал бы только по 3 ROI ОДНОЙ модальности, теряя "двух
    # модальностей" из формулировки п.34. Игнорируется, если method уже
    # head_motion (тогда это единственная модальность).
    include_head_motion: bool = True
    # Окно поиска лага при выравнивании сигналов перед суммированием
    # (color-rPPG и head-motion физически разные явления и не гарантированно
    # синфазны, см. signal/fusion.py::_align_sign_and_lag).
    max_lag_seconds: float = 0.3
    # Диапазон spectral SNR (дБ), отображаемый в вес источника [0,1] —
    # см. signal/fusion.snr_db_to_weight.
    weight_floor_db: float = -5.0
    weight_ceil_db: float = 15.0


@dataclass
class TrackerConfig:
    """Задача 12: сглаживание ОТОБРАЖАЕМОГО (не измеренного) BPM поверх
    честных пооконных PTSDPulseFeatures — см. signal/tracker.py::PulseTracker.
    НЕ часть PipelineConfig/RPPGPipeline — это утилита слоя представления
    (используется, например, в scripts/run_webcam.py, задача 14), а не
    часть самого сигнального пайплайна: RPPGPipeline отдаёт честные
    пооконные измерения, а сглаживание для ЭКРАНА — отдельная, опциональная
    надстройка над ними, которую потребитель включает сам, если она ему
    нужна."""

    window_size: int = 5
    # ЧЕСТНО: это сглаживание ОТОБРАЖЕНИЯ (частота обновления числа на
    # экране), а НЕ ограничение на то, какое измерение допустимо — в
    # отличие от прежнего BestEffortConfig.max_change_per_step_bpm, который
    # сглаживал шаги ВОКРУГ выдуманной константы (задача 8). Здесь
    # ограничивается изменение УЖЕ ИЗМЕРЕННОГО (взвешенно-усреднённого)
    # значения. None -> без сглаживания. TODO(калибровка).
    max_change_per_step_bpm: float | None = 5.0


@dataclass
class PipelineConfig:
    """Верхнеуровневая конфигурация — то, что передаётся в RPPGPipeline.

    Задача 8: раньше здесь была секция best_effort (BestEffortConfig) —
    сглаженный BPM, который ВСЕГДА был реальным числом в правдоподобном
    диапазоне, даже при полном отсутствии сигнала (стартовал с
    fallback_bpm=75). Удалена целиком (вариант А задачи 8, а не
    переименование/пометка deprecated): это была не "менее точная оценка",
    а генератор правдоподобного числа человеческого пульса по требованию —
    неотличимый от настоящего измерения на экране/в интеграции. Вместо неё
    PTSDPulseFeatures отдаёт честную пару last_valid_bpm/last_valid_bpm_age_s
    (см. pipeline.py::RPPGPipeline._update_last_valid_bpm)."""

    face: FaceModelConfig = field(default_factory=FaceModelConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    hrv: HRVConfig = field(default_factory=HRVConfig)
    accel: AccelerationConfig = field(default_factory=AccelerationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    method: ExtractionMethod = ExtractionMethod.POS
    frequency_method: FrequencyMethod = FrequencyMethod.WELCH

    @staticmethod
    def default_model_path() -> Path:
        return Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
