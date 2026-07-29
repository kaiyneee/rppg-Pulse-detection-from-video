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
    min_seconds_before_estimate: float = 4.0  # было "3*fps" в оригинале


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

    min_spectral_snr_db: float = 3.0
    max_cross_roi_bpm_diff: float = 8.0
    min_landmark_stability: float = 0.5
    # Итоговый BPM передаётся в систему ПТСР только если score >= порога.
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

    # --- п.21: гармоники/субгармоники ---
    # Порог отношения мощностей (энергия у 2f или f/2) / (энергия у f), при
    # превышении которого пик считается подозрительным на гармоническую
    # путаницу. 0.7 выбран с запасом над типичной долей мощности физиологической
    # 2-й гармоники PPG-волны (амплитудное отношение ~0.2-0.4 -> степенное
    # ~0.04-0.16), чтобы не штрафовать нормальный несинусоидальный пульс.
    # TODO(калибровка) на реальных данных с разметкой ложных гармонических
    # срабатываний.
    harmonic_ratio_threshold: float = 0.7

    # --- п.22: мерцание освещения ---
    # Допуск (Hz) для совпадения частоты фонового и лицевого пика.
    flicker_freq_tolerance_hz: float = 0.15
    # Минимальный spectral SNR фонового ROI, чтобы его пик вообще
    # рассматривался как "узкий и стабильный" (а не случайный шумовой максимум).
    # TODO(калибровка).
    flicker_min_background_snr_db: float = 3.0

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
class PipelineConfig:
    """Верхнеуровневая конфигурация — то, что передаётся в RPPGPipeline."""

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
