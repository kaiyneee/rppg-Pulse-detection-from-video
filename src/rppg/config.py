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
    tarvainen_lambda: float = 300.0
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
    """Пороги Signal Quality Index (пункт "Оценка качества сигнала" ТЗ)."""

    min_spectral_snr_db: float = 3.0
    max_cross_roi_bpm_diff: float = 8.0
    min_landmark_stability: float = 0.5
    # Итоговый BPM передаётся в систему ПТСР только если score >= порога.
    min_overall_score_to_publish: float = 0.5


@dataclass
class HRVConfig:
    pnn_threshold_ms: float = 50.0
    pnn20_threshold_ms: float = 20.0
    compute_frequency_domain: bool = True


@dataclass
class AccelerationConfig:
    use_numba: bool = True
    use_onnx_for_learned_method: bool = False
    onnx_model_path: str | None = None


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
    method: ExtractionMethod = ExtractionMethod.POS
    frequency_method: FrequencyMethod = FrequencyMethod.WELCH

    @staticmethod
    def default_model_path() -> Path:
        return Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
