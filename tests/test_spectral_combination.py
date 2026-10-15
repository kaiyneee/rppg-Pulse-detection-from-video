"""
Тесты оценки частоты по объединённому спектру нескольких ROI
(ROIConfig.spectral_combination="spectral_mean") и публичного выбора пика
по готовому спектру (frequency.select_peak_from_spectrum).
"""

from __future__ import annotations

import numpy as np
import pytest

from rppg.config import PipelineConfig
from rppg.signal.frequency import FrequencyEstimate, select_peak_from_spectrum


def _spectrum(freqs: np.ndarray, peaks: list[tuple[float, float]], floor: float = 0.01) -> np.ndarray:
    """Гауссовы пики (частота, высота) на ровном фоне."""
    s = np.full_like(freqs, floor)
    for f0, h in peaks:
        s += h * np.exp(-0.5 * ((freqs - f0) / 0.03) ** 2)
    return s


FREQS = np.linspace(0.7, 4.0, 400)


def test_select_peak_from_spectrum_finds_global_maximum():
    est = select_peak_from_spectrum(FREQS, _spectrum(FREQS, [(1.25, 1.0), (2.1, 0.4)]), "fft")
    assert est.freq_hz == pytest.approx(1.25, abs=0.01)
    assert est.bpm == pytest.approx(75.0, abs=0.6)


def test_select_peak_from_spectrum_respects_continuity_between_comparable_peaks():
    """Два почти равных пика: непрерывность выбирает ближний к истории —
    та же логика, что внутри estimate_hr."""
    power = _spectrum(FREQS, [(1.25, 1.0), (1.9, 1.02)])
    without = select_peak_from_spectrum(FREQS, power, "fft")
    with_history = select_peak_from_spectrum(FREQS, power, "fft", recent_median_hz=1.24,
                                             continuity_power_tolerance=0.15)
    assert without.freq_hz == pytest.approx(1.9, abs=0.01)
    assert with_history.freq_hz == pytest.approx(1.25, abs=0.01)


def test_select_peak_from_spectrum_handles_empty_input():
    est = select_peak_from_spectrum(np.array([]), np.array([]), "fft")
    assert np.isnan(est.bpm)


def _pipeline_stub(cfg: PipelineConfig):
    """Минимальный объект с методами пайплайна, нужными _spectral_mean_estimate,
    без инициализации MediaPipe."""
    from rppg.pipeline import RPPGPipeline

    stub = RPPGPipeline.__new__(RPPGPipeline)
    stub.cfg = cfg
    stub._recent_bpm_history = []
    return stub


def test_spectral_mean_recovers_shared_peak_that_single_best_roi_misses():
    """У каждого ROI свой случайный ложный пик, иногда выше истинного;
    истинный пульс (1.3 Гц) общий для всех. Один ROI ошибается, среднее —
    нет: на этом и основан выигрыш режима spectral_mean."""
    cfg = PipelineConfig()
    cfg.roi.spectral_combination = "spectral_mean"
    stub = _pipeline_stub(cfg)
    per_roi = {
        "forehead": FrequencyEstimate(np.nan, np.nan, "fft", FREQS, _spectrum(FREQS, [(1.3, 1.0), (2.4, 1.3)])),
        "left_cheek": FrequencyEstimate(np.nan, np.nan, "fft", FREQS, _spectrum(FREQS, [(1.3, 1.0), (0.9, 0.6)])),
        "right_cheek": FrequencyEstimate(np.nan, np.nan, "fft", FREQS, _spectrum(FREQS, [(1.3, 1.0), (3.1, 0.7)])),
    }
    single = per_roi["forehead"]
    assert FREQS[np.argmax(single.power)] == pytest.approx(2.4, abs=0.01)  # один ROI ошибся
    combined = stub._spectral_mean_estimate(per_roi)
    assert combined is not None
    assert combined.freq_hz == pytest.approx(1.3, abs=0.01)


def test_spectral_mean_needs_two_rois_on_the_same_grid():
    stub = _pipeline_stub(PipelineConfig())
    one = {"forehead": FrequencyEstimate(np.nan, np.nan, "fft", FREQS, _spectrum(FREQS, [(1.3, 1.0)]))}
    assert stub._spectral_mean_estimate(one) is None
    other_grid = np.linspace(0.7, 4.0, 300)
    mismatched = {
        "forehead": FrequencyEstimate(np.nan, np.nan, "fft", FREQS, _spectrum(FREQS, [(1.3, 1.0)])),
        "left_cheek": FrequencyEstimate(np.nan, np.nan, "fft", other_grid, _spectrum(other_grid, [(1.3, 1.0)])),
    }
    assert stub._spectral_mean_estimate(mismatched) is None


def test_default_config_is_fft_spectral_mean():
    cfg = PipelineConfig()
    assert cfg.preset == "fft_spectral_mean"
    assert cfg.frequency_method.value == "fft"
    assert cfg.roi.spectral_combination == "spectral_mean"


def test_legacy_preset_restores_configuration_of_both_papers():
    from rppg.config import make_pipeline_config

    cfg = make_pipeline_config("legacy_welch8")
    assert cfg.preset == "legacy_welch8"
    assert cfg.frequency_method.value == "welch"
    assert cfg.roi.spectral_combination == "best_snr"


def test_unknown_preset_is_rejected():
    from rppg.config import make_pipeline_config

    with pytest.raises(ValueError):
        make_pipeline_config("no_such_preset")


def test_config_file_without_preset_keeps_its_original_meaning():
    """Конфиг, записанный до появления пресетов, достраивается от прежних
    умолчаний, а не от текущих: иначе старый эксперимент молча сменил бы
    метод оценки частоты и способ объединения ROI."""
    from rppg.config_io import config_from_dict

    partial = config_from_dict({"window": {"window_seconds": 12.0}})
    assert partial.preset == "legacy_welch8"
    assert partial.frequency_method.value == "welch"
    assert partial.roi.spectral_combination == "best_snr"
    assert partial.window.window_seconds == 12.0


def test_explicit_values_in_old_file_win_over_preset():
    from rppg.config_io import config_from_dict

    cfg = config_from_dict({"frequency_method": "lomb_scargle"})
    assert cfg.frequency_method.value == "lomb_scargle"
    assert cfg.roi.spectral_combination == "best_snr"


def test_config_roundtrip_preserves_preset():
    from rppg.config import make_pipeline_config
    from rppg.config_io import config_from_dict, config_to_dict

    for name in ("fft_spectral_mean", "legacy_welch8"):
        original = make_pipeline_config(name)
        restored = config_from_dict(config_to_dict(original))
        assert restored.preset == name
        assert restored.frequency_method == original.frequency_method
        assert restored.roi.spectral_combination == original.roi.spectral_combination
