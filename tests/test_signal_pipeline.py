"""
Тесты на синтетических данных.

Осознанное ограничение (см. docs/research_report.md, раздел 8.4): в этой
среде нет доступа к реальным видео/камере и к MediaPipe-модели (файл модели
не скачать без сети), поэтому здесь проверяется весь сигнальный конвейер —
detrend/normalize/bandpass -> 6 методов извлечения -> 3 метода частотного
анализа -> HRV-признаки -- на синтетических, но физиологически правдоподобных
данных с известным "истинным" BPM. Это не заменяет валидацию на реальных
датасетах (VIPL-HR/UBFC-rPPG/PURE/MMSE-HR, см. benchmark/evaluate.py), но
достоверно проверяет, что математика реализована без ошибок.

Запуск: PYTHONPATH=src python3 -m pytest tests/ -v
        (или просто: PYTHONPATH=src python3 tests/test_signal_pipeline.py)
"""

from __future__ import annotations

import numpy as np

from rppg.signal.preprocessing import bandpass_filter, detrend, preprocess_signal
from rppg.signal.methods import SignalWindow, get_method
from rppg.signal.frequency import estimate_hr
from rppg.hrv.features import extract_hrv_features, detect_pulse_peaks, hrv_time_domain


FPS = 30.0
DURATION_S = 15.0
BAND = (0.7, 4.0)


def make_synthetic_rgb(bpm: float, fps: float = FPS, duration_s: float = DURATION_S,
                        noise_std: float = 0.3, seed: int = 0) -> np.ndarray:
    """Синтетический ROI-трейс R,G,B с известным BPM.

    Физиологически правдоподобные допущения:
      - зелёный канал имеет наибольшую пульсовую модуляцию, синий — наименьшую
        (Verkruysse et al., 2008);
      - форма волны — не чистая синусоида, а сумма 2 гармоник (грубая имитация
        асимметричной формы PPG-волны с быстрым систолическим фронтом);
      - амплитуда пульсовой модуляции ~1% от DC-уровня — реалистичный порядок
        величины для rPPG (в отличие от синтетики "для красоты" с амплитудой
        в десятки процентов, которая тривиально решается почти любым методом).
    """
    rng = np.random.default_rng(seed)
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    f = bpm / 60.0

    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t - 0.5)
    pulse /= np.max(np.abs(pulse))

    gains = {"R": 0.6, "G": 1.0, "B": 0.35}
    dc = {"R": 150.0, "G": 110.0, "B": 90.0}
    amp = 1.1

    channels = []
    for ch in ("R", "G", "B"):
        sig = dc[ch] + gains[ch] * amp * pulse + rng.normal(0, noise_std, n)
        channels.append(sig)
    return np.stack(channels, axis=1)


def make_synthetic_motion_trajectories(bpm: float, n_points: int = 40, fps: float = FPS,
                                        duration_s: float = DURATION_S, seed: int = 1) -> np.ndarray:
    """Синтетика для head-motion метода: (T, N, 2), где Y-компонента части
    точек несёт общий периодический сигнал на частоте пульса + шум/дрейф."""
    rng = np.random.default_rng(seed)
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    f = bpm / 60.0
    common = 0.6 * np.sin(2 * np.pi * f * t)

    traj = np.zeros((n, n_points, 2))
    for p in range(n_points):
        phase = rng.uniform(-0.1, 0.1)
        gain = rng.uniform(0.5, 1.0)
        drift = 0.02 * np.cumsum(rng.normal(0, 1, n))  # медленный дрейф трекинга
        y = 100 + gain * common * np.cos(phase) + drift + rng.normal(0, 0.15, n)
        x = 100 + rng.normal(0, 0.15, n)
        traj[:, p, 0] = x
        traj[:, p, 1] = y
    return traj


def _assert_bpm_close(estimated: float, true_bpm: float, tol_bpm: float, label: str) -> None:
    assert not np.isnan(estimated), f"{label}: BPM = NaN"
    diff = abs(estimated - true_bpm)
    assert diff <= tol_bpm, f"{label}: |{estimated:.2f} - {true_bpm:.2f}| = {diff:.2f} > tol {tol_bpm}"
    print(f"  [OK] {label}: истинный={true_bpm:.1f} BPM, оценка={estimated:.2f} BPM, "
          f"ошибка={diff:.2f} BPM")


def test_color_based_methods_recover_known_bpm():
    print("\n=== Тест: GREEN/CHROM/POS/PCA/ICA восстанавливают известный BPM ===")
    true_bpm = 72.0
    rgb = make_synthetic_rgb(true_bpm)
    window = SignalWindow(rgb_traces={"forehead": rgb}, fps=FPS, hr_band_hz=BAND)

    for method_name, tol in [("green", 2.0), ("chrom", 2.0), ("pos", 2.5), ("pca", 3.0), ("ica", 3.5)]:
        method = get_method(method_name)
        raw = method.extract(window, "forehead")
        processed = preprocess_signal(raw, FPS, *BAND, detrend_method="tarvainen", normalize_method="zscore")
        est = estimate_hr(processed, FPS, BAND, method="welch")
        _assert_bpm_close(est.bpm, true_bpm, tol, f"{method_name.upper()} + Welch")


def test_head_motion_method_recovers_known_bpm():
    print("\n=== Тест: HEAD_MOTION восстанавливает известный BPM ===")
    true_bpm = 68.0
    traj = make_synthetic_motion_trajectories(true_bpm)
    window = SignalWindow(rgb_traces={}, fps=FPS, landmark_trajectories=traj, hr_band_hz=BAND)

    method = get_method("head_motion")
    raw = method.extract(window, None)
    processed = preprocess_signal(raw, FPS, *BAND, detrend_method="tarvainen", normalize_method="zscore")
    est = estimate_hr(processed, FPS, BAND, method="welch")
    _assert_bpm_close(est.bpm, true_bpm, 4.0, "HEAD_MOTION + Welch")


def test_frequency_estimators_agree_on_clean_tone():
    print("\n=== Тест: FFT/Welch/Lomb-Scargle согласованы на чистом тоне ===")
    true_bpm = 90.0
    rgb = make_synthetic_rgb(true_bpm, noise_std=0.15)
    method = get_method("pos")
    window = SignalWindow(rgb_traces={"forehead": rgb}, fps=FPS, hr_band_hz=BAND)
    raw = method.extract(window, "forehead")
    processed = preprocess_signal(raw, FPS, *BAND)

    for fm, tol in [("fft", 2.0), ("welch", 2.5), ("lomb_scargle", 2.5)]:
        est = estimate_hr(processed, FPS, BAND, method=fm)
        _assert_bpm_close(est.bpm, true_bpm, tol, f"POS + {fm}")


def test_lombscargle_handles_irregular_sampling():
    """Ключевое обоснование включения Lomb-Scargle: он остаётся корректным,
    когда часть кадров исключена (окклюзия) и временная сетка неравномерна —
    ситуация, где FFT/Welch формально некорректны без интерполяции."""
    print("\n=== Тест: Lomb-Scargle на НЕРАВНОМЕРНОЙ сетке (после удаления кадров) ===")
    true_bpm = 75.0
    rgb = make_synthetic_rgb(true_bpm, noise_std=0.15)
    method = get_method("chrom")
    window = SignalWindow(rgb_traces={"forehead": rgb}, fps=FPS, hr_band_hz=BAND)
    raw = method.extract(window, "forehead")
    processed = preprocess_signal(raw, FPS, *BAND)

    rng = np.random.default_rng(2)
    keep_mask = rng.random(len(processed)) > 0.15  # "окклюзия" ~15% кадров случайно
    timestamps = (np.arange(len(processed)) / FPS)[keep_mask]
    irregular_signal = processed[keep_mask]

    from rppg.signal.frequency import estimate_hr_lombscargle
    est = estimate_hr_lombscargle(timestamps, irregular_signal, BAND)
    _assert_bpm_close(est.bpm, true_bpm, 3.0, "CHROM + Lomb-Scargle (нерегулярная сетка)")


def test_bandpass_rejects_out_of_band_and_keeps_in_band():
    print("\n=== Тест: Butterworth bandpass пропускает целевую полосу и режет вне неё ===")
    n = int(FPS * 12)
    t = np.arange(n) / FPS
    in_band_tone = np.sin(2 * np.pi * 1.2 * t)          # 72 BPM — внутри (0.7,4.0) Hz
    out_of_band_tone = np.sin(2 * np.pi * 0.05 * t)     # медленный дрейф — вне полосы
    mixed = in_band_tone + 3.0 * out_of_band_tone

    filtered = bandpass_filter(mixed, FPS, *BAND, order=4)

    # После фильтрации амплитуда, соответствующая полосовому тону, должна
    # доминировать, а не быть подавленной дрейфом.
    from numpy.fft import rfft, rfftfreq
    spec = np.abs(rfft(filtered))
    freqs = rfftfreq(n, d=1 / FPS)
    power_in_band = spec[(freqs > 1.0) & (freqs < 1.4)].max()
    power_near_dc = spec[(freqs > 0.0) & (freqs < 0.3)].max()

    assert power_in_band > 5 * power_near_dc, (
        f"bandpass не подавил внеполосный дрейф: in_band={power_in_band:.2f}, "
        f"near_dc={power_near_dc:.2f}"
    )
    print(f"  [OK] мощность в полосе {power_in_band:.2f} >> мощность у 0 Hz {power_near_dc:.2f}")


def test_detrend_removes_slow_trend():
    print("\n=== Тест: detrend убирает медленный тренд, сохраняя быструю компоненту ===")
    n = int(FPS * 12)
    t = np.arange(n) / FPS
    fast = np.sin(2 * np.pi * 1.3 * t)
    trend = 0.08 * t**2  # квадратичный дрейф (Tarvainen должен убрать лучше линейного)
    x = fast + trend

    for method in ("linear", "tarvainen"):
        y = detrend(x, method=method, lam=300.0)
        # Остаточный тренд: разница между началом и концом сигнала должна
        # резко уменьшиться относительно исходной.
        edge_diff_before = abs(np.mean(x[-30:]) - np.mean(x[:30]))
        edge_diff_after = abs(np.mean(y[-30:]) - np.mean(y[:30]))
        assert edge_diff_after < edge_diff_before, f"{method}: тренд не уменьшился"
        print(f"  [OK] {method}: edge_diff {edge_diff_before:.3f} -> {edge_diff_after:.3f}")


def test_hrv_features_match_known_ibi_statistics():
    print("\n=== Тест: HRV-признаки совпадают с аналитически ожидаемыми ===")
    rng = np.random.default_rng(3)
    mean_ibi_ms = 800.0  # 75 BPM
    true_sdnn = 40.0
    ibi = rng.normal(mean_ibi_ms, true_sdnn, 200)
    ibi = np.clip(ibi, 500, 1200)

    result = hrv_time_domain(ibi_ms=np.array([]))  # sanity: пустой ввод не падает
    assert np.isnan(result["mean_hr_bpm"])

    peak_times_s = np.cumsum(ibi) / 1000.0
    fps = 100.0  # высокая fps для точной сетки пиков в тесте
    n = int(peak_times_s[-1] * fps) + 10
    signal = np.zeros(n)
    peak_indices = (peak_times_s * fps).astype(int)
    peak_indices = peak_indices[peak_indices < n]
    signal[peak_indices] = 1.0
    # "размываем" импульсы в подобие пиков, чтобы find_peaks мог их различить
    from scipy.ndimage import gaussian_filter1d
    smooth_signal = gaussian_filter1d(signal, sigma=2)

    hrv = extract_hrv_features(smooth_signal, fps, compute_frequency_domain=False)

    print(f"  Заданные: mean_HR~{60000/mean_ibi_ms:.1f} BPM, SDNN~{true_sdnn:.1f} ms")
    print(f"  Получено: mean_HR={hrv.mean_hr_bpm:.1f} BPM, SDNN={hrv.sdnn_ms:.1f} ms, "
          f"RMSSD={hrv.rmssd_ms:.1f} ms, pNN50={hrv.pnn50_pct:.1f}%")

    assert abs(hrv.mean_hr_bpm - 60000 / mean_ibi_ms) < 3.0
    assert abs(hrv.sdnn_ms - true_sdnn) < 12.0  # find_peaks вносит небольшую погрешность привязки
    assert hrv.n_beats > 100


def run_all():
    tests = [
        test_color_based_methods_recover_known_bpm,
        test_head_motion_method_recovers_known_bpm,
        test_frequency_estimators_agree_on_clean_tone,
        test_lombscargle_handles_irregular_sampling,
        test_bandpass_rejects_out_of_band_and_keeps_in_band,
        test_detrend_removes_slow_trend,
        test_hrv_features_match_known_ibi_statistics,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed.append((test.__name__, str(e)))
            print(f"  [FAIL] {test.__name__}: {e}")

    print("\n" + "=" * 60)
    if failed:
        print(f"ИТОГ: {len(failed)}/{len(tests)} тестов провалено")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    else:
        print(f"ИТОГ: все {len(tests)} тестов пройдены успешно")


if __name__ == "__main__":
    run_all()
