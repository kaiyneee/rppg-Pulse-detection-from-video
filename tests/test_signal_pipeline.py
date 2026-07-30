"""
Тесты на синтетических данных.

Большинство тестов здесь проверяют сигнальный конвейер — detrend/normalize/
bandpass -> 6 методов извлечения -> 3 метода частотного анализа ->
HRV-признаки -- на синтетических, но физиологически правдоподобных данных
с известным "истинным" BPM, БЕЗ MediaPipe. Это не заменяет валидацию на
реальных датасетах (VIPL-HR/UBFC-rPPG/PURE/MMSE-HR, см. benchmark/evaluate.py),
но достоверно проверяет, что математика реализована без ошибок.

MediaPipe и модель Face Landmarker (models/face_landmarker.task) в этой
среде ДОСТУПНЫ (в отличие от более ранней версии этого докстринга) — часть
тестов (test_occluded_face_pipeline_never_publishes, test_end_to_end_*)
поэтому запускает РЕАЛЬНЫЙ RPPGPipeline целиком, а не только signal-уровень,
и грациозно пропускается (не падает), если модель/пакет всё же недоступны в
какой-то другой среде запуска.

Запуск: PYTHONPATH=src python3 -m pytest tests/ -v
        (или просто: PYTHONPATH=src python3 tests/test_signal_pipeline.py)
"""

from __future__ import annotations

import numpy as np

from rppg.signal.preprocessing import bandpass_filter, detrend, preprocess_signal
from rppg.signal.methods import SignalWindow, get_method
from rppg.signal.frequency import estimate_hr
from rppg.signal.respiration import estimate_respiration_rate
from rppg.hrv.features import (
    extract_hrv_features,
    extract_hrv_features_from_ibi,
    detect_pulse_peaks,
    refine_peaks_subsample,
    compute_ibi_ms,
    ectopic_artifact_mask,
    hrv_time_domain,
    hrv_frequency_domain,
)


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


def test_pos_numba_matches_numpy_reference():
    """Регрессия для п.45: PosMethod.extract утверждает в докстринге, что
    numba и numpy-реализации overlap-add дают "идентичный результат" — это
    нужно доказать тестом, а не принять на слово. Сравниваются НАПРЯМУЮ
    низкоуровневые overlap-add функции (pos_overlap_add_numba vs
    PosMethod._pos_overlap_add_reference) на ОДНОМ И ТОМ ЖЕ сыром RGB-входе,
    а не через полный сигнальный конвейер, где детрендинг/bandpass могли бы
    замаскировать расхождение."""
    print("\n=== Тест: POS numba и numpy-эталон дают идентичный результат (п.45) ===")
    from rppg.accel.fast_ops import pos_overlap_add_numba, NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        print("  [SKIP] numba не установлена в этой среде")
        return

    method = get_method("pos")
    rgb = make_synthetic_rgb(75.0, noise_std=0.4, seed=9)
    ws = max(3, int(round(method.window_seconds * FPS)))

    numba_result = pos_overlap_add_numba(rgb, ws, method.projection_matrix)
    numpy_result = method._pos_overlap_add_reference(rgb, ws)

    max_diff = float(np.max(np.abs(numba_result - numpy_result)))
    print(f"  максимальное расхождение numba vs numpy: {max_diff:.2e}")
    assert np.allclose(numba_result, numpy_result, atol=1e-9), (
        f"numba и numpy реализации POS должны давать идентичный результат, "
        f"макс. расхождение={max_diff:.2e}"
    )


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


def test_tarvainen_cutoff_matches_documented_reference_and_new_default_is_transparent_to_pulse_band():
    """Регрессия для п.36: tarvainen_lambda=300 в исходном коде был взят из
    HRV-литературы БЕЗ проверки, что он означает на fs=30 Hz (видео), а не
    на типичный для HRV-практики ресэмплинг RR-тахограммы на 4 Hz.

    Два независимых утверждения проверяются здесь:
      (1) сама формула АЧХ (tarvainen_cutoff_hz) даёт ПРАВИЛЬНЫЙ результат —
          сверено с независимо документированным эталоном (Kubios/PhysioData
          Toolbox: lambda=500 @ fs=4 Hz -> cutoff~=0.04 Hz);
      (2) новый дефолт (config.FilterConfig.tarvainen_lambda) на fs=30 Hz
          почти не трогает нижний край пульсовой полосы (0.7 Hz) — старое
          значение 300 давало заметное (2.36%) затухание уже на 0.7 Hz."""
    print("\n=== Тест: АЧХ tarvainen_detrend сверена с документированным эталоном (п.36) ===")
    from rppg.signal.preprocessing import tarvainen_cutoff_hz, tarvainen_frequency_response
    from rppg.config import FilterConfig

    ref_cutoff = tarvainen_cutoff_hz(500.0, 4.0)
    print(f"  lambda=500 @ fs=4 Hz -> cutoff={ref_cutoff:.4f} Hz (эталон ~0.04 Hz)")
    assert abs(ref_cutoff - 0.04) < 0.01, "формула АЧХ должна воспроизводить документированный эталон Kubios/PhysioData"

    old_cutoff_at_our_fs = tarvainen_cutoff_hz(300.0, FPS)
    print(f"  lambda=300 (старое значение) @ fs={FPS} Hz -> cutoff={old_cutoff_at_our_fs:.4f} Hz")
    assert old_cutoff_at_our_fs > 0.25, (
        "sanity: перенос lambda=300 без пересчёта на fs=30 действительно даёт "
        "cutoff в разы выше, чем предполагала HRV-литература (~0.04-0.05 Hz)"
    )

    new_lambda = FilterConfig().tarvainen_lambda
    _, h_old = tarvainen_frequency_response(300.0, FPS, freqs_hz=np.array([BAND[0]]))
    _, h_new = tarvainen_frequency_response(new_lambda, FPS, freqs_hz=np.array([BAND[0]]))
    print(f"  затухание на {BAND[0]} Hz: старое={100 * (1 - h_old[0]):.2f}%  новое={100 * (1 - h_new[0]):.3f}%")
    assert h_new[0] > 0.999, "новый дефолт должен быть почти прозрачен (>99.9%) для нижнего края пульсовой полосы"
    assert h_new[0] > h_old[0], "новый дефолт должен затухать на пульсовой полосе МЕНЬШЕ, чем старый"


def test_align_sign_and_lag_recovers_known_shift():
    """Регрессия для п.34: _align_sign_and_lag должна точно восстанавливать
    известный целочисленный сдвиг и инверсию знака — на этом инварианте
    держится вся fuse_signals_by_sqi (без него суммирование сигналов из
    разных модальностей рискует деструктивной интерференцией, см.
    docstring signal/fusion.py)."""
    print("\n=== Тест: выравнивание фазы/знака восстанавливает известный сдвиг (п.34) ===")
    from rppg.signal.fusion import _align_sign_and_lag

    n = 300
    t = np.arange(n) / FPS
    reference = np.sin(2 * np.pi * 1.2 * t)

    true_shift = 5
    shifted_and_flipped = -np.roll(reference, true_shift)

    aligned, recovered_lag, corr = _align_sign_and_lag(shifted_and_flipped, reference, max_lag_samples=20)
    print(f"  применённый сдвиг={true_shift}, восстановленный лаг={recovered_lag}, corr={corr:.4f}")
    edge = 20
    max_diff = np.max(np.abs(aligned[edge:-edge] - reference[edge:-edge]))
    print(f"  макс. отклонение после выравнивания (без краёв): {max_diff:.6f}")
    assert corr > 0.99, "после коррекции знака корреляция с референсом должна быть близка к 1"
    assert max_diff < 1e-6, "выровненный сигнал должен практически совпасть с референсом"


def test_fuse_signals_by_sqi_weighting_favors_clean_source():
    """Регрессия для п.34: источник с более высоким весом (например, из
    более высокого spectral SNR) должен доминировать в объединённом
    сигнале — грязный (шумный) источник с низким весом не должен портить
    результат, который дал бы чистый источник сам по себе."""
    print("\n=== Тест: SQI-взвешенное fusion — чистый источник доминирует над шумным ===")
    from rppg.signal.fusion import fuse_signals_by_sqi

    n = 300
    t = np.arange(n) / FPS
    true_signal = np.sin(2 * np.pi * 1.2 * t)
    rng = np.random.default_rng(0)

    clean = true_signal + rng.normal(0, 0.05, n)
    noisy = rng.normal(0, 3.0, n)  # почти чистый шум, не несёт пульса

    fused, diag = fuse_signals_by_sqi(
        {"clean": clean, "noisy": noisy},
        {"clean": 0.95, "noisy": 0.05},
        fps=FPS,
    )
    corr_fused = np.corrcoef(fused, true_signal)[0, 1]
    corr_noisy_alone = np.corrcoef(noisy, true_signal)[0, 1]
    print(f"  вес clean={diag['clean']['weight']:.3f}, вес noisy={diag['noisy']['weight']:.3f}")
    print(f"  corr(fused, true)={corr_fused:.3f}  corr(noisy_alone, true)={corr_noisy_alone:.3f}")
    assert corr_fused > 0.9, "при доминирующем весе чистого источника fused должен сильно коррелировать с истиной"


def test_fuse_signals_by_sqi_aligns_out_of_phase_source_without_cancellation():
    """Регрессия для п.34: если один источник (например, head-motion) на
    самом деле несёт тот же пульс, но со сдвигом фазы/инвертированным
    знаком относительно другого (color-rPPG) — БЕЗ выравнивания наивное
    суммирование могло бы дать деструктивную интерференцию (ослабление
    сигнала вместо усиления). fuse_signals_by_sqi обязана выровнять источник
    перед суммированием (см. _align_sign_and_lag), а не наивно складывать."""
    print("\n=== Тест: fusion выравнивает несинфазный источник вместо деструктивной интерференции ===")
    from rppg.signal.fusion import fuse_signals_by_sqi

    n = 300
    t = np.arange(n) / FPS
    rng = np.random.default_rng(2)
    true_signal = np.sin(2 * np.pi * 1.2 * t)

    source_a = true_signal + rng.normal(0, 0.05, n)
    # source_b несёт тот же пульс, но инвертирован по знаку и сдвинут по фазе
    # (имитация другой физической модальности, см. модульный docstring fusion.py).
    source_b = -np.roll(true_signal, 4) + rng.normal(0, 0.05, n)

    fused, diag = fuse_signals_by_sqi({"a": source_a, "b": source_b}, {"a": 0.6, "b": 0.4}, fps=FPS)

    naive_sum = source_a + source_b  # то, что получилось бы БЕЗ выравнивания
    rms_fused = np.sqrt(np.mean(fused**2))
    rms_naive = np.sqrt(np.mean(naive_sum**2))
    corr_fused_true = np.corrcoef(fused, true_signal)[0, 1]

    print(f"  лаг источника b относительно a: {diag['b']['lag_samples']} отсчётов, "
          f"corr после выравнивания={diag['b']['corr_with_reference']:.3f}")
    print(f"  RMS(fused)={rms_fused:.3f} vs RMS(наивная сумма без выравнивания)={rms_naive:.3f}")
    print(f"  corr(fused, true)={corr_fused_true:.3f}")

    assert corr_fused_true > 0.9, "после выравнивания fused должен сильно коррелировать с истинным пульсом"
    assert rms_fused > rms_naive, (
        "выровненное объединение должно давать БОЛЬШУЮ амплитуду, чем наивная сумма без "
        "выравнивания знака/фазы (которая для инвертированного источника частично гасит сигнал)"
    )


def test_fuse_signals_by_sqi_zero_weights_fallback_to_uniform():
    print("\n=== Тест: fusion при всех нулевых весах не падает, а использует равномерные ===")
    from rppg.signal.fusion import fuse_signals_by_sqi

    n = 100
    a = np.ones(n)
    b = np.ones(n) * 2.0
    fused, diag = fuse_signals_by_sqi({"a": a, "b": b}, {"a": 0.0, "b": 0.0}, fps=FPS)
    assert abs(diag["a"]["weight"] - 0.5) < 1e-9 and abs(diag["b"]["weight"] - 0.5) < 1e-9
    assert np.allclose(fused, 1.5)


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


def test_white_noise_is_not_publishable():
    """Регрессия: жёсткий SNR-гейт в assess_quality (см. quality.py).

    Раньше overall = 0.5*snr_score + 0.3*roi_score + 0.2*stability_score
    сравнивался с порогом 0.5 БЕЗ отдельной проверки SNR: на чистом шуме
    snr_score=0, а roi/stability при одном ROI/без landmark-точек нейтральны
    (0.5), что давало overall == 0.5 и publishable=True для сигнала без
    какой-либо пульсовой составляющей — прямая дыра в требовании ТЗ
    "не передавать BPM при низком качестве сигнала".

    Это же негативный контроль (в) из п.23 требований: белый шум -> SQI
    обязан сказать "не знаю", а не выдать правдоподобное число."""
    print("\n=== Негативный контроль (в): белый шум -> publishable == False ===")
    from rppg.signal.quality import assess_quality, dominant_frequency_and_snr, SQIInputs
    from rppg.config import QualityConfig

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, int(FPS * DURATION_S))
    peak_freq, snr_db = dominant_frequency_and_snr(noise, FPS, BAND)

    qcfg = QualityConfig()
    sqi = assess_quality(
        SQIInputs(
            spectral_snr_db=snr_db,
            peak_freq_hz=peak_freq,
            fps=FPS,
            band_hz=BAND,
            bpm_by_roi={"forehead": 75.0},  # один ROI -> нейтральный roi_score=0.5
            landmark_trajectories=None,  # нейтральный stability_score=0.5
        ),
        qcfg,
    )
    print(f"  spectral_snr_db={snr_db:.2f} dB, overall={sqi.overall_score:.3f}, "
          f"publishable={sqi.is_reliable}")
    assert not sqi.is_reliable, "Белый шум не должен быть publishable"


def test_landmark_stability_penalizes_strong_uniform_sway():
    """Регрессия для п.19: раньше джиттер нормировался на СОБСТВЕННУЮ медиану
    (относительная нормировка) -> человек, равномерно и сильно раскачивающийся
    ВЕСЬ интервал, получал стабильность 1.0, потому что весь его джиттер был
    "типичным" относительно самого себя. Нормировка на межзрачковое (IPD)
    расстояние обязана штрафовать такое раскачивание, а мелкий естественный
    джиттер — по-прежнему считать стабильным."""
    print("\n=== Тест: IPD-нормировка landmark_stability ловит сильное равномерное раскачивание ===")
    from rppg.signal.quality import landmark_stability_score

    rng = np.random.default_rng(0)
    n_frames, n_pts = 100, 10
    ipd = 80.0  # типичный масштаб лица в пикселях для веб-камеры
    t = np.arange(n_frames)

    strong_sway = np.zeros((n_frames, n_pts, 2))
    for p in range(n_pts):
        strong_sway[:, p, 0] = 100 + 12 * np.sin(0.5 * t) + rng.normal(0, 0.05, n_frames)
        strong_sway[:, p, 1] = 100 + rng.normal(0, 0.05, n_frames)

    small_jitter = np.zeros((n_frames, n_pts, 2))
    for p in range(n_pts):
        small_jitter[:, p, 0] = 100 + rng.normal(0, 0.3, n_frames)
        small_jitter[:, p, 1] = 100 + rng.normal(0, 0.3, n_frames)

    ipd_arr = np.full(n_frames, ipd)
    sway_score_ipd = landmark_stability_score(strong_sway, interocular_distances_px=ipd_arr)
    sway_score_selfnorm = landmark_stability_score(strong_sway, interocular_distances_px=None)
    small_score_ipd = landmark_stability_score(small_jitter, interocular_distances_px=ipd_arr)

    print(f"  сильное раскачивание, IPD-нормировка: {sway_score_ipd:.3f}")
    print(f"  сильное раскачивание, старая self-median нормировка: {sway_score_selfnorm:.3f}")
    print(f"  мелкий естественный джиттер, IPD-нормировка: {small_score_ipd:.3f}")

    assert sway_score_ipd < 0.7, "IPD-нормировка должна штрафовать сильное равномерное раскачивание"
    assert sway_score_selfnorm >= 0.99, (
        "sanity: self-median нормировка действительно 'слепа' к равномерному раскачиванию "
        "(это старое, ошибочное поведение, воспроизведённое намеренно для контраста)"
    )
    assert small_score_ipd >= 0.9, "мелкий естественный джиттер не должен штрафоваться IPD-нормировкой"


def test_temporal_consistency_penalizes_unstable_bpm_sequence():
    """Регрессия для п.20: temporal_consistency должна быть высокой для
    физиологически правдоподобной (плавно меняющейся) последовательности BPM
    между соседними окнами и низкой для скачущей последовательности —
    именно такую нестабильность cross_roi_agreement поймать не может, если
    все ROI ошибаются согласованно (см. docstring temporal_consistency_score)."""
    print("\n=== Тест: temporal_consistency штрафует скачущую последовательность BPM ===")
    from rppg.signal.quality import temporal_consistency_score

    stable = temporal_consistency_score([72.0, 73.0, 71.5, 72.5, 73.5, 72.0])
    unstable = temporal_consistency_score([72.0, 110.0, 65.0, 95.0, 60.0, 100.0])
    too_short = temporal_consistency_score([72.0, 73.0])

    print(f"  плавная последовательность: {stable:.3f}")
    print(f"  скачущая последовательность: {unstable:.3f}")
    print(f"  недостаточно истории (нейтрально): {too_short:.3f}")

    assert stable > 0.7, "плавная физиологичная последовательность BPM должна давать высокий score"
    assert unstable < 0.3, "резко скачущая последовательность BPM должна давать низкий score"
    assert too_short == 0.5, "при недостаточной истории score должен быть нейтральным (0.5)"


def test_harmonic_check_flags_fundamental_vs_second_harmonic_confusion():
    """Регрессия для п.21: если заявленный пик совпадает со ВТОРОЙ гармоникой
    реального тона (в сигнале сопоставимая энергия и на f, и на f/2), это
    классический failure mode rPPG — детектируется 2x или x0.5 от истинного
    пульса. Чистый одиночный тон не должен давать ложных срабатываний."""
    print("\n=== Тест: проверка на гармоники/субгармоники (п.21) ===")
    from rppg.signal.quality import harmonic_plausibility

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    rng = np.random.default_rng(1)

    clean_tone = np.sin(2 * np.pi * 1.0 * t) + 0.05 * rng.normal(0, 1, n)
    clean_score, clean_warnings = harmonic_plausibility(clean_tone, fps, peak_freq_hz=1.0)

    # Пик ошибочно "найден" на 2 Hz, но в сигнале сопоставимая энергия на
    # истинной частоте 1 Hz (= заявленный_пик / 2) -> подозрение на путаницу.
    ambiguous = np.sin(2 * np.pi * 1.0 * t) + 0.9 * np.sin(2 * np.pi * 2.0 * t)
    ambiguous_score, ambiguous_warnings = harmonic_plausibility(ambiguous, fps, peak_freq_hz=2.0)

    print(f"  чистый тон на f: score={clean_score:.2f}, warnings={clean_warnings}")
    print(f"  пик на 2f при сильной энергии на f: score={ambiguous_score:.2f}, warnings={ambiguous_warnings}")

    assert clean_score >= 0.9 and not clean_warnings, "чистый одиночный тон не должен давать ложных срабатываний"
    assert ambiguous_score < 0.7 and ambiguous_warnings, (
        "сопоставимая энергия на f/2 относительно заявленного пика на 2f должна флагироваться"
    )


def test_illumination_flicker_detected_via_background_roi():
    """Регрессия для п.22: если фоновый ROI (вне лица, например стена)
    САМ ПО СЕБЕ показывает узкий стабильный пик на частоте, совпадающей с
    "пульсом" на лице, это почти наверняка мерцание освещения, а не
    сердцебиение стены — жёсткий гейт публикации. Фон, который просто шумит
    без узкополосного пика, не должен ложно срабатывать."""
    print("\n=== Тест: детектор мерцания освещения через фоновый ROI (п.22) ===")
    from rppg.signal.quality import detect_illumination_flicker

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    face_peak_hz = 1.2  # 72 BPM

    flicker_bg = 2.0 * np.sin(2 * np.pi * face_peak_hz * t) + 0.05 * np.random.default_rng(2).normal(0, 1, n)
    flicker_suspected, warning = detect_illumination_flicker(flicker_bg, fps, face_peak_hz, band)

    clean_bg = np.random.default_rng(3).normal(0, 1, n)
    clean_suspected, clean_warning = detect_illumination_flicker(clean_bg, fps, face_peak_hz, band)

    print(f"  фон совпадает с частотой лица: flicker_suspected={flicker_suspected}, warning={bool(warning)}")
    print(f"  фон — просто шум: flicker_suspected={clean_suspected}, warning={bool(clean_warning)}")

    assert flicker_suspected and warning, "совпадающий узкополосный пик фона должен флагироваться как мерцание"
    assert not clean_suspected and clean_warning is None, "шумный фон без узкополосного пика не должен флагироваться"


def test_static_photo_or_mannequin_is_not_publishable():
    """Негативные контроли (а)/(б) из п.23 требований: статичное фото лица
    и видео манекена/распечатки — в обоих случаях ROI-сигнал не несёт
    физиологической пульсации: (а) буквально константа кадр к кадру,
    (б) константа + немодулированный сенсорный шум камеры (без какой-либо
    периодичности). Прогоняем через ПОЛНЫЙ сигнальный конвейер (извлечение
    -> препроцессинг -> SQI), а не напрямую через assess_quality, чтобы
    проверить оркестрацию, а не только формулу гейта."""
    print("\n=== Негативные контроли (а,б): статичное фото / манекен -> publishable == False ===")
    from rppg.signal.quality import assess_quality, dominant_frequency_and_snr, SQIInputs
    from rppg.config import QualityConfig

    n = int(FPS * DURATION_S)
    dc = {"R": 150.0, "G": 110.0, "B": 90.0}
    qcfg = QualityConfig()

    cases = {
        # (а) идеально статичное фото — без какого-либо шума камеры.
        "статичное фото": np.stack([np.full(n, dc[ch]) for ch in ("R", "G", "B")], axis=1),
        # (б) манекен/распечатка под реальным светом камеры — есть немодулированный
        # сенсорный шум, но нет ни одной периодической составляющей в HR-полосе.
        "манекен/распечатка": np.stack(
            [dc[ch] + np.random.default_rng(5).normal(0, 0.5, n) for ch in ("R", "G", "B")], axis=1
        ),
    }

    for label, rgb in cases.items():
        window = SignalWindow(rgb_traces={"forehead": rgb}, fps=FPS, hr_band_hz=BAND)
        method = get_method("pos")
        raw = method.extract(window, "forehead")
        processed = preprocess_signal(raw, FPS, *BAND, detrend_method="tarvainen", normalize_method="zscore")
        peak_freq, snr_db = dominant_frequency_and_snr(processed, FPS, BAND)

        sqi = assess_quality(
            SQIInputs(
                spectral_snr_db=snr_db,
                peak_freq_hz=peak_freq,
                fps=FPS,
                band_hz=BAND,
                bpm_by_roi={"forehead": 75.0},
            ),
            qcfg,
        )
        print(f"  [{label}] spectral_snr_db={snr_db:.2f} dB, overall={sqi.overall_score:.3f}, "
              f"publishable={sqi.is_reliable}")
        assert not sqi.is_reliable, f"{label}: не должно быть publishable (нет пульсации в сигнале)"


def test_occluded_face_pipeline_never_publishes():
    """Негативный контроль (г) из п.23 требований: видео с ПОЛНОСТЬЮ
    закрытым/отсутствующим лицом (камера смотрит в стену, лицо вне кадра)
    ни на одном окне не должно дать publishable=True — SQI обязан
    последовательно говорить "не знаю", а не подставлять число из шума
    интерполяции.

    В отличие от остальных тестов файла, это интеграционный тест ВСЕГО
    RPPGPipeline (реальный MediaPipe Face Landmarker + модель из models/),
    а не только сигнального уровня — пропускается, если модель/MediaPipe
    недоступны в текущей среде, вместо падения всего прогона."""
    print("\n=== Негативный контроль (г): лицо отсутствует в кадре -> publishable никогда не True ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    cfg = PipelineConfig()
    try:
        pipe = RPPGPipeline(cfg)
    except FileNotFoundError as exc:
        print(f"  [SKIP] модель Face Landmarker недоступна ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
        return

    try:
        blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)  # чёрный кадр -> лицо никогда не найдено
        got_any_result = False
        for i in range(200):  # ~6.7с при 30fps -> покрывает минимум одно BPM-окно
            timestamp_ms = int(i * 1000 / 30)
            result = pipe.process_frame(blank_frame, timestamp_ms)
            if result is None:
                continue
            got_any_result = True
            assert not result.publishable, (
                f"кадр без лица в кадре дал publishable=True (bpm={result.bpm}, "
                f"sqi={result.sqi_score:.3f})"
            )
    finally:
        pipe.close()

    assert got_any_result, (
        "sanity: пайплайн должен вернуть хотя бы один PTSDPulseFeatures "
        "(пусть и с publishable=False) за 200 кадров окна >= min_seconds_before_estimate"
    )
    print("  [OK] за 200 кадров без лица в кадре publishable не стал True ни разу")


def _make_synthetic_landmarks_for_pipeline_test(seed: int = 0, n_points: int = 478) -> np.ndarray:
    """478 точек в грубом эллипсе (нормализованные [0,1] координаты) —
    ТОЛЬКО чтобы ROI-полигоны строились на невырожденных наборах точек;
    реальная геометрия лица тут не нужна (см. докстринг
    test_end_to_end_pipeline_recovers_known_bpm_from_synthetic_video)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    radius = np.sqrt(rng.uniform(0, 1, n_points))
    x = 0.5 + 0.18 * radius * np.cos(theta)
    y = 0.5 + 0.25 * radius * np.sin(theta)
    z = rng.normal(0, 0.01, n_points)
    return np.stack([x, y, z], axis=1)


def _make_pulsating_frame(
    t_sec: float, true_bpm: float, rng: np.random.Generator, h: int = 240, w: int = 320
) -> np.ndarray:
    """Кадр с оттенком кожи (BGR=(40,60,100) — подобран так, чтобы попадать
    в диапазон face.roi.build_skin_mask), пульсирующим по цвету с известным
    true_bpm — ТОЛЬКО внутри эллипса, соответствующего разбросу точек
    _make_synthetic_landmarks_for_pipeline_test. Фон СТАТИЧЕН и НЕ
    пульсирует: если бы пульсировал весь кадр целиком, фоновый ROI (см.
    face/roi.py::build_background_roi_mask, п.22) показывал бы ту же
    частоту, что и лицо, и detect_illumination_flicker справедливо
    заблокировал бы публикацию — тест тогда проверял бы не то, что нужно."""
    frame = np.full((h, w, 3), (90, 70, 60), dtype=np.uint8)
    f_hz = true_bpm / 60.0
    pulse = np.sin(2 * np.pi * f_hz * t_sec)
    base_bgr = np.array([40.0, 60.0, 100.0])
    channel_amp = np.array([9.0, 15.0, 5.0])  # относительная чувствительность B/G/R как в make_synthetic_rgb
    color = base_bgr + channel_amp * pulse + rng.normal(0, 1.5, 3)
    color = np.clip(color, 0, 255).astype(np.uint8)

    cy, cx = h // 2, w // 2
    ry, rx = int(0.25 * h), int(0.18 * w)
    yy, xx = np.ogrid[:h, :w]
    ellipse_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    frame[ellipse_mask] = color
    return frame


def test_end_to_end_pipeline_recovers_known_bpm_from_synthetic_video():
    """Регрессия для п.39 требований: сквозной тест через
    RPPGPipeline.process_frame ЦЕЛИКОМ (буферизация окна по реальному
    времени -> ROI-экстракция со skin-маской -> метод извлечения сигнала ->
    оценка частоты -> SQI-гейтинг -> IBI-накопитель), а не только
    сигнальный уровень. Такой тест ловит баги ОРКЕСТРАЦИИ (неверная
    передача landmark_trajectories между стадиями, окно, считающееся по
    числу кадров вместо реального времени переменного fps, и т.п.),
    которые остальные тесты этого файла в принципе не видят — они вызывают
    сигнальные функции (extract/preprocess_signal/estimate_hr) напрямую, в
    обход самого RPPGPipeline.

    См. ОГОВОРКУ в докстринге _make_pulsating_frame про MediaPipe: реальное
    фото лица в этой среде недоступно (и не должно скачиваться из
    неофициальных источников без явного согласия, см. обсуждение доступа к
    rPPG-датасетам), поэтому детекция подменяется на фиксированный
    синтетический результат — для проверки ОРКЕСТРАЦИИ (а не точности
    детекции лица) этого достаточно: важно, что downstream-код
    (roi/method/frequency/sqi/ibi) реально исполняется на кадрах с
    ИЗВЕСТНЫМ пульсирующим цветом "кожи"."""
    print("\n=== Сквозной тест: RPPGPipeline.process_frame на синтетическом видео с известным BPM (п.39) ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
        from rppg.face.landmarker import FaceFrameResult, HeadPose
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    cfg = PipelineConfig()
    try:
        pipe = RPPGPipeline(cfg)
    except FileNotFoundError as exc:
        print(f"  [SKIP] модель Face Landmarker недоступна ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
        return

    true_bpm = 78.0
    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )
    pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result  # monkeypatch только для теста

    rng = np.random.default_rng(42)
    fps = 30.0
    n_frames = int(fps * 20.0)  # 20с синтетического видео

    results = []
    try:
        for i in range(n_frames):
            t_sec = i / fps
            frame = _make_pulsating_frame(t_sec, true_bpm, rng)
            result = pipe.process_frame(frame, int(t_sec * 1000))
            if result is not None:
                results.append(result)
    finally:
        pipe.close()

    print(f"  получено {len(results)} оценок за {n_frames} кадров")
    assert len(results) > 0, "пайплайн должен вернуть хотя бы одну оценку за 20с синтетического видео"

    publishable = [r for r in results if r.publishable]
    print(f"  publishable: {len(publishable)}/{len(results)}")
    for r in publishable[-3:]:
        per_roi = {k: round(v, 1) for k, v in r.per_roi_bpm.items()}
        print(f"    t={r.timestamp_ms}ms bpm={r.bpm:.2f} sqi={r.sqi_score:.3f} per_roi_bpm={per_roi}")
    assert publishable, "на чистом синтетическом пульсирующем видео хотя бы одна оценка должна быть publishable"

    last = publishable[-1]
    err = abs(last.bpm - true_bpm)
    print(f"  последняя publishable оценка: bpm={last.bpm:.2f} (истинный={true_bpm}), ошибка={err:.2f} BPM")
    assert err < 5.0, f"сквозная оценка BPM должна быть близка к истинной (получено {err:.2f} BPM ошибки)"


def test_auto_method_selection_avoids_specifically_corrupted_method():
    """Регрессия для запроса пользователя "адаптировать код под разные
    камеры (лучше и хуже)": ExtractionMethod.AUTO должен КАЖДОЕ окно
    заново оценивать все 5 цветовых методов по SNR и не залипать на одном
    методе, если он плохо подходит под текущие условия.

    Конструируем сценарий, где ИМЕННО зелёный канал сильно зашумлён
    (частая ситуация на некоторых камерах/условиях освещения — разные
    сенсоры по-разному распределяют шум/усиление по каналам), а остальные
    методы (комбинирующие все 3 канала — CHROM/POS/PCA/ICA) остаются
    рабочими. GREEN-метод использует ИСКЛЮЧИТЕЛЬНО зелёный канал (см.
    methods.GreenMethod.extract), поэтому AUTO обязан НИКОГДА не выбрать
    его в этом сценарии — иначе адаптация к камере не работает."""
    print("\n=== Тест: AUTO-выбор метода избегает специально испорченного канала ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig, ExtractionMethod
        from rppg.face.landmarker import FaceFrameResult, HeadPose
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    cfg = PipelineConfig(method=ExtractionMethod.AUTO)
    try:
        pipe = RPPGPipeline(cfg)
    except FileNotFoundError as exc:
        print(f"  [SKIP] модель Face Landmarker недоступна ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
        return

    def make_frame_green_corrupted(t_sec, true_bpm, rng, h=240, w=320):
        frame = np.full((h, w, 3), (90, 70, 60), dtype=np.uint8)
        f_hz = true_bpm / 60.0
        pulse = np.sin(2 * np.pi * f_hz * t_sec)
        base_bgr = np.array([40.0, 60.0, 100.0])
        channel_amp = np.array([9.0, 15.0, 5.0])
        noise = rng.normal(0, 1.5, 3)
        noise[1] += rng.normal(0, 40.0)  # BGR-индекс 1 = зелёный -> мощный шум ТОЛЬКО там
        color = np.clip(base_bgr + channel_amp * pulse + noise, 0, 255).astype(np.uint8)
        cy, cx = h // 2, w // 2
        ry, rx = int(0.25 * h), int(0.18 * w)
        yy, xx = np.ogrid[:h, :w]
        frame[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = color
        return frame

    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )
    pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result

    rng = np.random.default_rng(7)
    fps = 30.0
    n_frames = int(fps * 20.0)
    results = []
    try:
        for i in range(n_frames):
            t_sec = i / fps
            frame = make_frame_green_corrupted(t_sec, 78.0, rng)
            result = pipe.process_frame(frame, int(t_sec * 1000))
            if result is not None:
                results.append(result)
    finally:
        pipe.close()

    assert results, "должна быть хотя бы одна оценка за 20с видео"
    methods_used = {r.method_used for r in results}
    print(f"  методы, выбранные AUTO по ходу сессии: {methods_used}")
    assert all(m.startswith("auto:") for m in methods_used), "method_used должен быть помечен как auto:<метод>"
    assert not any("green" in m for m in methods_used), (
        "AUTO не должен выбирать метод, использующий исключительно испорченный зелёный канал"
    )


def test_best_effort_bpm_always_present_bounded_and_rate_limited():
    """Регрессия для интеграционного требования: этот модуль встраивается
    как одна из фич более крупной системы, которой на каждом шаге нужно
    ЗНАЧЕНИЕ (не отсутствие данных) — PTSDPulseFeatures.best_effort_bpm
    должен ВСЕГДА быть реальным числом в правдоподобном диапазоне
    (BestEffortConfig) и меняться постепенно (slew-rate limiting), а не
    прыгать как 'сырой' bpm до SQI-гейтинга."""
    print("\n=== Тест: best_effort_bpm всегда доступен, в диапазоне, ограничен по скорости ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    cfg = PipelineConfig()
    try:
        pipe = RPPGPipeline(cfg)
    except FileNotFoundError as exc:
        print(f"  [SKIP] модель Face Landmarker недоступна ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
        return

    try:
        be_cfg = cfg.best_effort
        assert pipe._best_effort_bpm == be_cfg.fallback_bpm, "до первого шага должен быть fallback_bpm"

        v0 = pipe._update_best_effort_bpm(float("nan"))
        assert v0 == be_cfg.fallback_bpm, "NaN-кандидат (нет сигнала) должен держать прежнее значение"

        v1 = pipe._update_best_effort_bpm(150.0)  # типичный шумовой выброс из живой отладки
        assert be_cfg.min_plausible_bpm <= v1 <= be_cfg.max_plausible_bpm
        assert v1 == v0, "кандидат ВЫШЕ диапазона должен игнорироваться целиком, не сдвигать значение вообще"

        v2 = pipe._update_best_effort_bpm(20.0)
        assert v2 == v1, "кандидат НИЖЕ диапазона тоже должен игнорироваться целиком"

        target = be_cfg.fallback_bpm + 20.0
        assert be_cfg.min_plausible_bpm <= target <= be_cfg.max_plausible_bpm
        v3 = pipe._update_best_effort_bpm(target)
        change = abs(v3 - v2)
        print(f"  один шаг к правдоподобной цели {target}: {v2:.1f} -> {v3:.1f} (Δ={change:.1f})")
        assert 0 < change <= be_cfg.max_change_per_step_bpm + 1e-9, (
            "правдоподобный кандидат должен сдвигать значение, но не больше max_change_per_step_bpm за шаг"
        )

        v_final = v3
        for _ in range(50):
            v_final = pipe._update_best_effort_bpm(target)
        assert abs(v_final - target) < 1e-6, "после достаточного числа шагов значение должно сойтись к цели"
        assert not np.isnan(v_final), "best_effort_bpm не должен становиться NaN ни на одном шаге"
    finally:
        pipe.close()


def test_best_effort_bpm_stays_smooth_under_50_150_noise_pattern():
    """Регрессия для реального инцидента с камерой пользователя: 'сырой'
    per-window bpm скакал между ~50 и ~150+ из-за расхождения ROI на
    слабом сигнале (см. живую отладочную сессию). best_effort_bpm должен
    оставаться плавным под ТОЧНО ТАКИМ ЖЕ паттерном входа — иначе шум
    всё равно доходит до принимающей системы, просто под другим именем поля."""
    print("\n=== Тест: best_effort_bpm гасит паттерн 50<->150 из реальной сессии ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    cfg = PipelineConfig()
    try:
        pipe = RPPGPipeline(cfg)
    except FileNotFoundError as exc:
        print(f"  [SKIP] модель Face Landmarker недоступна ({exc})")
        return
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
        return

    try:
        # Реальный наблюдавшийся паттерн per_roi_bpm (forehead/left_cheek/right_cheek
        # чередующиеся победители в best_roi) из живой отладочной сессии.
        raw_sequence = [101.0, 49.8, 74.8, 100.6, 50.7, 77.0, 66.0, 79.1, 176.7, 78.7, 126.4, 178.0, 50.3]
        outputs = [pipe._update_best_effort_bpm(v) for v in raw_sequence]
        print("  сырые входы:     ", [round(v, 1) for v in raw_sequence])
        print("  best_effort_bpm: ", [round(v, 1) for v in outputs])

        be_cfg = cfg.best_effort
        assert all(be_cfg.min_plausible_bpm <= v <= be_cfg.max_plausible_bpm for v in outputs), (
            "best_effort_bpm не должен покидать правдоподобный диапазон ни разу"
        )
        step_changes = [abs(b - a) for a, b in zip(outputs, outputs[1:])]
        print(f"  максимальное изменение за шаг: {max(step_changes):.2f} (лимит {be_cfg.max_change_per_step_bpm})")
        assert max(step_changes) <= be_cfg.max_change_per_step_bpm + 1e-9, (
            "ни один шаг не должен превышать max_change_per_step_bpm"
        )
        assert not any(np.isnan(v) for v in outputs)
    finally:
        pipe.close()


def test_config_yaml_json_roundtrip_preserves_non_default_values():
    """Регрессия для п.42 требований: save_config/load_config должны точно
    восстанавливать PipelineConfig, включая НЕстандартные значения (Enum-
    поля method/frequency_method, tuple Enum-полей roi.enabled_rois,
    вложенный FusionConfig.enabled) — иначе "конфиг эксперимента,
    сохранённый рядом с результатами" (п.42) не гарантирует, что при
    повторном запуске это будет ТОТ ЖЕ конфиг."""
    print("\n=== Тест: YAML/JSON roundtrip PipelineConfig сохраняет нестандартные значения (п.42) ===")
    import tempfile
    from pathlib import Path
    from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod, ROIName
    from rppg.config_io import save_config, load_config, config_from_dict

    cfg = PipelineConfig(method=ExtractionMethod.CHROM, frequency_method=FrequencyMethod.LOMB_SCARGLE)
    cfg.fusion.enabled = True
    cfg.roi.enabled_rois = (ROIName.FOREHEAD, ROIName.LEFT_CHEEK)
    cfg.filt.tarvainen_lambda = 1234.5

    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in (".yaml", ".json"):
            path = Path(tmpdir) / f"config{ext}"
            save_config(cfg, path)
            loaded = load_config(path)
            print(f"  {ext}: loaded == original: {loaded == cfg}")
            assert loaded == cfg, f"{ext}: roundtrip должен точно восстановить конфиг"
            assert loaded.roi.enabled_rois == (ROIName.FOREHEAD, ROIName.LEFT_CHEEK)
            assert isinstance(loaded.method, ExtractionMethod) and loaded.method == ExtractionMethod.CHROM

    partial = config_from_dict({"method": "pos", "quality": {"min_overall_score_to_publish": 0.7}})
    print(f"  частичный конфиг: method={partial.method.value}, "
          f"min_score={partial.quality.min_overall_score_to_publish}, "
          f"остальные поля quality не тронуты (min_spectral_snr_db={partial.quality.min_spectral_snr_db})")
    assert partial.method == ExtractionMethod.POS
    assert partial.quality.min_overall_score_to_publish == 0.7
    assert partial.quality.min_spectral_snr_db == PipelineConfig().quality.min_spectral_snr_db, (
        "частичный конфиг не должен трогать НЕуказанные поля секции"
    )


def test_structured_window_logger_writes_valid_jsonl_with_sqi_components():
    """Регрессия для п.43 требований: "Каждое окно -> строка с timestamp,
    BPM по каждому ROI, компоненты SQI, warnings" — прогоняем ТОТ ЖЕ
    сквозной сценарий, что и test_end_to_end_pipeline_recovers_known_bpm_
    from_synthetic_video, но с log_path, и проверяем, что получившийся
    JSONL реально содержит ВСЕ обещанные поля, а не только сводный
    sqi_score/sqi_level."""
    print("\n=== Тест: структурированный JSONL-лог по окнам (п.43) ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
        from rppg.face.landmarker import FaceFrameResult, HeadPose
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "windows.jsonl"
        cfg = PipelineConfig()
        try:
            pipe = RPPGPipeline(cfg, log_path=log_path)
        except FileNotFoundError as exc:
            print(f"  [SKIP] модель Face Landmarker недоступна ({exc})")
            return
        except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
            print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
            return

        true_bpm = 75.0
        landmarks = _make_synthetic_landmarks_for_pipeline_test()
        synthetic_result = FaceFrameResult(
            detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
        )
        pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result

        rng = np.random.default_rng(11)
        fps = 30.0
        n_frames = int(fps * 8.0)
        try:
            for i in range(n_frames):
                t_sec = i / fps
                frame = _make_pulsating_frame(t_sec, true_bpm, rng)
                pipe.process_frame(frame, int(t_sec * 1000))
        finally:
            pipe.close()

        # ВАЖНО: всё, что читает log_path, должно оставаться ВНУТРИ этого
        # `with tempfile.TemporaryDirectory()` — сама директория удаляется
        # при выходе из блока.
        assert log_path.exists(), "log_path должен быть создан, раз пайплайн выдал хотя бы одно окно"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        print(f"  строк в логе: {len(lines)}")
        assert len(lines) > 0, "должна быть хотя бы одна строка лога за 8с синтетического видео"

        required_fields = {
            "timestamp_ms", "bpm", "per_roi_bpm", "publishable", "method_used",
            "frequency_method_used", "sqi_overall_score", "sqi_level",
            "sqi_spectral_snr_db", "sqi_cross_roi_agreement", "sqi_landmark_stability",
            "sqi_temporal_consistency", "sqi_harmonic_score", "sqi_flicker_suspected",
            "respiration_rate_bpm", "warnings",
        }
        for line in lines:
            record = json.loads(line)  # sanity: каждая строка — валидный JSON
            missing = required_fields - set(record.keys())
            assert not missing, f"строка лога не содержит поля {missing}: {record}"
            assert isinstance(record["per_roi_bpm"], dict)
            assert isinstance(record["warnings"], list)

        last = json.loads(lines[-1])
        print(f"  последняя запись: bpm={last['bpm']:.2f}, sqi_overall={last['sqi_overall_score']:.3f}, "
              f"sqi_landmark_stability={last['sqi_landmark_stability']:.3f}, "
              f"sqi_temporal_consistency={last['sqi_temporal_consistency']:.3f}")


def test_subsample_peak_refinement_reduces_rmssd_quantization_error():
    """Регрессия для п.13 (Этап C): на 30 fps шаг индекса пика = 33.3 мс,
    сопоставимый с самим RMSSD (20-50 мс у здоровых в покое). Без
    суб-сэмпловой параболической интерполяции RMSSD в основном измеряет
    квантование кадров, а не физиологию — проверяем на сигнале с известным
    (очень малым) истинным джиттером IBI, что интерполяция реально снижает
    эту ошибку, а не просто существует в коде."""
    print("\n=== Тест: суб-сэмпловая интерполяция пиков снижает шум квантования RMSSD ===")
    fps = 30.0
    rng = np.random.default_rng(7)
    true_ibi_ms = 800.0
    jitter_ms = rng.normal(0, 3.0, 400)  # истинный джиттер ~3мс (намеренно << шага 33.3мс)
    true_ibis = true_ibi_ms + jitter_ms
    beat_times_s = np.cumsum(true_ibis) / 1000.0
    n = int((beat_times_s[-1] + 1) * fps)
    signal = np.zeros(n)
    for bt in beat_times_s:
        center_idx = bt * fps
        idxs = np.arange(max(0, int(center_idx) - 5), min(n, int(center_idx) + 6))
        signal[idxs] += np.exp(-0.5 * ((idxs - center_idx) / 1.2) ** 2)

    true_rmssd = np.sqrt(np.mean(np.diff(true_ibis) ** 2))

    peaks = detect_pulse_peaks(signal, fps)
    ibi_int = compute_ibi_ms(peaks.astype(float), fps)
    rmssd_int = np.sqrt(np.mean(np.diff(ibi_int) ** 2))

    peaks_sub = refine_peaks_subsample(signal, peaks)
    ibi_sub = compute_ibi_ms(peaks_sub, fps)
    rmssd_sub = np.sqrt(np.mean(np.diff(ibi_sub) ** 2))

    err_int = abs(rmssd_int - true_rmssd)
    err_sub = abs(rmssd_sub - true_rmssd)
    print(f"  истинный RMSSD={true_rmssd:.2f}мс, целые индексы={rmssd_int:.2f}мс (ошибка {err_int:.2f}), "
          f"суб-сэмпл={rmssd_sub:.2f}мс (ошибка {err_sub:.2f})")
    assert err_sub < err_int, "суб-сэмпловая интерполяция должна снижать ошибку RMSSD от квантования кадров"
    assert err_sub < 2.0, "суб-сэмпловая оценка должна быть близка к истинному RMSSD"


def test_ectopic_masking_does_not_stitch_false_diff():
    """Регрессия для п.16: старое поведение (удаление выброса из ряда)
    "сшивает" соседние валидные интервалы через дыру и даёт ложную разность
    в np.diff. Правильно — маскировать и исключать из diff() любую пару,
    где хотя бы один сосед невалиден."""
    print("\n=== Тест: маскирование эктопических IBI не создаёт ложный 'сшитый' diff ===")
    ibi = np.full(20, 800.0)
    ibi[9] = 850.0
    ibi[10] = 2000.0  # явный артефакт
    ibi[11] = 750.0
    mask = ectopic_artifact_mask(ibi, max_relative_change=0.4)
    assert not mask[10], "явный выброс должен быть замаскирован"

    # То, что дал бы старый подход "удалить и сшить": прыжок idx9(850) -> idx12(800)
    # напрямую, полностью пропуская замаскированные idx10/idx11 — фиктивная разность -50,
    # которой в реальности между двумя РЕАЛЬНО соседними ударами не было.
    deleted = ibi[mask]
    stitched_diffs = np.diff(deleted)
    assert -50.0 in stitched_diffs, "sanity: удаление действительно фабрикует эту фиктивную разность"

    # То, что реально использует hrv_time_domain (маскирование): пары, где хотя бы
    # один сосед невалиден, из diff() исключаются целиком, а не сшиваются.
    pair_valid = mask[:-1] & mask[1:]
    used_diffs = np.diff(ibi)[pair_valid]
    assert -50.0 not in used_diffs, (
        "маскирование не должно давать ту же фиктивную разность, что и удаление со сшиванием"
    )
    print("  [OK] маскирование исключает 'сшитый' diff, который производит удаление")


def test_hrv_artifact_fraction_gates_publishable():
    """Регрессия для п.17: если доля отбракованных IBI превышает порог
    (по умолчанию 5%), HRV за это окно не должен публиковаться — тот же
    принцип flag-based гейта, что и PTSDPulseFeatures.publishable для BPM."""
    print("\n=== Тест: доля артефактов > порога -> HRVFeatures.publishable == False ===")
    rng = np.random.default_rng(11)
    clean_ibi = rng.normal(800, 20, 200)
    hrv_clean = extract_hrv_features_from_ibi(clean_ibi)
    assert hrv_clean.publishable, "чистый ряд IBI должен быть publishable"

    dirty_ibi = clean_ibi.copy()
    for i in range(10, 200, 13):  # ~15% изолированных выбросов
        dirty_ibi[i] *= 2.5
    hrv_dirty = extract_hrv_features_from_ibi(dirty_ibi)
    print(f"  чистый: artifact_fraction={hrv_clean.artifact_fraction:.3f} publishable={hrv_clean.publishable}")
    print(f"  зашумлённый: artifact_fraction={hrv_dirty.artifact_fraction:.3f} publishable={hrv_dirty.publishable}")
    assert hrv_dirty.artifact_fraction > 0.05
    assert not hrv_dirty.publishable, "HRV с >5% отбракованных интервалов не должен быть publishable"


def test_lf_hf_require_separate_minimum_durations():
    """Регрессия для п.15: старый общий порог duration<20с был мягче ЛЮБОГО
    периода LF-полосы (6.7-25с) и на 10-секундном окне обнулял LF/HF ещё до
    проверки. Нужны раздельные пороги: HF >= 60с, LF >= 120с."""
    print("\n=== Тест: LF требует >=120с, HF требует >=60с ===")
    rng = np.random.default_rng(1)

    def make_ibi(duration_s):
        n = int(duration_s * 1000 / 800.0) + 5
        return rng.normal(800.0, 40.0, n)

    fd_short = hrv_frequency_domain(make_ibi(10))
    assert fd_short["lf_power"] is None and fd_short["hf_power"] is None

    fd_mid = hrv_frequency_domain(make_ibi(90))
    assert fd_mid["hf_power"] is not None and fd_mid["lf_power"] is None, "90с: только HF, ещё не LF"

    fd_long = hrv_frequency_domain(make_ibi(150))
    assert fd_long["lf_power"] is not None and fd_long["hf_power"] is not None, "150с: и LF, и HF"
    print("  [OK] LF/HF появляются на разных минимальных длительностях, а не одновременно на duration<20")


def test_respiration_rate_from_amplitude_modulation():
    """Регрессия для п.18: частота дыхания оценивается по огибающей
    (амплитудной модуляции) пульсовой волны в полосе 0.15-0.4 Гц."""
    print("\n=== Тест: оценка частоты дыхания по амплитудной модуляции ===")
    fps = 30.0
    duration = 40.0
    n = int(fps * duration)
    t = np.arange(n) / fps
    hr_hz = 75 / 60.0
    resp_hz_true = 0.25  # 15 дыханий/мин
    carrier = np.sin(2 * np.pi * hr_hz * t)
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * resp_hz_true * t)
    rng = np.random.default_rng(4)
    pulse_with_resp = envelope * carrier + rng.normal(0, 0.02, n)

    resp_bpm, resp_hz = estimate_respiration_rate(pulse_with_resp, fps)
    print(f"  истинная частота дыхания={resp_hz_true * 60:.1f} дых/мин, оценка={resp_bpm:.1f} дых/мин")
    assert resp_hz is not None and abs(resp_hz - resp_hz_true) < 0.03

    short_bpm, short_hz = estimate_respiration_rate(pulse_with_resp[:100], fps)
    assert short_bpm is None and short_hz is None, "слишком короткий сигнал должен безопасно вернуть None"


def run_all():
    tests = [
        test_color_based_methods_recover_known_bpm,
        test_pos_numba_matches_numpy_reference,
        test_head_motion_method_recovers_known_bpm,
        test_frequency_estimators_agree_on_clean_tone,
        test_lombscargle_handles_irregular_sampling,
        test_bandpass_rejects_out_of_band_and_keeps_in_band,
        test_detrend_removes_slow_trend,
        test_tarvainen_cutoff_matches_documented_reference_and_new_default_is_transparent_to_pulse_band,
        test_align_sign_and_lag_recovers_known_shift,
        test_fuse_signals_by_sqi_weighting_favors_clean_source,
        test_fuse_signals_by_sqi_aligns_out_of_phase_source_without_cancellation,
        test_fuse_signals_by_sqi_zero_weights_fallback_to_uniform,
        test_hrv_features_match_known_ibi_statistics,
        test_white_noise_is_not_publishable,
        test_landmark_stability_penalizes_strong_uniform_sway,
        test_temporal_consistency_penalizes_unstable_bpm_sequence,
        test_harmonic_check_flags_fundamental_vs_second_harmonic_confusion,
        test_illumination_flicker_detected_via_background_roi,
        test_static_photo_or_mannequin_is_not_publishable,
        test_occluded_face_pipeline_never_publishes,
        test_end_to_end_pipeline_recovers_known_bpm_from_synthetic_video,
        test_auto_method_selection_avoids_specifically_corrupted_method,
        test_best_effort_bpm_always_present_bounded_and_rate_limited,
        test_best_effort_bpm_stays_smooth_under_50_150_noise_pattern,
        test_config_yaml_json_roundtrip_preserves_non_default_values,
        test_structured_window_logger_writes_valid_jsonl_with_sqi_components,
        test_subsample_peak_refinement_reduces_rmssd_quantization_error,
        test_ectopic_masking_does_not_stitch_false_diff,
        test_hrv_artifact_fraction_gates_publishable,
        test_lf_hf_require_separate_minimum_durations,
        test_respiration_rate_from_amplitude_modulation,
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
