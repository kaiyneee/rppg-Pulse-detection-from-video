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


def test_white_noise_is_not_publishable():
    """Регрессия: жёсткий SNR-гейт в assess_quality (см. quality.py).

    Раньше overall = 0.5*snr_score + 0.3*roi_score + 0.2*stability_score
    сравнивался с порогом 0.5 БЕЗ отдельной проверки SNR: на чистом шуме
    snr_score=0, а roi/stability при одном ROI/без landmark-точек нейтральны
    (0.5), что давало overall == 0.5 и publishable=True для сигнала без
    какой-либо пульсовой составляющей — прямая дыра в требовании ТЗ
    "не передавать BPM при низком качестве сигнала"."""
    print("\n=== Тест: белый шум -> publishable == False ===")
    from rppg.signal.quality import assess_quality, dominant_frequency_and_snr
    from rppg.config import QualityConfig

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, int(FPS * DURATION_S))
    _, snr_db = dominant_frequency_and_snr(noise, FPS, BAND)

    qcfg = QualityConfig()
    sqi = assess_quality(
        spectral_snr_db=snr_db,
        bpm_by_roi={"forehead": 75.0},  # один ROI -> нейтральный roi_score=0.5
        landmark_trajectories=None,  # нейтральный stability_score=0.5
        min_spectral_snr_db=qcfg.min_spectral_snr_db,
        max_cross_roi_bpm_diff=qcfg.max_cross_roi_bpm_diff,
        min_landmark_stability=qcfg.min_landmark_stability,
        min_overall_score_to_publish=qcfg.min_overall_score_to_publish,
    )
    print(f"  spectral_snr_db={snr_db:.2f} dB, overall={sqi.overall_score:.3f}, "
          f"publishable={sqi.is_reliable}")
    assert not sqi.is_reliable, "Белый шум не должен быть publishable"


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
        test_head_motion_method_recovers_known_bpm,
        test_frequency_estimators_agree_on_clean_tone,
        test_lombscargle_handles_irregular_sampling,
        test_bandpass_rejects_out_of_band_and_keeps_in_band,
        test_detrend_removes_slow_trend,
        test_hrv_features_match_known_ibi_statistics,
        test_white_noise_is_not_publishable,
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
