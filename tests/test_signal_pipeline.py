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
    mean_hr_from_peaks_in_window,
    cross_check_spectral_vs_peaks,
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


def test_chrom_alpha_robust_to_linear_drift():
    """Верификация задачи 32: CHROM должен считать alpha (=std(Xf)/std(Yf))
    ПОСЛЕ полосовой фильтрации Xs/Ys, а не на сырых Xs/Ys — иначе
    низкочастотный дрейф освещения доминирует над std() и искажает alpha
    (де-факто это уже так реализовано в ChromMethod.extract, см. его
    docstring — здесь закрепляем регрессионным тестом, а не принимаем на
    слово). Реализация alpha воспроизводится напрямую из тех же
    строительных блоков (_temporal_normalize + bandpass_filter), что
    использует ChromMethod.extract, т.к. сам alpha не выставлен наружу в
    возвращаемом сигнале."""
    print("\n=== Тест: alpha CHROM устойчив к сильному линейному дрейфу освещения (задача 32) ===")
    from rppg.signal.methods import _temporal_normalize

    def chrom_alpha(rgb, fps, band):
        cn = _temporal_normalize(rgb)
        rn, gn, bn = cn[:, 0], cn[:, 1], cn[:, 2]
        xs = 3 * rn - 2 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        xf = bandpass_filter(xs, fps, *band)
        yf = bandpass_filter(ys, fps, *band)
        return np.std(xf) / np.std(yf)

    rgb_clean = make_synthetic_rgb(72.0, noise_std=0.0, seed=7, duration_s=20.0)
    n = len(rgb_clean)
    # Дрейф ~9% от типичного DC-уровня (110, канал G) за 20с — заметно
    # сильнее пульсовой модуляции (~1% DC), но не абсурдно доминирующий.
    drift = np.linspace(0.0, 10.0, n)[:, None]
    rgb_drifted = rgb_clean + drift

    alpha_clean = chrom_alpha(rgb_clean, FPS, BAND)
    alpha_drifted = chrom_alpha(rgb_drifted, FPS, BAND)
    rel_diff = abs(alpha_drifted - alpha_clean) / abs(alpha_clean)

    print(f"  alpha (без дрейфа) = {alpha_clean:.4f}, alpha (с дрейфом) = {alpha_drifted:.4f}, "
          f"относительная разница = {rel_diff:.1%}")
    assert rel_diff < 0.05, f"сильный линейный дрейф исказил alpha CHROM больше чем на 5%: {rel_diff:.1%}"


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

    for fm, tol in [("fft", 2.0), ("welch", 2.5), ("lomb_scargle", 2.5), ("harmonic_sum", 2.5)]:
        est = estimate_hr(processed, FPS, BAND, method=fm)
        _assert_bpm_close(est.bpm, true_bpm, tol, f"POS + {fm}")


def test_harmonic_sum_avoids_octave_error_that_confuses_plain_argmax():
    """Регрессия: harmonic_sum должен правильно найти истинный f0=72 BPM
    даже когда ОДИНОЧНАЯ точка спектра на 2f0=144 BPM мощнее самого f0
    (реалистичный случай: артефакт движения/освещения на кратной частоте) —
    тогда как обычный argmax по PSD (welch) в ЭТОЙ ЖЕ ситуации ошибочно
    берёт 2f0. Harmonic sum побеждает, т.к. у ложного пика на 2f0 НЕТ
    согласованной энергии на СВОИХ гармониках (4f0, 6f0 — просто шум), а у
    истинного f0 есть реальная физиологическая энергия сразу на 2f0 И 3f0."""
    print("\n=== Тест: harmonic_sum устойчив к точечному выбросу на 2f, где argmax ошибается ===")
    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    f0 = 1.2  # 72 BPM

    rng = np.random.default_rng(2)
    signal = (
        0.6 * np.sin(2 * np.pi * f0 * t)
        + 1.3 * np.sin(2 * np.pi * 2 * f0 * t)  # артефакт/2-я гармоника, точечно мощнее f0
        + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)  # 3-я гармоника — только у истинного f0
        + 0.05 * rng.normal(0, 1, n)
    )

    welch_est = estimate_hr(signal, fps, band, method="welch")
    harmonic_est = estimate_hr(signal, fps, band, method="harmonic_sum")

    print(f"  welch: {welch_est.bpm:.1f} BPM, harmonic_sum: {harmonic_est.bpm:.1f} BPM "
          f"(истина: {f0 * 60:.0f} BPM)")

    assert abs(welch_est.bpm - 2 * f0 * 60) < 3.0, (
        "sanity: обычный welch-argmax должен ОШИБАТЬСЯ на этом сигнале и брать 2f0 — "
        "иначе сценарий не демонстрирует разницу методов"
    )
    assert abs(harmonic_est.bpm - f0 * 60) < 3.0, (
        f"harmonic_sum должен найти истинный f0={f0 * 60:.0f} BPM, получено {harmonic_est.bpm:.1f}"
    )


def test_harmonic_sum_not_biased_by_unavailable_high_harmonics():
    """Регрессия задачи 35: раньше недоступная (за пределами Найквиста)
    гармоника подставлялась как 0.0 в СУММУ (не среднее), а log_psd почти
    всегда отрицателен — значит 0.0 действовал как БОНУС, а не нейтральный
    пропуск, систематически смещая выбор к кандидатам с МЕНЬШИМ числом
    доступных гармоник. Здесь у истинного f0=1.0 Гц при fps=5 (nyquist=2.5)
    доступны только k=1,2 (навсегда, а не k=3), а его же СОБСТВЕННАЯ 2-я
    гармоника (2.0 Гц) как САМОСТОЯТЕЛЬНЫЙ кандидат имеет ЕЩЁ меньше
    доступных гармоник (только k=1) — раньше это делало её сумму (без
    усреднения) сильнее суммы истинного f0, и алгоритм ошибочно запирался
    на октаву выше (2f0) именно при fps=5, хотя при fps=30 (все 3 гармоники
    доступны у f0) корректно находил f0. После фикса (среднее по числу
    доступных гармоник) обе частоты дискретизации дают ОДИН И ТОТ ЖЕ
    результат — прямое требование задачи 35."""
    print("\n=== Тест: harmonic_sum не смещается октавой из-за недоступных гармоник (задача 35) ===")
    f0 = 1.0  # 60 BPM
    band = (0.5, 2.4)  # верх полосы < nyquist(fps=5)=2.5, иначе полоса обрежется целиком

    fps_high = 30.0
    n_high = int(fps_high * 20)
    t_high = np.arange(n_high) / fps_high
    signal_high = (
        0.35 * np.sin(2 * np.pi * f0 * t_high)
        + 0.30 * np.sin(2 * np.pi * 2 * f0 * t_high)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t_high)
    )

    fps_low = 5.0
    n_low = int(fps_low * 20)
    t_low = np.arange(n_low) / fps_low
    # 3-я гармоника (3 Гц) физически недоступна при fps=5 (nyquist=2.5 Гц) —
    # намеренно НЕ добавляем её, это и есть проверяемый случай.
    signal_low = 0.35 * np.sin(2 * np.pi * f0 * t_low) + 0.30 * np.sin(2 * np.pi * 2 * f0 * t_low)

    est_high = estimate_hr(signal_high, fps_high, band, method="harmonic_sum")
    est_low = estimate_hr(signal_low, fps_low, band, method="harmonic_sum")

    print(f"  fps=30: {est_high.freq_hz:.3f} Hz ({est_high.bpm:.1f} BPM), "
          f"fps=5: {est_low.freq_hz:.3f} Hz ({est_low.bpm:.1f} BPM), истина: {f0} Hz")
    assert abs(est_high.freq_hz - f0) < 0.1, f"sanity: fps=30 должен найти {f0} Hz, получено {est_high.freq_hz}"
    assert abs(est_low.freq_hz - f0) < 0.1, (
        f"fps=5 (3-я гармоника недоступна) должен находить ту же частоту {f0} Hz, а не смещаться "
        f"октавой выше на 2f0 из-за старого бонуса за недоступность (задача 35), "
        f"получено {est_low.freq_hz:.3f} Hz"
    )


def test_octave_error_detector_finds_errors_on_drifting_true_signal():
    """Регрессия для задачи 28: старая метрика (задача 27) сравнивала
    оценку с МЕДИАНОЙ ПО ВСЕМУ ВИДЕО и находила 0% ошибок на дрейфующем
    истинном пульсе (см. докстринг scripts/compare_frequency_methods.py —
    реальные ошибки на Веронике лежали у 2×/3× ЛОКАЛЬНОГО значения ~61.4
    BPM, а не у 2× глобальной медианы 69.9 BPM). Здесь — синтетический
    дрейфующий ряд (плавно 60->61 BPM за 30 окон) с тремя ИСКУССТВЕННО
    внедрёнными ошибками (×2, ×3, ÷2 от локального окружения) — детектор
    должен найти именно эти три окна и не более."""
    print("\n=== Тест: детектор октавных ошибок находит их на дрейфующем сигнале (задача 28) ===")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_frequency_methods import local_median_series, octave_error_labels, _grid_step_bpm

    t = np.arange(0, 30, 1.0)
    bpm = 60.0 + 0.5 * np.sin(t / 10.0)  # плавный дрейф, БЕЗ ошибок

    up_idx, down_idx, x3_idx = 15, 20, 25
    m_at_up = float(np.median(np.delete(bpm, up_idx)))  # приближённо локальное окружение
    bpm[up_idx] = m_at_up * 2.0
    m_at_x3 = float(np.median(np.delete(bpm, x3_idx)))
    bpm[x3_idx] = m_at_x3 * 3.0
    m_at_down = float(np.median(np.delete(bpm, down_idx)))
    bpm[down_idx] = m_at_down / 2.0

    grid_step = _grid_step_bpm(30.0, 300)
    local_medians = local_median_series(t, bpm)
    up, down = octave_error_labels(bpm, local_medians, grid_step)

    print(f"  найдено up={np.where(up)[0].tolist()}, down={np.where(down)[0].tolist()}")

    assert up[up_idx] and up[x3_idx], "детектор должен найти обе внедрённые ошибки ×2 и ×3"
    assert down[down_idx], "детектор должен найти внедрённую ошибку ÷2"
    assert int(up.sum()) == 2, f"ложные срабатывания up на чистых окнах: {np.where(up)[0].tolist()}"
    assert int(down.sum()) == 1, f"ложные срабатывания down на чистых окнах: {np.where(down)[0].tolist()}"
    print("  [OK] детектор нашёл ровно 3 внедрённые ошибки, ложных срабатываний нет")


def test_continuity_check_prefers_peak_near_recent_history_within_tolerance():
    """Регрессия: если в полосе два локальных максимума СОПОСТАВИМОЙ
    мощности, а один явно ближе к медиане недавних BPM-оценок — при
    continuity_power_tolerance, покрывающем разницу их мощности, должен
    выбираться ближний к истории, а не автоматически глобальный максимум.
    При continuity_power_tolerance=0.0 (или без recent_median_hz) поведение
    ДОЛЖНО остаться обычным argmax — continuity не подменяет измерение, а
    только разрешает неоднозначность между уже существующими кандидатами."""
    print("\n=== Тест: continuity-проверка выбирает пик ближе к истории при сопоставимой мощности ===")
    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    fa, fb = 1.0, 1.3  # 60 BPM vs 78 BPM, fb немного мощнее

    rng = np.random.default_rng(5)
    signal = 1.0 * np.sin(2 * np.pi * fa * t) + 1.05 * np.sin(2 * np.pi * fb * t) + 0.03 * rng.normal(0, 1, n)

    plain = estimate_hr(signal, fps, band, method="welch")
    near_a = estimate_hr(signal, fps, band, method="welch", recent_median_hz=fa, continuity_power_tolerance=0.15)
    near_a_zero_tolerance = estimate_hr(
        signal, fps, band, method="welch", recent_median_hz=fa, continuity_power_tolerance=0.0
    )

    print(f"  без истории: {plain.bpm:.1f} BPM")
    print(f"  история near fa={fa * 60:.0f} BPM, tolerance=0.15: {near_a.bpm:.1f} BPM")
    print(f"  история near fa={fa * 60:.0f} BPM, tolerance=0.0:  {near_a_zero_tolerance.bpm:.1f} BPM")

    assert abs(plain.bpm - fb * 60) < 3.0, "sanity: без continuity должен побеждать чуть более мощный fb"
    assert abs(near_a.bpm - fa * 60) < 3.0, (
        "с историей рядом с fa и допуском, покрывающим разницу мощности, должен победить fa"
    )
    assert abs(near_a_zero_tolerance.bpm - plain.bpm) < 1e-6, (
        "tolerance=0.0 должен вести себя ТОЧНО как обычный argmax (continuity не подменяет измерение)"
    )


def test_physiological_rate_limit_blocks_implausible_jump_without_power_advantage():
    """Регрессия задачи 37: ЖЁСТКИЙ предел скорости изменения BPM —
    ОТДЕЛЬНЫЙ от мягкой continuity выше (test_continuity_check_*) и СТРОЖЕ
    её. Кандидат, "прыгнувший" от recent_median_hz дальше max_change_hz,
    должен отклоняться в пользу ближайшего допустимого локального
    максимума, ЕСЛИ у него нет явного (>= min_power_advantage раз)
    превосходства по мощности — а при наличии такого превосходства прыжок,
    наоборот, должен разрешаться (сильное прямое свидетельство важнее
    мягкого предела)."""
    print("\n=== Тест: жёсткий физиологический предел скорости BPM (задача 37) ===")
    from rppg.signal.frequency import _apply_physiological_rate_limit

    freqs = np.array([1.0, 1.2, 1.4, 2.0])  # Hz
    recent_median_hz = 1.0
    max_change_hz = 0.3  # допустимый диапазон без превосходства мощности: [0.7, 1.3] Hz
    selected_idx = 3  # кандидат на 2.0 Гц (уже выбран argmax/continuity)

    power_no_advantage = np.array([1.0, 3.0, 1.0, 3.5])   # 2.0 Гц лишь немного мощнее альтернативы (1.2 Гц)
    power_with_advantage = np.array([1.0, 3.0, 1.0, 10.0])  # 2.0 Гц явно (>=2x) мощнее альтернативы

    idx_no_adv = _apply_physiological_rate_limit(
        freqs, power_no_advantage, recent_median_hz, selected_idx, max_change_hz,
        power_is_log=False, min_power_advantage=2.0,
    )
    idx_with_adv = _apply_physiological_rate_limit(
        freqs, power_with_advantage, recent_median_hz, selected_idx, max_change_hz,
        power_is_log=False, min_power_advantage=2.0,
    )

    print(f"  без превосходства мощности: выбрано {freqs[idx_no_adv]:.1f} Hz (ожидание: 1.2 Hz — блок прыжка)")
    print(f"  с превосходством мощности:  выбрано {freqs[idx_with_adv]:.1f} Hz (ожидание: 2.0 Hz — прыжок разрешён)")

    assert freqs[idx_no_adv] == 1.2, (
        f"без явного превосходства мощности прыжок должен блокироваться жёстким пределом, "
        f"выбран {freqs[idx_no_adv]} Hz вместо ближайшего допустимого 1.2 Hz"
    )
    assert freqs[idx_with_adv] == 2.0, (
        f"с явным (>=2x) превосходством мощности прыжок должен разрешаться, "
        f"выбран {freqs[idx_with_adv]} Hz вместо 2.0 Hz"
    )


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


def test_bandpass_sos_retains_amplitude_at_low_normalized_cutoff():
    """Регрессия задачи 33: переход с (b,a)+filtfilt на SOS+sosfiltfilt не
    должен искажать амплитуду сигнала В ПОЛОСЕ пропускания — особенно при
    типичной для этого проекта низкой нормированной частоте среза
    (low_hz=0.7 / nyquist=15 Гц при fps=30 = 0.047), где (b,a)-форма
    численно наименее устойчива. 1.2 Гц (72 BPM) — юнит-амплитудная
    синусоида строго внутри рабочей полосы (0.7,4.0) Гц."""
    print("\n=== Тест: SOS-фильтр сохраняет амплитуду в полосе при низком cutoff (задача 33) ===")
    fps = 30.0
    n = int(fps * 20)
    t = np.arange(n) / fps
    x = np.sin(2 * np.pi * 1.2 * t)  # юнит-амплитуда, 72 BPM, внутри (0.7,4.0) Hz

    filtered = bandpass_filter(x, fps, *BAND, order=4)

    # отбросить края (переходный процесс zero-phase фильтра на границах окна)
    edge = int(fps * 3)
    core = filtered[edge:-edge]
    amplitude = (core.max() - core.min()) / 2.0

    print(f"  амплитуда после фильтрации: {amplitude:.4f} (истина: 1.0)")
    assert 0.95 <= amplitude <= 1.05, (
        f"полосовой SOS-фильтр исказил амплитуду сигнала в полосе больше чем на 5%: {amplitude:.4f}"
    )


def test_roi_mean_rgb_truncated_mean_recovers_subquantization_signal():
    """Регрессия задачи 31 (КРИТИЧНАЯ, отменяет более раннее решение в
    пользу медианы): усечённое среднее должно восстанавливать пульсовую
    модуляцию яркости, ЦЕЛИКОМ находящуюся ПОД порогом одного уровня
    квантования (peak-to-peak 0.3 из 256 градаций) — пространственное
    усреднение по ~5000 пикселям ROI снижает шум квантования в sqrt(N) раз.
    Медиана на ТОЙ ЖЕ выборке физически не может этого сделать: она сама
    квантована на целочисленную пиксельную сетку и отражает плавный сдвиг
    среднего яркости СТУПЕНЬКОЙ, а не непрерывно. Явный guard против
    повторного возврата к медиане в roi_mean_rgb."""
    print("\n=== Тест: усечённое среднее восстанавливает суб-квантовый сигнал, медиана — нет (задача 31) ===")
    from rppg.face.roi import roi_mean_rgb
    from rppg.signal.quality import dominant_frequency_and_snr

    fps = 30.0
    duration_s = 20.0
    n_frames = int(fps * duration_s)
    f0 = 1.2  # 72 BPM
    t = np.arange(n_frames) / fps

    side = 71  # 71*71 = 5041 пикселей, ~5000 как в критерии приёмки
    rng = np.random.default_rng(42)
    texture = rng.uniform(-3.0, 3.0, size=(side, side))  # фиксированная пространственная текстура кожи
    dc = 128.0
    amplitude = 0.15  # peak-to-peak 0.3 уровня — глубоко под порогом одного LSB

    mask = np.ones((side, side), dtype=np.uint8)
    trunc_series = np.zeros(n_frames)
    median_series = np.zeros(n_frames)

    for i, ti in enumerate(t):
        modulation = amplitude * np.sin(2 * np.pi * f0 * ti)
        continuous = dc + modulation + texture
        quantized = np.clip(np.round(continuous), 0, 255).astype(np.uint8)
        frame = np.stack([quantized, quantized, quantized], axis=-1)  # (H,W,3) BGR, все каналы равны

        rgb, valid, n_valid = roi_mean_rgb(frame, mask, min_valid_fraction=0.6, trim_fraction=0.10)
        assert valid and n_valid > 4500
        trunc_series[i] = rgb[1]

        ys, xs = np.where(mask > 0)
        pixels = frame[ys, xs].astype(np.float64)
        brightness = pixels.mean(axis=1)
        good = (brightness > 10) & (brightness < 250)
        median_series[i] = np.median(pixels[good], axis=0)[1]  # если бы использовалась медиана

    _, trunc_snr_db = dominant_frequency_and_snr(trunc_series, fps, BAND)
    _, median_snr_db = dominant_frequency_and_snr(median_series, fps, BAND)

    print(f"  усечённое среднее: SNR={trunc_snr_db:.1f} dB   медиана (для сравнения): SNR={median_snr_db:.1f} dB")
    assert trunc_snr_db > 20.0, (
        f"усечённое среднее должно восстанавливать суб-квантовый сигнал, SNR={trunc_snr_db:.1f} dB"
    )
    assert median_snr_db < 5.0, (
        f"медиана НЕ должна восстанавливать суб-квантовый сигнал (guard против регресса задачи 31) — "
        f"если это упало, кто-то заменил усечённое среднее обратно на медиану, получено SNR={median_snr_db:.1f} dB"
    )


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


def test_temporal_consistency_does_not_penalize_spectral_grid_jitter():
    """Регрессия для задачи 13в: дрожание BPM МЕНЬШЕ шага частотной сетки
    эстиматора физически неотличимо от дискретности самой сетки, а не
    признак нестабильности сигнала — formula не должна штрафовать за него."""
    print("\n=== Тест: temporal_consistency не штрафует дрожание в пределах разрешения спектра (13в) ===")
    from rppg.signal.quality import temporal_consistency_score, resolvable_bpm_difference

    fps, n_samples = 30.0, 300
    resolution = resolvable_bpm_difference(fps, n_samples, "welch")
    print(f"  resolvable_bpm_difference(fps=30, n=300, welch) = {resolution:.4f} BPM")
    # Задача 18: nperseg=min(300, max(64, int(30*8)))=240 -> 1.44*30/240*60 ≈ 10.8 BPM
    # (старая формула через zero-padded n_fft давала ошибочные ≈0.88 BPM).
    assert 5.0 < resolution < 20.0, "sanity: реальная ширина лепестка Welch-окна на 10с/30fps — единицы-десятки BPM"

    # Дрожание ЗАМЕТНО МЕНЬШЕ шага сетки (5%) — при жёстком max_expected_change_bpm
    # (меньше самого шага сетки) старая формула ложно оштрафовала бы даже это.
    tiny_jitter = [72.0, 72.0 + resolution * 0.05, 72.0, 72.0 + resolution * 0.05, 72.0, 72.0 + resolution * 0.05]
    score_without_floor = temporal_consistency_score(tiny_jitter, max_expected_change_bpm=0.01, min_resolvable_bpm=0.0)
    score_with_floor = temporal_consistency_score(
        tiny_jitter, max_expected_change_bpm=0.01, min_resolvable_bpm=resolution
    )
    print(f"  без нижней границы: {score_without_floor:.3f}, с нижней границей: {score_with_floor:.3f}")
    assert score_without_floor < 0.5, "sanity: без нижней границы жёсткий max_expected_change_bpm душит даже мелкое дрожание"
    assert score_with_floor > 0.9, (
        "дрожание в пределах разрешения спектра НЕ должно штрафоваться, если применена нижняя граница"
    )


def test_resolvable_bpm_difference_reflects_main_lobe_width():
    """Регрессия для задачи 18: resolvable_bpm_difference должна отражать
    РЕАЛЬНУЮ разрешающую способность Welch-эстиматора (ширина главного
    лепестка окна на СЕГМЕНТЕ nperseg), а не плотность zero-padded сетки
    (старая формула fps/n_fft). Классический критерий разрешения: два
    равных по амплитуде тона, отстоящие МЕНЬШЕ ширины лепестка, сливаются в
    спектре в ОДИН горб без провала между ними; отстоящие ЗАМЕТНО БОЛЬШЕ —
    дают ДВА различимых локальных максимума. Используется
    frequency.estimate_hr_welch напрямую (та же nperseg/nfft сетка, которую
    resolvable_bpm_difference("welch") зеркалит), а не quality.compute_psd,
    у которого ДРУГАЯ (для SNR, не для локализации пика) формула nperseg."""
    print("\n=== Тест: resolvable_bpm_difference отражает ширину лепестка окна (задача 18) ===")
    from rppg.signal.quality import resolvable_bpm_difference
    from rppg.signal.frequency import estimate_hr_welch

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    f0_hz = 1.2  # 72 BPM

    resolution_hz = resolvable_bpm_difference(fps, n, "welch") / 60.0

    def _n_local_maxima_between(delta_hz: float) -> int:
        signal = np.sin(2 * np.pi * (f0_hz - delta_hz / 2) * t) + np.sin(2 * np.pi * (f0_hz + delta_hz / 2) * t)
        est = estimate_hr_welch(signal, fps, band)
        mask = (est.freqs_hz >= f0_hz - delta_hz) & (est.freqs_hz <= f0_hz + delta_hz)
        p = est.power[mask]
        if len(p) < 3:
            return 0
        return int(np.sum((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])))

    n_merged = _n_local_maxima_between(delta_hz=resolution_hz * 0.3)
    n_resolved = _n_local_maxima_between(delta_hz=resolution_hz * 4.0)
    print(f"  resolution={resolution_hz * 60:.2f} BPM, слитые тона (Δ={resolution_hz * 0.3 * 60:.2f} BPM): "
          f"{n_merged} локальных максимумов, разнесённые тона (Δ={resolution_hz * 4.0 * 60:.2f} BPM): "
          f"{n_resolved} локальных максимумов")

    assert n_merged <= 1, (
        f"тона, отстоящие МЕНЬШЕ разрешающей способности, должны сливаться в спектре в один горб "
        f"(получено {n_merged} локальных максимумов)"
    )
    assert n_resolved >= 2, (
        f"тона, отстоящие ЗАМЕТНО БОЛЬШЕ разрешающей способности, должны быть различимы как два горба "
        f"(получено {n_resolved} локальных максимумов)"
    )


def test_cross_roi_agreement_robust_to_single_outlier_roi():
    """Регрессия для задачи 13б: раньше сравнивался РАЗМАХ (max-min) — одно
    выпавшее ROI (например, засвеченная щека) утягивало его и обнуляло
    компоненту целиком, даже если ОСТАЛЬНЫЕ ROI прекрасно согласны. MAD
    (медианное абсолютное отклонение) должен быть устойчив к такому выбросу."""
    print("\n=== Тест: cross_roi_agreement устойчив к одному выпавшему ROI (13б) ===")
    from rppg.signal.quality import cross_roi_agreement_score

    two_agree_one_outlier = {"forehead": 70.0, "left_cheek": 71.0, "right_cheek": 150.0}
    all_disagree = {"forehead": 70.0, "left_cheek": 110.0, "right_cheek": 150.0}

    score_outlier, n_outlier, insufficient_outlier = cross_roi_agreement_score(two_agree_one_outlier)
    score_all_disagree, n_disagree, insufficient_disagree = cross_roi_agreement_score(all_disagree)
    print(f"  два согласны + один выброс: {score_outlier:.3f} (n={n_outlier})")
    print(f"  все три расходятся:         {score_all_disagree:.3f} (n={n_disagree})")

    assert score_outlier > 0.8, "два согласных ROI из трёх должны давать высокий score несмотря на один выброс"
    assert score_all_disagree < score_outlier, "когда РЕАЛЬНО все три расходятся, score должен быть ниже"
    assert not insufficient_outlier and not insufficient_disagree, "3 ROI с валидным BPM -> компонента измерима"
    assert n_outlier == 3 and n_disagree == 3, "без snr_by_roi_db фильтрация не применяется -> все ROI участвуют"


def test_cross_roi_agreement_ignores_low_snr_roi():
    """Регрессия для задачи 13б (порог на SNR-разрыв сделан ОТНОСИТЕЛЬНЫМ в
    задаче 17, см. cross_roi_agreement_score): сравнивать оценку заведомо
    шумного ROI (SNR намного ниже лучшего в этом же окне — пульсовой
    компоненты там нет) с оценкой хорошего ROI бессмысленно. Шумный ROI
    должен ИСКЛЮЧАТЬСЯ из сравнения, а не тянуть агрегированную оценку вниз
    наравне с хорошими."""
    print("\n=== Тест: cross_roi_agreement игнорирует ROI с низким собственным SNR (13б/17) ===")
    from rppg.signal.quality import cross_roi_agreement_score

    bpm_by_roi = {"forehead": 70.0, "left_cheek": 71.0, "right_cheek": 140.0}
    # right_cheek отстаёт от лучшего (forehead, 8.0 дБ) на 14 дБ — намного
    # больше стандартного max_snr_gap_db=6.0, должен быть исключён.
    snr_by_roi_db = {"forehead": 8.0, "left_cheek": 7.5, "right_cheek": -6.0}

    score_without_filter, n_without, _ = cross_roi_agreement_score(bpm_by_roi)
    score_with_filter, n_with, _ = cross_roi_agreement_score(
        bpm_by_roi, snr_by_roi_db=snr_by_roi_db, max_snr_gap_db=6.0
    )
    print(f"  без фильтра по SNR: {score_without_filter:.3f} (n={n_without})")
    print(f"  с фильтром по SNR:  {score_with_filter:.3f} (n={n_with})")

    assert score_with_filter > score_without_filter, (
        "исключение заведомо шумного ROI по его собственному SNR должно повышать согласие "
        "оставшихся (реально согласных) ROI"
    )
    assert score_with_filter > 0.9, "после исключения шумного ROI два оставшихся почти идентичных значения должны давать высокий score"
    assert n_with == 2, "right_cheek должен быть исключён относительным SNR-фильтром"


def test_cross_roi_agreement_guarantees_minimum_two_participants():
    """Регрессия для задачи 17: раньше абсолютный SNR-фильтр мог отсечь
    ВСЕ ROI до < 2 участников, даже если формально было доступно 3 —
    компонента тогда молча вырождалась в нейтральную 0.5 при каждом окне
    реальной записи (см. cross_roi_agreement_score). Теперь минимум 2
    участника гарантируется явно: если относительный gap-фильтр оставил
    только 1 ROI, второй по величине собственного SNR добавляется
    принудительно — компонента остаётся ИЗМЕРИМОЙ, а не "недостаточно
    данных", даже когда реально хорош только один источник."""
    print("\n=== Тест: cross_roi_agreement гарантирует минимум 2 участника (задача 17) ===")
    from rppg.signal.quality import cross_roi_agreement_score

    bpm_by_roi = {"forehead": 72.0, "left_cheek": 73.0, "right_cheek": 110.0}
    # Только forehead проходит стандартный gap=6.0 дБ от лучшего; left_cheek
    # и right_cheek оба отстают намного больше — но left_cheek отстаёт МЕНЬШЕ
    # right_cheek, поэтому должен быть добавлен вторым принудительно.
    snr_by_roi_db = {"forehead": 10.0, "left_cheek": -20.0, "right_cheek": -25.0}

    score, n_participants, insufficient = cross_roi_agreement_score(
        bpm_by_roi, snr_by_roi_db=snr_by_roi_db, max_snr_gap_db=6.0
    )
    print(f"  score={score:.3f}, n_participants={n_participants}, insufficient_in_principle={insufficient}")

    assert n_participants >= 2, "минимум 2 участника должен быть гарантирован, даже если gap-фильтр оставил 1"
    assert not insufficient, "3 ROI с валидным BPM доступны -> компонента в принципе измерима"


def test_cross_roi_agreement_single_source_is_insufficient_in_principle():
    """Регрессия для задачи 17: когда исходно доступен ТОЛЬКО ОДИН ROI с
    валидным BPM (например, method=head_motion — источник ровно один),
    компонента в принципе НЕ МОЖЕТ быть измерена — это отдельный случай от
    "измерена и оказалась низкой", и должен различаться отдельным флагом,
    а не сливаться с рабочим режимом в статистике."""
    print("\n=== Тест: единственный источник -> insufficient_in_principle=True (задача 17) ===")
    from rppg.signal.quality import cross_roi_agreement_score

    score, n_participants, insufficient = cross_roi_agreement_score({"face": 72.0})
    print(f"  score={score:.3f}, n_participants={n_participants}, insufficient_in_principle={insufficient}")

    assert score == 0.5
    assert n_participants == 1
    assert insufficient, "единственный ROI -> компонента в принципе неизмерима"


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
    band = (0.7, 4.0)
    rng = np.random.default_rng(1)

    clean_tone = np.sin(2 * np.pi * 1.0 * t) + 0.05 * rng.normal(0, 1, n)
    clean_score, clean_warnings, _ = harmonic_plausibility(clean_tone, fps, peak_freq_hz=1.0, band_hz=band)

    # Пик ошибочно "найден" на 2 Hz, но в сигнале сопоставимая энергия на
    # истинной частоте 1 Hz (= заявленный_пик / 2) -> подозрение на путаницу.
    ambiguous = np.sin(2 * np.pi * 1.0 * t) + 0.9 * np.sin(2 * np.pi * 2.0 * t)
    ambiguous_score, ambiguous_warnings, _ = harmonic_plausibility(ambiguous, fps, peak_freq_hz=2.0, band_hz=band)

    print(f"  чистый тон на f: score={clean_score:.2f}, warnings={clean_warnings}")
    print(f"  пик на 2f при сильной энергии на f: score={ambiguous_score:.2f}, warnings={ambiguous_warnings}")

    assert clean_score >= 0.9 and not clean_warnings, "чистый одиночный тон не должен давать ложных срабатываний"
    assert ambiguous_score < 0.7 and ambiguous_warnings, (
        "сопоставимая энергия на f/2 относительно заявленного пика на 2f должна флагироваться"
    )


def test_harmonic_check_ignores_1_over_f_noise_floor():
    """Регрессия для задачи 4: раньше сравнивалась ДОЛЯ мощности p_half/p_f
    с фиксированным порогом 0.7 — это ломалось именно на ЧИСТОМ сигнале,
    потому что в детрендированном rPPG-сигнале спектр падает как 1/f
    (красный шум): при пульсе 60-72 BPM субгармоника f/2 попадает в
    0.5-0.6 Hz, где мощности 1/f-шума заведомо больше, чем на самой
    пульсовой частоте — p_half/p_f > 0.7 срабатывало почти ВСЕГДА, не имея
    отношения к реальной субгармонике.

    "Чистый пульс 1.0 Гц + красный шум 1/f" должен давать harmonic_score
    РОВНО 1.0 (см. задача 4, критерий приёмки), а "пульс 2.0 Гц + равный по
    мощности пик на 1.0 Гц" — заметно меньше 1.0 (реальная субгармоника,
    должна флагироваться)."""
    print("\n=== Тест: гармоническая проверка НЕ путает 1/f-шум с реальной субгармоникой (задача 4) ===")
    from rppg.signal.quality import harmonic_plausibility

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    rng = np.random.default_rng(7)

    # Красный шум (1/f) — интеграл белого шума (винеровский процесс),
    # классическая аппроксимация 1/f^2 спектра остаточного дрейфа
    # освещения/дыхания/микродвижений в детрендированном rPPG-сигнале.
    red_noise = np.cumsum(rng.normal(0, 1, n))
    red_noise -= np.mean(red_noise)
    red_noise /= np.std(red_noise) + 1e-9
    clean_pulse_plus_red_noise = np.sin(2 * np.pi * 1.0 * t) + 0.3 * red_noise
    score_clean, warnings_clean, _ = harmonic_plausibility(
        clean_pulse_plus_red_noise, fps, peak_freq_hz=1.0, band_hz=band
    )
    print(f"  чистый пульс 1.0 Гц + 1/f-шум: score={score_clean:.3f}, warnings={warnings_clean}")
    assert score_clean == 1.0 and not warnings_clean, (
        f"1/f-шум не должен ложно флагироваться как субгармоника (score={score_clean}, ожидалось 1.0)"
    )

    # Реальная субгармоника: равный по мощности пик на f/2=1.0 Hz при
    # заявленном пике 2.0 Hz — настоящий локальный горб спектра, не точка
    # на монотонном спаде.
    real_subharmonic = np.sin(2 * np.pi * 2.0 * t) + np.sin(2 * np.pi * 1.0 * t)
    score_real, warnings_real, _ = harmonic_plausibility(real_subharmonic, fps, peak_freq_hz=2.0, band_hz=band)
    print(f"  пульс 2.0 Гц + равный пик на 1.0 Гц: score={score_real:.3f}, warnings={warnings_real}")
    assert score_real < 0.9 and warnings_real, (
        f"реальная субгармоника равной мощности должна заметно снижать score (получено {score_real})"
    )


def test_second_harmonic_energy_is_not_penalized():
    """Регрессия для задачи 20: горб на 2f — ОЖИДАЕМЫЙ признак чистого
    несинусоидального пульса (резкий систолический подъём, пологий
    диастолический спад), а не путаницы, и больше НЕ штрафуется (см.
    docstring harmonic_plausibility). Раньше сигнал с сильной, физиологически
    правдоподобной 2-й гармоникой получал предупреждение "путаница с 2-й
    гармоникой" и штраф — ровно на том сигнале, где качество пульса самое
    высокое (найдено на реальной записи: пик 85.5 BPM с горбом на 171 BPM).
    f=1.2 Гц выбрана так, что f/2=0.6 Гц < band[0]=0.7 Гц — субгармонический
    тест физически неинформативен и не мешает изолированно проверить именно
    2f-ветку."""
    print("\n=== Тест: энергия на 2f не штрафуется, только логируется как harmonic_support (задача 20) ===")
    from rppg.signal.quality import harmonic_plausibility

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    f_hz = 1.2  # 72 BPM, f/2=0.6 Hz < band[0] -> субгармонический тест не сработает

    clean_pulse_with_second_harmonic = np.sin(2 * np.pi * f_hz * t) + 0.8 * np.sin(2 * np.pi * 2.0 * f_hz * t)
    score, warnings, harmonic_support = harmonic_plausibility(
        clean_pulse_with_second_harmonic, fps, peak_freq_hz=f_hz, band_hz=band
    )
    print(f"  score={score:.3f}, harmonic_support={harmonic_support}, warnings={warnings}")

    assert score == 1.0, f"горб на 2f не должен штрафовать score (получено {score})"
    assert not warnings, "горб на 2f не должен генерировать warning о путанице (задача 20)"
    assert harmonic_support, "значимый горб на 2f должен быть отражён в harmonic_support (информационно)"


def test_illumination_flicker_detected_via_background_roi():
    """Регрессия для п.22 (обновлена для задачи 7): если фоновый ROI (вне
    лица, например стена) САМ ПО СЕБЕ показывает узкий стабильный пик на
    частоте, совпадающей с "пульсом" на лице, это ОДНОконный "кандидат" на
    мерцание — детектор возвращает (candidate, bg_freq_hz, bg_snr_db), а
    финальное решение flicker_suspected (с требованием устойчивости в
    нескольких окнах подряд) принимает assess_quality, не эта функция (см.
    test_flicker_requires_consecutive_windows_to_confirm ниже). Фон, который
    просто шумит без узкополосного пика, не должен давать кандидата вовсе."""
    print("\n=== Тест: детектор мерцания освещения через фоновый ROI (п.22) ===")
    from rppg.signal.quality import detect_illumination_flicker

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    face_peak_hz = 1.2  # 72 BPM

    flicker_bg = 2.0 * np.sin(2 * np.pi * face_peak_hz * t) + 0.05 * np.random.default_rng(2).normal(0, 1, n)
    candidate, bg_freq, bg_snr = detect_illumination_flicker(flicker_bg, fps, face_peak_hz, band)

    clean_bg = np.random.default_rng(3).normal(0, 1, n)
    clean_candidate, clean_bg_freq, clean_bg_snr = detect_illumination_flicker(clean_bg, fps, face_peak_hz, band)

    print(f"  фон совпадает с частотой лица: candidate={candidate}, bg_freq={bg_freq:.2f} Hz, snr={bg_snr:.1f} dB")
    print(f"  фон — просто шум: candidate={clean_candidate}, snr={clean_bg_snr:.1f} dB")

    assert candidate, "совпадающий узкополосный пик фона должен давать candidate=True"
    assert not clean_candidate, "шумный фон без узкополосного пика не должен давать candidate=True"


def test_flicker_requires_consecutive_windows_to_confirm():
    """Регрессия для задачи 7 (устойчивость во времени): мерцание —
    стабильное во времени явление. Одно-единственное окно с совпадающим
    фоновым пиком НЕ должно блокировать публикацию (могло быть случайным
    шумовым пиком, усиленным z-score нормализацией) — только
    flicker_min_consecutive_windows окон подряд."""
    print("\n=== Тест: flicker_suspected требует несколько окон подряд (задача 7) ===")
    from rppg.signal.quality import assess_quality, dominant_frequency_and_snr, SQIInputs
    from rppg.config import QualityConfig

    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    face_peak_hz = 1.2
    qcfg = QualityConfig()

    flicker_bg = 2.0 * np.sin(2 * np.pi * face_peak_hz * t) + 0.05 * np.random.default_rng(2).normal(0, 1, n)

    def _assess(recent_candidates):
        return assess_quality(
            SQIInputs(
                spectral_snr_db=20.0, peak_freq_hz=face_peak_hz, fps=fps, band_hz=band,
                bpm_by_roi={"forehead": face_peak_hz * 60.0},
                background_signal=flicker_bg,
                recent_flicker_candidates=recent_candidates,
            ),
            qcfg,
        )

    single_window = _assess(recent_candidates=None)  # первое окно вообще, история пуста
    print(f"  1-е окно (история пуста): flicker_candidate={single_window.flicker_candidate}, "
          f"flicker_suspected={single_window.flicker_suspected}")
    assert single_window.flicker_candidate, "sanity: этот фон должен давать candidate=True"
    assert not single_window.flicker_suspected, (
        "одно-единственное окно НЕ должно блокировать публикацию по мерцанию"
    )

    almost_confirmed = _assess(recent_candidates=[True] * (qcfg.flicker_min_consecutive_windows - 2))
    assert not almost_confirmed.flicker_suspected, (
        f"{qcfg.flicker_min_consecutive_windows - 1} окон подряд < порога — ещё не должно подтверждаться"
    )

    confirmed = _assess(recent_candidates=[True] * (qcfg.flicker_min_consecutive_windows - 1))
    print(f"  {qcfg.flicker_min_consecutive_windows} окон подряд: flicker_suspected={confirmed.flicker_suspected}")
    assert confirmed.flicker_suspected, (
        f"{qcfg.flicker_min_consecutive_windows} совпадающих окон подряд должны подтвердить мерцание"
    )

    broken_streak = _assess(recent_candidates=[True, False] * qcfg.flicker_min_consecutive_windows)
    assert not broken_streak.flicker_suspected, (
        "прерванная (не подряд) серия совпадений не должна подтверждать мерцание"
    )


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
        # 400 кадров = ~13.3с при 30fps: с задачи 6 min_seconds_before_estimate
        # по умолчанию РАВЕН window_seconds (10с, см. WindowConfig) — первая
        # оценка не считается, пока окно не заполнено полностью, поэтому нужно
        # покрыть >= 10с, а не ~6.7с, как раньше.
        for i in range(400):
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
        "(пусть и с publishable=False) за 400 кадров >= min_seconds_before_estimate"
    )
    print("  [OK] за 400 кадров без лица в кадре publishable не стал True ни разу")


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
        # 13с, не 8с: с задачи 6 min_seconds_before_estimate по умолчанию
        # РАВЕН window_seconds (10с) — окно короче 10с не даёт вообще ни
        # одной оценки (и значит ни одной строки лога), см. WindowConfig.
        n_frames = int(fps * 13.0)
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
        from rppg.structured_log import read_jsonl_records

        headers, records = read_jsonl_records(log_path)
        print(f"  заголовков: {len(headers)}, строк окон: {len(records)}")
        assert len(headers) == 1, (
            f"задача 26: КАЖДОЕ открытие лога должно писать РОВНО одну строку заголовка, "
            f"получено {len(headers)}"
        )
        assert len(records) > 0, "должна быть хотя бы одна строка окна за 13с синтетического видео"

        required_fields = {
            "timestamp_ms", "bpm", "per_roi_bpm", "publishable", "method_used",
            "frequency_method_used", "sqi_overall_score", "sqi_level",
            "sqi_spectral_snr_db", "sqi_cross_roi_agreement", "sqi_landmark_stability",
            "sqi_temporal_consistency", "sqi_harmonic_score", "sqi_flicker_suspected",
            "sqi_flicker_background_freq_hz", "sqi_flicker_background_snr_db",
            "sqi_subharmonic_check_informative",
            "respiration_rate_bpm", "warnings",
        }
        for record in records:
            missing = required_fields - set(record.keys())
            assert not missing, f"строка лога не содержит поля {missing}: {record}"
            assert isinstance(record["per_roi_bpm"], dict)
            assert isinstance(record["warnings"], list)

        last = records[-1]
        print(f"  последняя запись: bpm={last['bpm']:.2f}, sqi_overall={last['sqi_overall_score']:.3f}, "
              f"sqi_landmark_stability={last['sqi_landmark_stability']:.3f}, "
              f"sqi_temporal_consistency={last['sqi_temporal_consistency']:.3f}")


def test_publishable_matches_across_csv_and_jsonl():
    """Регрессия для задачи 16: PTSDPulseFeatures.publishable (то, что видит
    вызывающий код в памяти и что превращается в колонку CSV в
    scripts/run_on_video.py) и WindowLogRecord.publishable (JSONL-лог,
    structured_log.py) ОБЯЗАНЫ совпадать окно в окно — оба теперь читают
    ровно SQIResult.is_reliable, а не два независимо пересчитанных правила
    (см. docstring PTSDPulseFeatures.publishable). Раньше PTSDPulseFeatures.
    publishable был мягче (confidence >= порога, без учёта spectral_snr_db
    и flicker_suspected), и на реальных записях число publishable=True в
    CSV расходилось с числом publishable=True в JSONL для тех же самых окон
    (доказательство бага — три реальных прогона с точным совпадением "число
    publishable в CSV == число окон, провалившихся ТОЛЬКО по SNR")."""
    print("\n=== Тест: publishable совпадает между PTSDPulseFeatures и JSONL-логом (задача 16) ===")
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

        landmarks = _make_synthetic_landmarks_for_pipeline_test()
        synthetic_result = FaceFrameResult(
            detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
        )
        pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result

        true_bpm = 75.0
        rng = np.random.default_rng(21)
        fps = 30.0
        n_frames = int(fps * 15.0)
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

        assert log_path.exists(), "log_path должен быть создан, раз пайплайн выдал хотя бы одно окно"
        from rppg.structured_log import read_jsonl_records
        _headers, logged = read_jsonl_records(log_path)

        assert len(results) == len(logged), (
            f"число окон из process_frame ({len(results)}) не совпадает с числом строк JSONL ({len(logged)})"
        )

        for i, (result, record) in enumerate(zip(results, logged)):
            assert result.publishable == record["publishable"], (
                f"окно #{i} (t={result.timestamp_ms}ms): PTSDPulseFeatures.publishable="
                f"{result.publishable}, WindowLogRecord.publishable={record['publishable']} — "
                "два источника истины разошлись (см. задачу 16)"
            )

        n_publishable_from_results = sum(1 for r in results if r.publishable)
        n_publishable_from_jsonl = sum(1 for r in logged if r["publishable"])
        assert n_publishable_from_results == n_publishable_from_jsonl
        print(f"  [OK] {len(results)} окон, publishable совпадает поэлементно "
              f"({n_publishable_from_results} publishable=True с обеих сторон)")


def test_window_status_labels_degenerate_and_no_face_windows():
    """Регрессия для задачи 21: window_status в JSONL-логе должен различать
    "ok" (реальная, пусть и шумная, временная вариация ROI) от
    "degenerate_signal" (сигнал технически посчитан, но std ~ 0 после
    препроцессинга — dominant_frequency_and_snr возвращает -inf, см.
    quality.compute_psd). Раньше ОБА случая молча получали одинаковый
    status="ok" и тихо выпадали из перцентилей как "N inf/nan исключено" в
    analyze_log.py — без объяснения причины (см. критерий приёмки задачи 21:
    сумма окон по всем status равна общему числу окон, и причина видна)."""
    print("\n=== Тест: window_status различает ok/degenerate_signal (задача 21) ===")
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

    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )
    fps = 30.0

    def _run(frame_source, n_frames: int, log_path: Path) -> list[dict]:
        cfg = PipelineConfig()
        try:
            pipe = RPPGPipeline(cfg, log_path=log_path)
        except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
            print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
            return []
        pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result
        try:
            for i in range(n_frames):
                t_sec = i / fps
                pipe.process_frame(frame_source(t_sec), int(t_sec * 1000))
        finally:
            pipe.close()
        if not log_path.exists():
            return []
        from rppg.structured_log import read_jsonl_records
        _headers, records = read_jsonl_records(log_path)
        return records

    with tempfile.TemporaryDirectory() as tmpdir:
        # true_bpm=0.0 -> нет пульсации, но per-frame шум камеры (см.
        # _make_pulsating_frame) даёт РЕАЛЬНУЮ временную вариацию -> "ok"
        # (низкое качество, но НЕ вырожденный сигнал).
        rng_noise = np.random.default_rng(9)
        noisy_records = _run(
            lambda t_sec: _make_pulsating_frame(t_sec, 0.0, rng_noise),
            int(fps * 13.0), Path(tmpdir) / "noisy.jsonl",
        )
        assert noisy_records, "sanity: шумные (но не замороженные) кадры должны дать хотя бы одну строку лога"
        noisy_statuses = {r["window_status"] for r in noisy_records}
        assert noisy_statuses == {"ok"}, (
            f"кадры с реальной (пусть и шумовой) временной вариацией не должны получать "
            f"degenerate_signal/too_short, получено: {noisy_statuses}"
        )

        # ОДИН и тот же кадр (бит в бит) на всех итерациях -> нулевая
        # временная вариация ROI -> сигнал после препроцессинга вырождается
        # в std ~ 0 -> degenerate_signal.
        rng_frozen = np.random.default_rng(10)
        frozen_frame = _make_pulsating_frame(0.0, 0.0, rng_frozen)
        frozen_records = _run(
            lambda t_sec: frozen_frame,
            int(fps * 13.0), Path(tmpdir) / "frozen.jsonl",
        )
        assert frozen_records, "sanity: замороженный кадр должен дать хотя бы одну строку лога"
        frozen_statuses = {r["window_status"] for r in frozen_records}
        assert frozen_statuses == {"degenerate_signal"}, (
            f"полностью замороженный (нулевая временная вариация) кадр должен давать "
            f"window_status=degenerate_signal во всех окнах, получено: {frozen_statuses}"
        )

    print(f"  [OK] шумные кадры -> window_status=ok ({len(noisy_records)} окон), "
          f"замороженный кадр -> window_status=degenerate_signal ({len(frozen_records)} окон)")


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


def _make_pulse_wave_with_dicrotic_notch(
    bpm: float, fps: float, duration_s: float, notch_amplitude: float = 0.5, notch_phase: float = -1.5,
) -> np.ndarray:
    """Синтетическая пульсовая волна с выраженным вторичным подъёмом
    (дикротической выемкой) внутри каждого удара — сумма гармоник со
    сдвигом фазы у 2-й гармоники (тот же приём, что и в тестах
    harmonic_sum на несинусоидальность реального пульса)."""
    t = np.arange(int(fps * duration_s)) / fps
    f = bpm / 60.0
    return (
        np.sin(2 * np.pi * f * t)
        + notch_amplitude * np.sin(2 * np.pi * 2 * f * t + notch_phase)
        + 0.1 * np.sin(2 * np.pi * 3 * f * t)
    )


def test_cross_check_agreement_on_clean_signal():
    """Регрессия задачи 46: чистый однопиковый сигнал + верная спектральная
    оценка -> ближайший кандидат к peak_hr — сама f -> verdict="agreement",
    bpm не меняется, confidence_boost>0."""
    print("\n=== Тест: cross_check согласен на чистом сигнале (задача 46) ===")
    true_bpm = 72.0
    signal = make_synthetic_rgb(true_bpm, noise_std=0.05, seed=1)[:, 1]  # G-канал как proxy пульсовой волны
    result = cross_check_spectral_vs_peaks(true_bpm, signal, FPS, BAND)
    print(f"  verdict={result.verdict} ratio={result.ratio} peaks_hr={result.mean_hr_from_peaks}")
    assert result.verdict == "agreement", f"ожидался agreement, получено {result.verdict}"
    assert abs(result.bpm - true_bpm) < 1e-9
    assert result.confidence_boost > 0.0
    assert result.is_bpm_reliable and result.peaks_ok_for_ibi


def test_cross_check_corrects_subharmonic_spectral_estimate():
    """Регрессия задачи 46 (основной сценарий задачи 42/43): сигнал с
    истинными пиками на 120 BPM, но НАМЕРЕННО заниженная вдвое спектральная
    оценка (60 BPM, как в доказанном баге result_ya_squats.csv: 63.92 ->
    127.84) -> среди кандидатов {60, 120, 30} ближайший к peak_hr~120 —
    сам 2f=120 -> verdict="corrected_to_2f", bpm=120."""
    print("\n=== Тест: cross_check исправляет заниженную вдвое спектральную оценку (задача 46) ===")
    true_bpm = 120.0
    signal = make_synthetic_rgb(true_bpm, noise_std=0.05, seed=2)[:, 1]
    wrong_spectral_bpm = true_bpm / 2.0  # 60 BPM — намеренно неверная (субгармоническая) оценка
    result = cross_check_spectral_vs_peaks(wrong_spectral_bpm, signal, FPS, BAND)
    print(f"  verdict={result.verdict} ratio={result.ratio} corrected_bpm={result.bpm}")
    assert result.verdict == "corrected_to_2f", f"ожидался corrected_to_2f, получено {result.verdict}"
    assert abs(result.bpm - true_bpm) < 3.0, f"скорректированный bpm={result.bpm} должен быть близко к {true_bpm}"
    assert result.is_bpm_reliable and result.peaks_ok_for_ibi


def test_cross_check_flags_unexplained_mismatch_as_ambiguous():
    """Регрессия задачи 46: расхождение, не объяснимое ни близостью к f,
    ни к 2f, ни к f/2 достаточно уверенно (ratio далеко от 0.5/1.0/2.0,
    ближайший кандидат не превосходит следующего хотя бы вдвое по
    |ln(...)|) -> verdict="ambiguous", bpm остаётся исходным."""
    print("\n=== Тест: необъяснимое расхождение помечается ambiguous (задача 46) ===")
    signal = make_synthetic_rgb(80.0, noise_std=0.05, seed=3)[:, 1]  # истинные пики ~80 BPM
    implausible_spectral_bpm = 80.0 / 1.4  # ratio~1.4 — на полпути между f и 2f
    result = cross_check_spectral_vs_peaks(implausible_spectral_bpm, signal, FPS, BAND)
    print(f"  verdict={result.verdict} ratio={result.ratio}")
    assert result.verdict == "ambiguous", f"ожидался ambiguous, получено {result.verdict}"
    assert result.bpm == implausible_spectral_bpm, "при ambiguous bpm должен остаться исходным"
    assert not result.peaks_ok_for_ibi


def test_cross_check_dicrotic_notch_suppressed_by_prominence():
    """Регрессия задачи 44/46 (ПЕРЕСМОТРЕНО эмпирически — см. git history и
    docstring detect_pulse_peaks: расстояние, посчитанное от спектральной
    подсказки, ломало обнаружение реальных субгармонических ошибок
    циклической зависимостью). Дикротическая выемка в 2 раза удваивает
    счёт "сырых" пиков (find_peaks с одним физиологическим потолком
    расстояния), но у выемки заметно (в разы) меньше prominence, чем у
    настоящего систолического пика — _collapse_alternating_low_
    prominence_peaks должна убрать именно её, БЕЗ какой-либо спектральной
    подсказки. Критично и для задачи 46: ближайшим кандидатом на записи
    покоя обязан оказаться сам f, а не 2f."""
    print("\n=== Тест: дикротическая выемка подавляется по prominence (задача 44/46) ===")
    true_bpm = 62.0
    signal = _make_pulse_wave_with_dicrotic_notch(true_bpm, FPS, DURATION_S)

    mean_hr, peaks = mean_hr_from_peaks_in_window(signal, FPS)
    print(f"  mean_hr_from_peaks={mean_hr}, пиков={len(peaks)}")

    assert mean_hr is not None
    assert abs(mean_hr - true_bpm) < 5.0, (
        f"выемка должна быть подавлена по prominence, mean_hr_from_peaks должен остаться "
        f"близко к {true_bpm}, получено {mean_hr}"
    )

    result = cross_check_spectral_vs_peaks(true_bpm, signal, FPS, BAND)
    print(f"  cross_check: verdict={result.verdict} corrected_bpm={result.bpm}")
    assert result.verdict == "agreement", (
        f"верная спектральная оценка покоя НЕ должна ложно скорректироваться из-за выемки "
        f"(задача 44/46 — правка, чинящая нагрузку и ломающая покой, не годится), получено {result.verdict}"
    )
    assert abs(result.bpm - true_bpm) < 1e-9, "покой не должен быть поднят с 62 до ~124"


def test_hrv_rejects_physiologically_impossible_rmssd_at_high_hr():
    """Регрессия задачи 44, часть 3: RMSSD>200мс или pNN50>50% при пульсе
    выше 100 BPM физиологически невозможны и отбраковываются БЕЗУСЛОВНО —
    даже когда старый фильтр эктопических интервалов (сравнение с
    непосредственными соседями) не ловит их из-за регулярного чередования
    длинных/коротких интервалов (задача 43: "соседи тоже испорчены")."""
    print("\n=== Тест: HRV отбраковывает физиологически невозможный RMSSD при высоком пульсе (задача 44) ===")
    mean_ibi = 500.0  # 120 BPM
    n = 200
    # Строго альтернирующий короткий/длинный интервал — artifact_fraction
    # у существующего соседского фильтра остаётся 0.0 (ни один интервал не
    # выглядит выбросом относительно СВОИХ соседей, они тоже чередуются).
    ibi = np.array([mean_ibi + (60.0 if i % 2 == 0 else -60.0) for i in range(n)])

    mask = ectopic_artifact_mask(ibi)
    artifact_fraction = float((~mask).mean())
    hrv = extract_hrv_features_from_ibi(ibi, compute_frequency_domain=False)
    print(f"  mean_hr={hrv.mean_hr_bpm:.1f} rmssd={hrv.rmssd_ms:.1f} pnn50={hrv.pnn50_pct:.1f}% "
          f"artifact_fraction={artifact_fraction:.3f} publishable={hrv.publishable}")

    assert hrv.mean_hr_bpm > 100.0, "sanity: пульс должен быть выше 100 BPM для этой проверки"
    assert hrv.rmssd_ms > 200.0 or hrv.pnn50_pct > 50.0, (
        "sanity: сценарий должен реально давать физиологически невозможные значения"
    )
    assert artifact_fraction < 0.05, (
        "sanity: старый соседский фильтр НЕ должен отбраковать эти интервалы сам по себе — "
        "иначе тест не демонстрирует именно новую безусловную проверку (задача 44, часть 3)"
    )
    assert not hrv.publishable, (
        "физиологически невозможные RMSSD/pNN50 при пульсе >100 BPM должны безусловно "
        "блокировать публикацию HRV, даже если доля отбракованных интервалов низкая"
    )


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


def _make_pipeline_for_aggregate_test(cfg=None):
    """Общий хелпер для тестов задачи 25 — RPPGPipeline с реальной моделью
    Face Landmarker (нужна только для конструктора, process_frame НЕ
    вызывается: тесты дёргают _compute_aggregate напрямую как white-box)."""
    from rppg.pipeline import RPPGPipeline
    from rppg.config import PipelineConfig

    try:
        return RPPGPipeline(cfg or PipelineConfig()), None
    except FileNotFoundError as exc:
        return None, f"модель Face Landmarker недоступна ({exc})"
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        return None, f"не удалось инициализировать MediaPipe ({exc})"


def _make_cross_check_result(verdict: str, bpm: float, is_bpm_reliable: bool) -> "CrossCheckResult":
    from rppg.hrv.features import CrossCheckResult
    return CrossCheckResult(
        verdict=verdict, bpm=bpm, ratio=None, mean_hr_from_peaks=None,
        peaks_sub=np.array([]), is_bpm_reliable=is_bpm_reliable,
        peaks_ok_for_ibi=False, confidence_boost=0.0,
    )


def test_bpm_smoothing_corrects_unconfirmed_subharmonic_using_trusted_history():
    """Регрессия задачи 44 (сглаживание после cross-check, добавлено по
    итогам эмпирической проверки на 19-минутной записи высокого пульса):
    если недавняя история ПОДТВЕРЖДЁННЫХ окон ясно указывает на ~126 BPM,
    а текущее окно (cross_check САМ не смог подтвердить — verdict=
    "ambiguous") даёт ровно половину — сгладить до удвоенного значения."""
    print("\n=== Тест: сглаживание исправляет неподтверждённую субгармонику по истории (задача 44) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    try:
        for v in [124.0, 127.0, 126.0]:
            pipe._trusted_bpm_history.append(v)
        cross_check = _make_cross_check_result("ambiguous", 63.0, is_bpm_reliable=False)
        smoothed_bpm, was_smoothed, _hist_insuff = pipe._smooth_bpm_against_trusted_history(63.0, cross_check, entered_history=False)
        print(f"  smoothed_bpm={smoothed_bpm:.1f} was_smoothed={was_smoothed}")
        assert was_smoothed
        assert abs(smoothed_bpm - 126.0) < 1.0, f"ожидался ~126 (63*2), получено {smoothed_bpm}"
    finally:
        pipe.close()


def test_bpm_smoothing_requires_minimum_history_length():
    """Регрессия задачи 44/46: сглаживание не должно срабатывать, пока не
    накопилось хотя бы 2 подтверждённых значения (задача 46 снизила порог с
    3 до 2 — иначе на реальных данных "я обычное" ложное corrected_to_2f в
    самом начале записи, до накопления 3-й точки истории, оставалось
    неисправленным) — но одного-единственного значения всё ещё
    недостаточно: единственная точка не должна задавать опорную точку на
    весь ряд."""
    print("\n=== Тест: сглаживание не срабатывает при недостаточной истории (задача 44/46) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    try:
        pipe._trusted_bpm_history.append(126.0)
        cross_check = _make_cross_check_result("ambiguous", 63.0, is_bpm_reliable=False)
        smoothed_bpm, was_smoothed, _hist_insuff = pipe._smooth_bpm_against_trusted_history(63.0, cross_check, entered_history=False)
        print(f"  smoothed_bpm={smoothed_bpm:.1f} was_smoothed={was_smoothed}")
        assert not was_smoothed
        assert smoothed_bpm == 63.0
    finally:
        pipe.close()


def test_bpm_smoothing_does_not_trigger_on_resting_range():
    """Регрессия задачи 44 (найдено эмпирически на записи покоя): если
    история подтверждённых значений и текущий bpm оба лежат в
    физиологическом диапазоне покоя (~62 BPM, ratio~1.0), сглаживание НЕ
    должно срабатывать — иначе именно оно подняло бы верные 62 до 124,
    ровно то, что задача 44 явно запрещает."""
    print("\n=== Тест: сглаживание не трогает согласованные окна покоя (задача 44) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    try:
        for v in [62.0, 63.0, 61.5, 62.5]:
            pipe._trusted_bpm_history.append(v)
        cross_check = _make_cross_check_result("ambiguous", 62.1, is_bpm_reliable=False)
        smoothed_bpm, was_smoothed, _hist_insuff = pipe._smooth_bpm_against_trusted_history(62.1, cross_check, entered_history=False)
        print(f"  smoothed_bpm={smoothed_bpm:.1f} was_smoothed={was_smoothed}")
        assert not was_smoothed
        assert smoothed_bpm == 62.1
    finally:
        pipe.close()


def test_bpm_smoothing_skips_already_corrected_windows_but_updates_history():
    """Регрессия задачи 46/47: с ПУСТОЙ историей (< history_min_windows)
    сверять ещё не с чем — ЛЮБОЙ verdict (в т.ч. "corrected_to_2f") входит
    как есть, но обязан добавиться в историю подтверждённых значений
    (иначе история никогда бы не начала расти)."""
    print("\n=== Тест: пустая история не пересматривает ни один verdict, но растёт (задача 44/47) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    try:
        assert len(pipe._trusted_bpm_history) == 0
        cross_check = _make_cross_check_result("corrected_to_2f", 126.0, is_bpm_reliable=True)
        smoothed_bpm, was_smoothed, _hist_insuff = pipe._smooth_bpm_against_trusted_history(126.0, cross_check, entered_history=True)
        print(f"  smoothed_bpm={smoothed_bpm:.1f} was_smoothed={was_smoothed} "
              f"history={list(pipe._trusted_bpm_history)}")
        assert not was_smoothed
        assert smoothed_bpm == 126.0
        assert list(pipe._trusted_bpm_history) == [126.0]
    finally:
        pipe.close()


def test_bpm_smoothing_reconsiders_agreement_that_contradicts_established_history():
    """Регрессия задачи 47 (доказанная на реальных данных "я пульс 125"):
    "agreement" (оба независимых оценщика — спектр и пики — сошлись на f
    БЕЗ коррекции) раньше считалось золотым сигналом и никогда не
    пересматривалось историей. Это оказалось неверно: спектральная оценка
    и счётчик пиков смотрят на ОДИН и тот же сигнал и могут ОДНОВРЕМЕННО
    соскользнуть на одну и ту же гармонику — "agreement" на вдвое заниженном
    значении на фоне уже твёрдо установленной истории (~125, 4+
    подтверждённых окна) должно переворачиваться так же, как и любой
    другой verdict, а не слепо анкерить историю на неверном значении."""
    print("\n=== Тест: agreement пересматривается, если противоречит установленной истории (задача 47) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    try:
        for v in [125.2, 125.2, 126.1, 124.3]:
            pipe._trusted_bpm_history.append(v)
        cross_check = _make_cross_check_result("agreement", 63.0, is_bpm_reliable=True)
        smoothed_bpm, was_smoothed, _hist_insuff = pipe._smooth_bpm_against_trusted_history(63.0, cross_check, entered_history=True)
        print(f"  smoothed_bpm={smoothed_bpm:.1f} was_smoothed={was_smoothed}")
        assert was_smoothed, "agreement, противоречащий твёрдой истории вдвое, должен быть перевёрнут"
        assert abs(smoothed_bpm - 126.0) < 2.0, f"ожидался ~126 (63*2), получено {smoothed_bpm}"
    finally:
        pipe.close()


def test_apply_history_ratio_check_reports_insufficient_history():
    """Регрессия задачи 48: apply_history_ratio_check — общая функция между
    потоковым первым проходом (RPPGPipeline._smooth_bpm_against_trusted_history)
    и вторым (офлайн) проходом scripts/run_on_video.py. С историей короче
    min_windows решение не принимается вообще (history_was_insufficient=True,
    bpm не меняется) — ЭТО и есть флаг, по которому второй проход отбирает
    окна для пересмотра."""
    print("\n=== Тест: apply_history_ratio_check сообщает о недостаточной истории (задача 48) ===")
    from rppg.pipeline import apply_history_ratio_check

    bpm, was_smoothed, insufficient = apply_history_ratio_check(63.0, [125.0], min_windows=3)
    print(f"  bpm={bpm} was_smoothed={was_smoothed} insufficient={insufficient}")
    assert insufficient
    assert not was_smoothed
    assert bpm == 63.0


def test_apply_history_ratio_check_flips_against_full_history():
    """Регрессия задачи 48: с историей >= min_windows, ясно указывающей на
    вдвое большее значение, apply_history_ratio_check переворачивает bpm и
    сообщает history_was_insufficient=False — это ровно то решение, которое
    второй проход применяет задним числом к окнам первого прохода,
    которым не хватило истории (например, самое первое окно записи)."""
    print("\n=== Тест: apply_history_ratio_check переворачивает по полной истории (задача 48) ===")
    from rppg.pipeline import apply_history_ratio_check

    bpm, was_smoothed, insufficient = apply_history_ratio_check(
        67.5, [125.2, 126.1, 124.3, 127.8], min_windows=3,
    )
    print(f"  bpm={bpm:.1f} was_smoothed={was_smoothed} insufficient={insufficient}")
    assert not insufficient
    assert was_smoothed
    assert abs(bpm - 135.0) < 1.0, f"ожидался ~135 (67.5*2), получено {bpm}"


def test_apply_history_ratio_check_does_not_flip_consistent_value():
    """Регрессия задачи 48 (симметричная проверка — предотвращает ложные
    удвоения второго прохода на записи покоя): если значение УЖЕ близко к
    полной истории, apply_history_ratio_check не должен его трогать."""
    print("\n=== Тест: apply_history_ratio_check не трогает согласованное значение (задача 48) ===")
    from rppg.pipeline import apply_history_ratio_check

    bpm, was_smoothed, insufficient = apply_history_ratio_check(
        62.1, [63.0, 62.0, 64.8, 61.3], min_windows=3,
    )
    print(f"  bpm={bpm:.1f} was_smoothed={was_smoothed} insufficient={insufficient}")
    assert not insufficient
    assert not was_smoothed
    assert bpm == 62.1


def test_skin_mask_cached_per_step_not_recomputed_every_frame():
    """Регрессия задачи 34: маска кожи кэшируется и обновляется раз в
    step_seconds (по медианному кадру, накопленному с прошлого
    обновления), а НЕ пересчитывается на каждый кадр — покадровый
    пересчёт даёт ту же проблему "меняющегося состава выборки", что и
    медиана в задаче 31 (блик/шум/лёгкий поворот головы меняют бинарное
    решение "кожа/не кожа" по краю маски, никак не будучи связаны с
    пульсом). White-box тест напрямую против RPPGPipeline._refresh_skin_
    mask_cache, без реальных видеокадров/MediaPipe."""
    print("\n=== Тест: маска кожи кэшируется раз в шаг, не на каждый кадр (задача 34) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return

    try:
        h, w = 64, 64
        skin_bgr = np.array([40, 60, 100], dtype=np.int64)  # тот же оттенок, что и _make_pulsating_frame

        def make_frame(seed):
            r = np.random.default_rng(seed)
            noise = r.integers(-2, 3, size=(h, w, 3))
            return np.clip(skin_bgr[None, None, :] + noise, 0, 255).astype(np.uint8)

        step_ms = int(pipe.cfg.window.step_seconds * 1000)

        mask0 = pipe._refresh_skin_mask_cache(make_frame(1), 0)
        mask_mid = pipe._refresh_skin_mask_cache(make_frame(2), max(1, step_ms // 3))
        assert mask_mid is mask0, "маска не должна пересчитываться на кадре ДО истечения step_seconds"

        mask_next_step = pipe._refresh_skin_mask_cache(make_frame(3), step_ms + 1)
        assert mask_next_step is not mask0, (
            "маска ДОЛЖНА пересчитаться после истечения step_seconds с прошлого обновления"
        )
        print(f"  [OK] кэш маски не менялся внутри шага ({step_ms}мс) и обновился сразу после")

        # Резкая смена состава (skin -> полностью чёрный кадр) на границе следующего
        # шага должна выставить предупреждение (см. QualityConfig/ROIConfig.
        # max_skin_mask_pass_fraction_change).
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        pipe._refresh_skin_mask_cache(black_frame, 2 * step_ms + 1)
        assert pipe._pending_skin_mask_warning is not None, (
            "резкая смена доли skin-mask (кожа -> чёрный кадр) должна выставлять предупреждение"
        )
        print(f"  [OK] предупреждение выставлено: {pipe._pending_skin_mask_warning}")
    finally:
        pipe.close()


def test_aggregate_median_robust_to_single_outlier():
    """Регрессия для задачи 25: bpm_aggregate — ВЗВЕШЕННАЯ МЕДИАНА, не
    среднее. Одна октавная ошибка (150 среди восьмидесятых) не должна
    утягивать агрегат за собой — ровно то требование, ради которого медиана
    и выбрана вместо среднего (среднее сдвинулось бы примерно на 13 BPM).

    ВАЖНО (проверено численно): межквартильный размах (IQR) при n=5 с ОДНИМ
    выбросом на самом краю сортировки СТРУКТУРНО не меняется — 25-й/75-й
    перцентиль (в т.ч. в numpy) попадают строго ВНУТРЬ кластера [84,84,85,85],
    минуя единственную крайнюю точку 150 (np.percentile([84,85,84,150,85],
    [25,75]) == np.percentile([84,85,84,85,84], [25,75]) == (84.0, 85.0)
    буквально). Поэтому "spread_bpm/quality заметно ниже" на ЭТОМ конкретном
    5-точечном примере не воспроизводится ни при какой корректной реализации
    IQR — это демонстрируется отдельно, на сценарии, где перцентильная
    интерполяция реально задевает выброс (см.
    test_aggregate_spread_and_quality_respond_to_dispersion)."""
    print("\n=== Тест: агрегат устойчив к одиночному выбросу через медиану (задача 25) ===")
    pipe_clean, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe_clean is None:
        print(f"  [SKIP] {skip_reason}")
        return
    pipe_outlier, _ = _make_pipeline_for_aggregate_test()

    try:
        clean_seq = [84.0, 85.0, 84.0, 85.0, 84.0]
        outlier_seq = [84.0, 85.0, 84.0, 150.0, 85.0]
        result_clean = result_outlier = None
        for i, (bpm_c, bpm_o) in enumerate(zip(clean_seq, outlier_seq)):
            result_clean = pipe_clean._compute_aggregate(i * 1000, bpm_c, confidence=0.8, is_reliable=True, window_status="ok")
            result_outlier = pipe_outlier._compute_aggregate(i * 1000, bpm_o, confidence=0.8, is_reliable=True, window_status="ok")

        print(f"  clean:   bpm={result_clean.bpm:.1f}, spread={result_clean.spread_bpm:.2f}, "
              f"quality={result_clean.quality:.3f}")
        print(f"  outlier: bpm={result_outlier.bpm:.1f}, spread={result_outlier.spread_bpm:.2f}, "
              f"quality={result_outlier.quality:.3f}")

        assert result_clean.status == "measured" and result_outlier.status == "measured"
        assert abs(result_outlier.bpm - 84.5) <= 1.5, (
            f"выброс 150 не должен утягивать медиану за собой, получено bpm={result_outlier.bpm}"
        )
        assert result_outlier.bpm < 90.0, "150 явно не должно попадать в итоговый агрегат"
        assert result_outlier.spread_bpm == result_clean.spread_bpm, (
            "sanity: при n=5 с одним крайним выбросом IQR структурно не должен меняться"
        )
    finally:
        pipe_clean.close()
        pipe_outlier.close()


def test_aggregate_spread_and_quality_respond_to_dispersion():
    """Дополнение к test_aggregate_median_robust_to_single_outlier (задача
    25): IQR ЧУВСТВИТЕЛЕН к выбросу, когда перцентильная интерполяция
    реально его задевает (n=4, где 75-й перцентиль интерполирует МЕЖДУ
    предпоследней и последней точками) — здесь spread_bpm заметно растёт, а
    quality заметно падает, именно так, как качественно ожидает задача 25."""
    print("\n=== Тест: spread_bpm/quality реагируют на реальный разброс (задача 25) ===")
    pipe_clean, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe_clean is None:
        print(f"  [SKIP] {skip_reason}")
        return
    pipe_dispersed, _ = _make_pipeline_for_aggregate_test()

    try:
        clean_seq = [80.0, 82.0, 80.0, 82.0]
        dispersed_seq = [80.0, 82.0, 80.0, 90.0]
        result_clean = result_dispersed = None
        for i, (bpm_c, bpm_d) in enumerate(zip(clean_seq, dispersed_seq)):
            result_clean = pipe_clean._compute_aggregate(i * 1000, bpm_c, confidence=0.8, is_reliable=True, window_status="ok")
            result_dispersed = pipe_dispersed._compute_aggregate(i * 1000, bpm_d, confidence=0.8, is_reliable=True, window_status="ok")

        print(f"  clean:      spread={result_clean.spread_bpm:.2f}, quality={result_clean.quality:.3f}, "
              f"status={result_clean.status}")
        print(f"  dispersed:  spread={result_dispersed.spread_bpm:.2f}, quality={result_dispersed.quality:.3f}, "
              f"status={result_dispersed.status}")

        assert result_clean.status == "measured" and result_dispersed.status == "measured"
        assert result_dispersed.spread_bpm > result_clean.spread_bpm, (
            "разброс с реальным выбросом должен давать заметно больший spread_bpm"
        )
        assert result_dispersed.quality < result_clean.quality, (
            "больший разброс должен давать заметно более низкий quality"
        )
    finally:
        pipe_clean.close()
        pipe_dispersed.close()


def test_aggregate_no_data_when_all_confidence_zero():
    """Регрессия для задачи 25: интервал целиком из окон с confidence==0
    (например, лицо не найдено) обязан дать bpm_aggregate=NaN,
    quality=0.0, status='no_data' — никогда не число."""
    print("\n=== Тест: агрегат -> no_data, когда confidence==0 весь интервал (задача 25) ===")
    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return

    try:
        result = None
        for i in range(5):
            result = pipe._compute_aggregate(i * 1000, float("nan"), confidence=0.0, is_reliable=False, window_status="no_signal")
        print(f"  bpm={result.bpm}, quality={result.quality}, status={result.status}")
        assert np.isnan(result.bpm), "bpm_aggregate должен быть NaN, когда данных нет"
        assert result.quality == 0.0
        assert result.status == "no_data"
        assert result.n_windows == 0
    finally:
        pipe.close()


def test_aggregate_stale_decays_then_becomes_no_data():
    """Регрессия для задачи 25: после серии measured-окон подать серию
    пустых интервалов (confidence=0) — status должен перейти в stale,
    quality монотонно убывать, а по истечении max_stale_seconds — в
    no_data. Проверяет, что удержание НЕ бесконечное."""
    print("\n=== Тест: stale монотонно затухает и переходит в no_data (задача 25) ===")
    from rppg.config import PipelineConfig

    cfg = PipelineConfig()
    cfg.aggregate.max_stale_seconds = 10.0  # короче — быстрее тест
    pipe, skip_reason = _make_pipeline_for_aggregate_test(cfg)
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return

    try:
        t = 0
        result = None
        for bpm in [80.0, 81.0, 80.0, 81.0]:
            result = pipe._compute_aggregate(t, bpm, confidence=0.8, is_reliable=True, window_status="ok")
            t += 1000
        assert result.status == "measured", f"sanity: должно стать measured, получено {result.status}"
        measured_quality = result.quality

        # Шаг 2с (шире aggregate_span_seconds=5с по умолчанию) — каждый шаг
        # гарантированно вымывает предыдущие measured-окна из истории.
        step_ms = 2000
        n_steps = int(cfg.aggregate.max_stale_seconds * 1000 / step_ms) + 3
        statuses, qualities = [], []
        for _ in range(n_steps):
            t += step_ms
            result = pipe._compute_aggregate(t, float("nan"), confidence=0.0, is_reliable=False, window_status="no_signal")
            statuses.append(result.status)
            qualities.append(result.quality)

        print(f"  measured quality={measured_quality:.3f}")
        print(f"  statuses:  {statuses}")
        print(f"  qualities: {[round(q, 3) for q in qualities]}")

        assert "stale" in statuses, "должно появиться хотя бы одно stale-окно после measured"
        assert statuses[-1] == "no_data", "удержание НЕ должно быть бесконечным — обязан наступить no_data"

        stale_qualities = [q for q, s in zip(qualities, statuses) if s == "stale"]
        assert stale_qualities, "sanity: должна быть хотя бы одна stale-точка"
        assert stale_qualities[0] <= measured_quality
        assert all(a >= b for a, b in zip(stale_qualities, stale_qualities[1:])), (
            f"quality в stale-состоянии должно монотонно убывать, получено {stale_qualities}"
        )
    finally:
        pipe.close()


def test_aggregate_nan_iff_no_data():
    """Регрессия для задачи 25: not np.isnan(bpm_aggregate) <=> status !=
    'no_data' — инвариант обязан выполняться на КАЖДОМ окне, включая самое
    первое (когда истории агрегата ещё нет вообще)."""
    print("\n=== Тест: bpm_aggregate NaN <=> status=='no_data' (задача 25) ===")
    try:
        from rppg.face.landmarker import FaceFrameResult, HeadPose
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    pipe, skip_reason = _make_pipeline_for_aggregate_test()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return

    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )
    pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result

    rng = np.random.default_rng(30)
    fps = 30.0
    n_checked = 0
    try:
        for i in range(int(fps * 20.0)):
            t_sec = i / fps
            frame = _make_pulsating_frame(t_sec, 72.0, rng, h=480, w=640)
            result = pipe.process_frame(frame, int(t_sec * 1000))
            if result is None:
                continue
            n_checked += 1
            assert result.bpm_aggregate_status in ("measured", "stale", "no_data"), (
                f"неизвестный bpm_aggregate_status={result.bpm_aggregate_status!r}"
            )
            assert np.isnan(result.bpm_aggregate) == (result.bpm_aggregate_status == "no_data"), (
                f"инвариант нарушен: bpm_aggregate={result.bpm_aggregate}, "
                f"status={result.bpm_aggregate_status}"
            )
    finally:
        pipe.close()

    assert n_checked > 0, "sanity: должно быть хотя бы одно окно за 20с"
    print(f"  [OK] инвариант выполнен на всех {n_checked} окнах")


def test_aggregate_matches_across_csv_and_jsonl():
    """Регрессия для задачи 25 (тот же регрессионный подход, что и в
    задаче 16): bpm_aggregate* в PTSDPulseFeatures (то, что превращается в
    колонки CSV в run_on_video.py) и в WindowLogRecord (JSONL) должны
    совпадать поокновно — единственный источник (RPPGPipeline.
    _compute_aggregate), а не два независимо заполняемых пути."""
    print("\n=== Тест: bpm_aggregate* совпадает между PTSDPulseFeatures и JSONL-логом (задача 25) ===")
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

        landmarks = _make_synthetic_landmarks_for_pipeline_test()
        synthetic_result = FaceFrameResult(
            detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
        )
        pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result

        rng = np.random.default_rng(31)
        fps = 30.0
        results = []
        try:
            for i in range(int(fps * 15.0)):
                t_sec = i / fps
                frame = _make_pulsating_frame(t_sec, 75.0, rng)
                result = pipe.process_frame(frame, int(t_sec * 1000))
                if result is not None:
                    results.append(result)
        finally:
            pipe.close()

        assert log_path.exists()
        from rppg.structured_log import read_jsonl_records
        _headers, logged = read_jsonl_records(log_path)
        assert len(results) == len(logged)

        for i, (result, record) in enumerate(zip(results, logged)):
            assert result.bpm_aggregate_status == record["bpm_aggregate_status"], (
                f"окно #{i}: bpm_aggregate_status разошёлся ({result.bpm_aggregate_status} vs "
                f"{record['bpm_aggregate_status']})"
            )
            assert result.bpm_aggregate_n_windows == record["bpm_aggregate_n_windows"], f"окно #{i}: n_windows разошёлся"
            if np.isnan(result.bpm_aggregate):
                assert record["bpm_aggregate"] is None or np.isnan(record["bpm_aggregate"])
            else:
                assert abs(result.bpm_aggregate - record["bpm_aggregate"]) < 1e-6, f"окно #{i}: bpm_aggregate разошёлся"
            assert abs(result.bpm_aggregate_quality - record["bpm_aggregate_quality"]) < 1e-6, (
                f"окно #{i}: bpm_aggregate_quality разошёлся"
            )

        print(f"  [OK] {len(results)} окон, bpm_aggregate* совпадает поэлементно между CSV-источником и JSONL")


def test_rerun_on_same_log_path_overwrites_not_appends():
    """Регрессия для задачи 26 (доказанный баг): повторный прогон на тот
    же --log путь БЕЗ --append должен дать файл с ТЕМ ЖЕ числом строк
    окон, что и один прогон — не удвоенное. Раньше StructuredWindowLogger
    всегда открывал файл в режиме "a" (дозапись), и повторный запуск молча
    склеивал записи (найдено на реальном veronika_session.jsonl: 204
    строки при 68 окнах за прогон — ровно 3 склеенных прогона)."""
    print("\n=== Тест: повторный прогон перезаписывает лог, а не дозаписывает (задача 26) ===")
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
        from rppg.face.landmarker import FaceFrameResult, HeadPose
        from rppg.structured_log import read_jsonl_records
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    import tempfile
    from pathlib import Path

    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )
    fps = 30.0

    def _run(log_path: Path) -> int:
        cfg = PipelineConfig()
        try:
            pipe = RPPGPipeline(cfg, log_path=log_path)
        except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
            print(f"  [SKIP] не удалось инициализировать MediaPipe ({exc})")
            return -1
        pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result
        rng = np.random.default_rng(40)
        try:
            for i in range(int(fps * 13.0)):
                t_sec = i / fps
                pipe.process_frame(_make_pulsating_frame(t_sec, 72.0, rng), int(t_sec * 1000))
        finally:
            pipe.close()
        _headers, records = read_jsonl_records(log_path)
        return len(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "rerun.jsonl"
        n_first = _run(log_path)
        if n_first < 0:
            return
        n_second = _run(log_path)  # ТОТ ЖЕ путь, второй раз, без --append

        print(f"  первый прогон: {n_first} строк окон, второй прогон (тот же путь): {n_second} строк окон")
        assert n_first > 0, "sanity: должно быть хотя бы одно окно за 13с"
        assert n_second == n_first, (
            f"повторный прогон на тот же --log путь должен ПЕРЕЗАПИСАТЬ файл ({n_first} строк), "
            f"а не дописать в него (получено {n_second}) — задача 26"
        )
    print("  [OK] повторный прогон не удваивает число строк")


def test_analyze_log_rejects_file_with_mixed_run_headers():
    """Регрессия для задачи 26, второй критерий приёмки: искусственно
    склеенный файл с ДВУМЯ разными заголовками прогона должен заставить
    analyze_log.load_log упасть с понятным сообщением, а не молча
    посчитать статистику по смеси данных от разных конфигураций."""
    print("\n=== Тест: analyze_log.py отказывается считать статистику по смешанным заголовкам (задача 26) ===")
    import json
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_log import load_log

    header_a = {"__run_header__": True, "min_spectral_snr_db": -3.0, "git_hash": "aaa111"}
    header_b = {"__run_header__": True, "min_spectral_snr_db": -6.0, "git_hash": "bbb222"}
    fake_record = {
        "timestamp_ms": 0, "bpm": 72.0, "publishable": True, "sqi_overall_score": 0.6,
        "sqi_spectral_snr_db": 1.0, "sqi_cross_roi_agreement": 0.8, "sqi_landmark_stability": 0.9,
        "sqi_temporal_consistency": 0.8, "sqi_harmonic_score": 1.0, "sqi_flicker_suspected": False,
        "warnings": [], "window_status": "ok",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        mixed_path = Path(tmpdir) / "mixed.jsonl"
        with open(mixed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header_a, ensure_ascii=False) + "\n")
            f.write(json.dumps(fake_record, ensure_ascii=False) + "\n")
            f.write(json.dumps(header_b, ensure_ascii=False) + "\n")  # склейка второго прогона, другой конфиг
            f.write(json.dumps(fake_record, ensure_ascii=False) + "\n")

        raised = False
        message = ""
        try:
            load_log(mixed_path)
        except SystemExit as exc:
            raised = True
            message = str(exc)

        print(f"  SystemExit поднят: {raised}")
        if raised:
            print(f"  сообщение: {message[:150]}...")
        assert raised, "load_log должен упасть на файле с двумя разными заголовками, а не посчитать смесь"
        assert "конфигураций" in message.lower() or "заголовк" in message.lower(), (
            f"сообщение об ошибке должно объяснять причину (разные заголовки), получено: {message!r}"
        )
    print("  [OK] analyze_log.py явно отказался считать статистику по смешанным заголовкам")


def test_ranking_preserved():
    """Регрессия для задачи 23: качественный порядок 6 реальных видео
    (расширено с исходных 4 — добавлены "мужчина" и "вероника", см.
    scripts/run_benchmark.py), найденный при анализе на min_spectral_snr_db=
    -3.0 (задача 22) — готовый регрессионный тест против "починки" SQI
    простым ослаблением порогов.

    Два инварианта:
    1. "девочка" (движение в кадре) — ХУДШАЯ по покрытию среди ВСЕХ 6 видео.
    2. ЧЁТКИЙ РАЗРЫВ на две группы: "хорошая" (тётя, темнокожая
       вертикально/горизонтально — 26-51% покрытия на момент калибровки) и
       "проблемная" (вероника, мужчина, девочка — 0-19%) — минимум покрытия
       хорошей группы должен оставаться СТРОГО ВЫШЕ максимума проблемной.
       Если разрыв схлопнулся — компонента перестала различать хороший и
       плохой сигнал, а не просто "числа немного сдвинулись".

    Пропускается, если видео недоступны локально (они не часть репозитория —
    см. scripts/run_benchmark.py.DEFAULT_VIDEOS) — это не CI-тест, а
    инструмент разработчика с доступом к исходным записям."""
    print("\n=== Тест: качественный порядок 6 видео сохраняется (задача 23) ===")
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from run_benchmark import DEFAULT_VIDEOS, run_video

    videos = [Path(v) for v in DEFAULT_VIDEOS]
    missing = [v for v in videos if not v.exists()]
    if missing:
        print(f"  [SKIP] видео недоступны локально ({len(missing)}/{len(videos)} не найдено) — "
              "этот тест не часть CI, требует исходных записей задач 16-24")
        return

    try:
        from rppg.config import PipelineConfig
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        print(f"  [SKIP] MediaPipe недоступен в этой среде ({exc})")
        return

    model_path = str(Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task")
    if not Path(model_path).exists():
        print(f"  [SKIP] модель Face Landmarker недоступна ({model_path})")
        return

    coverage_by_video: dict[str, float] = {}
    for video in videos:
        summary = run_video(str(video), model_path, PipelineConfig())
        coverage_by_video[video.stem] = summary["coverage"]
        print(f"  {video.stem:<45s} coverage={summary['coverage'] * 100:5.1f}%")

    def _find(substrings: tuple[str, ...]) -> str:
        return next(
            name for name in coverage_by_video
            if any(s in name.lower() for s in substrings)
        )

    girl_name = _find(("девочк",))
    good_group = [_find(("тет",)), _find(("вертикально",)), _find(("горизонтально",))]
    problem_group = [_find(("верони",)), _find(("мужчин",)), girl_name]

    assert coverage_by_video[girl_name] == min(coverage_by_video.values()), (
        f"'{girl_name}' (движение) должна оставаться ХУДШЕЙ по покрытию среди всех 6 видео, "
        f"получено: {coverage_by_video}"
    )

    min_good = min(coverage_by_video[n] for n in good_group)
    max_problem = max(coverage_by_video[n] for n in problem_group)
    assert min_good > max_problem, (
        "разрыв между 'хорошей' (тётя/темнокожая) и 'проблемной' (вероника/мужчина/девочка) "
        f"группами схлопнулся: min(хорошая)={min_good:.3f} <= max(проблемная)={max_problem:.3f}, "
        f"получено: {coverage_by_video}"
    )
    print(f"  [OK] 'девочка' худшая среди 6; разрыв хорошая/проблемная группы сохранён "
          f"(min={min_good:.1%} > max={max_problem:.1%})")


def run_all():
    tests = [
        test_color_based_methods_recover_known_bpm,
        test_chrom_alpha_robust_to_linear_drift,
        test_pos_numba_matches_numpy_reference,
        test_head_motion_method_recovers_known_bpm,
        test_frequency_estimators_agree_on_clean_tone,
        test_harmonic_sum_avoids_octave_error_that_confuses_plain_argmax,
        test_harmonic_sum_not_biased_by_unavailable_high_harmonics,
        test_octave_error_detector_finds_errors_on_drifting_true_signal,
        test_continuity_check_prefers_peak_near_recent_history_within_tolerance,
        test_physiological_rate_limit_blocks_implausible_jump_without_power_advantage,
        test_lombscargle_handles_irregular_sampling,
        test_bandpass_rejects_out_of_band_and_keeps_in_band,
        test_bandpass_sos_retains_amplitude_at_low_normalized_cutoff,
        test_roi_mean_rgb_truncated_mean_recovers_subquantization_signal,
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
        test_temporal_consistency_does_not_penalize_spectral_grid_jitter,
        test_resolvable_bpm_difference_reflects_main_lobe_width,
        test_cross_roi_agreement_robust_to_single_outlier_roi,
        test_cross_roi_agreement_ignores_low_snr_roi,
        test_cross_roi_agreement_guarantees_minimum_two_participants,
        test_cross_roi_agreement_single_source_is_insufficient_in_principle,
        test_harmonic_check_flags_fundamental_vs_second_harmonic_confusion,
        test_harmonic_check_ignores_1_over_f_noise_floor,
        test_second_harmonic_energy_is_not_penalized,
        test_illumination_flicker_detected_via_background_roi,
        test_flicker_requires_consecutive_windows_to_confirm,
        test_static_photo_or_mannequin_is_not_publishable,
        test_occluded_face_pipeline_never_publishes,
        test_end_to_end_pipeline_recovers_known_bpm_from_synthetic_video,
        test_auto_method_selection_avoids_specifically_corrupted_method,
        test_config_yaml_json_roundtrip_preserves_non_default_values,
        test_structured_window_logger_writes_valid_jsonl_with_sqi_components,
        test_publishable_matches_across_csv_and_jsonl,
        test_window_status_labels_degenerate_and_no_face_windows,
        test_subsample_peak_refinement_reduces_rmssd_quantization_error,
        test_ectopic_masking_does_not_stitch_false_diff,
        test_hrv_artifact_fraction_gates_publishable,
        test_cross_check_agreement_on_clean_signal,
        test_cross_check_corrects_subharmonic_spectral_estimate,
        test_cross_check_flags_unexplained_mismatch_as_ambiguous,
        test_cross_check_dicrotic_notch_suppressed_by_prominence,
        test_hrv_rejects_physiologically_impossible_rmssd_at_high_hr,
        test_bpm_smoothing_corrects_unconfirmed_subharmonic_using_trusted_history,
        test_bpm_smoothing_requires_minimum_history_length,
        test_bpm_smoothing_does_not_trigger_on_resting_range,
        test_bpm_smoothing_skips_already_corrected_windows_but_updates_history,
        test_bpm_smoothing_reconsiders_agreement_that_contradicts_established_history,
        test_apply_history_ratio_check_reports_insufficient_history,
        test_apply_history_ratio_check_flips_against_full_history,
        test_apply_history_ratio_check_does_not_flip_consistent_value,
        test_lf_hf_require_separate_minimum_durations,
        test_respiration_rate_from_amplitude_modulation,
        test_skin_mask_cached_per_step_not_recomputed_every_frame,
        test_aggregate_median_robust_to_single_outlier,
        test_aggregate_spread_and_quality_respond_to_dispersion,
        test_aggregate_no_data_when_all_confidence_zero,
        test_aggregate_stale_decays_then_becomes_no_data,
        test_aggregate_nan_iff_no_data,
        test_aggregate_matches_across_csv_and_jsonl,
        test_rerun_on_same_log_path_overwrites_not_appends,
        test_analyze_log_rejects_file_with_mixed_run_headers,
        test_ranking_preserved,
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
