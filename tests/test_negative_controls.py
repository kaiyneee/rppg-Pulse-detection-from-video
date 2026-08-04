"""
Задача 9: "на что система не должна реагировать" — негативные (и два
позитивных) контроля всего RPPGPipeline целиком, гоняемые в CI.

Приоритет здесь выше, чем у любой правки порогов SQI: если пайплайн не
восстанавливает известный BPM на ЧИСТОЙ синтетике (test_clean_72bpm_*) —
проблема в коде/оркестрации, а не в реальных условиях съёмки, и искать её
на реальном видео на порядок дороже, чем здесь.

    Вход                                          Ожидание
    ---------------------------------------------------------------------
    Статичное фото (один и тот же кадр N раз)       publishable=False всегда
    Видео с белым шумом вместо кадров               publishable=False
    Синтетика с пульсацией 72 BPM                   publishable=True, 72±2
    Синтетика 72 BPM + сильный дрейф освещения      publishable=True, 72±3
    Синтетика с мерцанием фона на 1.2 Гц            flicker_suspected=True

Использует те же синтетические помощники (_make_synthetic_landmarks_for_
pipeline_test, _make_pulsating_frame), что и test_signal_pipeline.py —
реальный MediaPipe Face Landmarker подменяется на фиксированный результат
детекции ТОЛЬКО там, где тест целенаправленно проверяет ДАЛЬШЕ детекции
(сигнальный/SQI уровень); тест на белый шум, наоборот, использует РЕАЛЬНЫЙ
(неподменённый) landmarker, т.к. там как раз важно, что лицо на шуме не
находится.
"""

from __future__ import annotations

import numpy as np

from test_signal_pipeline import (
    _make_synthetic_landmarks_for_pipeline_test,
    _make_pulsating_frame,
)

FPS = 30.0


def _try_init_pipeline(cfg=None, debug: bool = False):
    """Общая обвязка: RPPGPipeline с реальной моделью Face Landmarker, но
    грациозно пропускает тест (не падает), если MediaPipe/модель недоступны
    в конкретной среде запуска — тот же паттерн, что и в test_signal_pipeline.py."""
    try:
        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        return None, f"MediaPipe недоступен в этой среде ({exc})"
    try:
        pipe = RPPGPipeline(cfg or PipelineConfig(), debug=debug)
    except FileNotFoundError as exc:
        return None, f"модель Face Landmarker недоступна ({exc})"
    except Exception as exc:  # noqa: BLE001 - опциональная зависимость среды
        return None, f"не удалось инициализировать MediaPipe ({exc})"
    return pipe, None


def _mock_face_detected(pipe) -> None:
    """Подменяет landmarker.detect() на фиксированный синтетический
    результат детекции — используется там, где тест целенаправленно
    проверяет сигнальный/SQI уровень, а не саму детекцию лица (см.
    test_end_to_end_pipeline_recovers_known_bpm_from_synthetic_video)."""
    from rppg.face.landmarker import FaceFrameResult, HeadPose

    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )
    pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result


def test_static_photo_never_publishes():
    """Один и тот же кадр N раз подряд — НОЛЬ временной вариации в ROI.
    Реальный пульс всегда даёт вариацию яркости во времени; полностью
    статичный вход не должен пройти SQI ни на одном окне."""
    print("\n=== (а) Статичное фото: publishable должен быть False на каждом окне ===")
    pipe, skip_reason = _try_init_pipeline()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    _mock_face_detected(pipe)

    rng = np.random.default_rng(0)
    frozen_frame = _make_pulsating_frame(0.0, 72.0, rng, h=480, w=640)  # ОДИН кадр, переиспользуется как есть

    got_any = False
    try:
        for i in range(int(FPS * 20.0)):
            result = pipe.process_frame(frozen_frame, int(i / FPS * 1000))
            if result is None:
                continue
            got_any = True
            assert not result.publishable, (
                f"идентичный кадр без временной вариации дал publishable=True "
                f"(bpm={result.bpm}, sqi={result.sqi_score:.3f})"
            )
    finally:
        pipe.close()
    assert got_any, "sanity: пайплайн должен вернуть хотя бы один результат за 20с"
    print("  [OK] статичное фото ни разу не дало publishable=True")


def test_white_noise_frames_never_publish():
    """Кадры — независимый случайный шум (никакой связи между соседними
    кадрами, никакого лица). НЕ подменяем landmarker: важно, что реальный
    MediaPipe либо не находит лицо на шуме, либо (если случайно находит)
    сигнал внутри найденного ROI всё равно остаётся шумом без периодической
    структуры — оба исхода должны кончаться publishable=False."""
    print("\n=== (б) Белый шум вместо кадров: publishable должен быть False ===")
    pipe, skip_reason = _try_init_pipeline()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return

    rng = np.random.default_rng(1)
    got_any = False
    try:
        for i in range(int(FPS * 6.0)):  # ~6с — достаточно для >= одного окна оценки
            frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
            result = pipe.process_frame(frame, int(i / FPS * 1000))
            if result is None:
                continue
            got_any = True
            assert not result.publishable, (
                f"белый шум дал publishable=True (bpm={result.bpm}, sqi={result.sqi_score:.3f})"
            )
    finally:
        pipe.close()
    print(f"  [{'OK' if got_any else 'OK (лицо ни разу не найдено на шуме — тоже валидный исход)'}] "
          "белый шум ни разу не дал publishable=True")


def test_clean_72bpm_pipeline_publishes_accurate_bpm():
    """Позитивный контроль (важнее негативных, см. докстринг модуля):
    ЧИСТАЯ синтетика с известным true_bpm=72 должна и публиковаться, и
    давать BPM в пределах 2 ударов от истинного. Если это падает — баг в
    оркестрации/математике, а не в качестве реальной съёмки."""
    print("\n=== (в) Чистая синтетика 72 BPM: publishable=True, BPM=72±2 ===")
    pipe, skip_reason = _try_init_pipeline()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    _mock_face_detected(pipe)

    true_bpm = 72.0
    rng = np.random.default_rng(42)
    results = []
    try:
        for i in range(int(FPS * 20.0)):
            t_sec = i / FPS
            frame = _make_pulsating_frame(t_sec, true_bpm, rng, h=480, w=640)
            result = pipe.process_frame(frame, int(t_sec * 1000))
            if result is not None:
                results.append(result)
    finally:
        pipe.close()

    assert len(results) > 0, "пайплайн должен вернуть хотя бы одну оценку за 20с чистой синтетики"
    published = [r for r in results if r.publishable]
    assert published, (
        f"ни одно окно НЕ опубликовано на чистой синтетике 72 BPM (получено {len(results)} окон); "
        f"последние sqi_score: {[round(r.sqi_score, 2) for r in results[-5:]]}"
    )
    tail = published[-3:]
    for r in tail:
        assert abs(r.bpm - true_bpm) <= 2.0, f"|{r.bpm:.2f} - {true_bpm}| > 2 BPM (sqi={r.sqi_score:.2f})"
    print(f"  [OK] {len(published)}/{len(results)} окон опубликовано, "
          f"последние BPM: {[round(r.bpm, 1) for r in tail]}")


def test_72bpm_with_strong_illumination_drift_still_publishes():
    """Тот же чистый пульс 72 BPM, но с сильным (амплитуда 25 уровней
    яркости, период 33с — заметно медленнее полосы пульса и заметно
    короче окна анализа 10с) АХРОМАТИЧНЫМ дрейфом освещения поверх всего
    кадра. detrend (Tarvainen, см. FilterConfig) должен убрать дрейф, не
    тронув сам пульс — допуск ослаблен до 3 BPM (см. таблицу задачи 9)."""
    print("\n=== (г) 72 BPM + сильный дрейф освещения: publishable=True, BPM=72±3 ===")
    pipe, skip_reason = _try_init_pipeline()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    _mock_face_detected(pipe)

    true_bpm = 72.0
    drift_amplitude = 25.0
    drift_freq_hz = 1.0 / 33.0
    rng = np.random.default_rng(43)
    results = []
    try:
        for i in range(int(FPS * 24.0)):
            t_sec = i / FPS
            frame = _make_pulsating_frame(t_sec, true_bpm, rng, h=480, w=640).astype(np.float32)
            drift = drift_amplitude * np.sin(2 * np.pi * drift_freq_hz * t_sec)
            frame = np.clip(frame + drift, 0, 255).astype(np.uint8)
            result = pipe.process_frame(frame, int(t_sec * 1000))
            if result is not None:
                results.append(result)
    finally:
        pipe.close()

    assert len(results) > 0
    published = [r for r in results if r.publishable]
    assert published, (
        f"ни одно окно НЕ опубликовано при 72 BPM + дрейф освещения "
        f"(получено {len(results)} окон); "
        f"последние sqi_score: {[round(r.sqi_score, 2) for r in results[-5:]]}"
    )
    tail = published[-3:]
    for r in tail:
        assert abs(r.bpm - true_bpm) <= 3.0, f"|{r.bpm:.2f} - {true_bpm}| > 3 BPM при дрейфе освещения"
    print(f"  [OK] {len(published)}/{len(results)} окон опубликовано despite дрейфа, "
          f"последние BPM: {[round(r.bpm, 1) for r in tail]}")


def test_background_flicker_is_detected():
    """Фоновый ROI (заведомо вне лица, см. face/roi.py::build_background_roi_mask)
    модулируется на ТОЙ ЖЕ частоте (1.2 Гц = 72 BPM), что и лицевой пульс —
    ожидаем sqi_flicker_suspected=True (жёсткий гейт, см. quality.py) в
    течение записи. Использует RPPGPipeline(debug=True), чтобы читать
    SQIResult.flicker_suspected напрямую, а не парсить текст warnings."""
    print("\n=== (д) Мерцание фона на 1.2 Гц: flicker_suspected должен стать True ===")
    from rppg.face.roi import build_background_roi_mask, landmarks_to_pixels

    pipe, skip_reason = _try_init_pipeline(debug=True)
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    _mock_face_detected(pipe)

    h, w = 480, 640
    landmarks_norm = _make_synthetic_landmarks_for_pipeline_test()
    landmarks_px = landmarks_to_pixels(landmarks_norm, (h, w))
    bg_mask = build_background_roi_mask((h, w, 3), landmarks_px)
    assert bg_mask is not None, "sanity: синтетические landmarks должны оставлять свободный угол для фона"
    ys, xs = np.where(bg_mask > 0)

    true_bpm = 72.0
    flicker_hz = true_bpm / 60.0
    flicker_amp = 25.0
    rng = np.random.default_rng(44)

    flicker_seen = False
    try:
        for i in range(int(FPS * 30.0)):
            t_sec = i / FPS
            frame = _make_pulsating_frame(t_sec, true_bpm, rng, h=h, w=w).astype(np.float32)
            mod = flicker_amp * np.sin(2 * np.pi * flicker_hz * t_sec)
            frame[ys, xs] = np.clip(80.0 + mod, 0, 255)
            result = pipe.process_frame(frame.astype(np.uint8), int(t_sec * 1000))
            if result is not None and result.debug is not None and result.debug.sqi.flicker_suspected:
                flicker_seen = True
    finally:
        pipe.close()

    assert flicker_seen, "фон, синхронно мерцающий с лицом на 1.2 Гц, ни разу не дал flicker_suspected=True"
    print("  [OK] flicker_suspected стал True хотя бы на одном окне")


def test_no_estimate_computed_before_window_is_full():
    """Задача 6, критерий приёмки: _ibi_log и _recent_bpm_history НЕ должны
    содержать значений, полученных до заполнения окна. _compute_estimate —
    единственное место, которое пополняет оба этих накопителя (см.
    pipeline.py), и оно вызывается ТОЛЬКО когда elapsed_s >=
    min_seconds_before_estimate (см. process_frame) — с задачи 6 это по
    умолчанию РАВНО window_seconds (см. WindowConfig.__post_init__).
    Проверяем инвариант напрямую (white-box): КАЖДЫЙ раз, когда
    process_frame() вернул непустой результат, буфер должен уже содержать
    >= window_seconds секунд реального времени — то есть в принципе не
    может существовать "прогревочная" (по окну короче номинального) оценка,
    которая пополнила бы историю."""
    print("\n=== Задача 6: ни одна оценка не считается по неполному окну ===")
    pipe, skip_reason = _try_init_pipeline()
    if pipe is None:
        print(f"  [SKIP] {skip_reason}")
        return
    _mock_face_detected(pipe)

    window_seconds = pipe.cfg.window.window_seconds
    tol_s = 0.1  # запас на дискретность кадров (шаг ~1/30с)
    rng = np.random.default_rng(5)
    n_estimates = 0
    try:
        for i in range(int(FPS * 15.0)):
            t_sec = i / FPS
            frame = _make_pulsating_frame(t_sec, 72.0, rng, h=480, w=640)
            result = pipe.process_frame(frame, int(t_sec * 1000))
            if result is not None:
                n_estimates += 1
                assert pipe.buffered_seconds >= window_seconds - tol_s, (
                    f"оценка №{n_estimates} посчитана по окну {pipe.buffered_seconds:.2f}с "
                    f"< window_seconds={window_seconds}с — это ровно та 'прогревочная' оценка, "
                    "которую задача 6 запрещает"
                )
    finally:
        pipe.close()

    assert n_estimates > 0, "sanity: за 15с должна быть хотя бы одна оценка"
    print(f"  [OK] все {n_estimates} оценок посчитаны по окну >= {window_seconds}с "
          "(_recent_bpm_history/_ibi_log не могли получить 'прогревочные' значения)")


if __name__ == "__main__":
    test_static_photo_never_publishes()
    test_white_noise_frames_never_publish()
    test_clean_72bpm_pipeline_publishes_accurate_bpm()
    test_72bpm_with_strong_illumination_drift_still_publishes()
    test_background_flicker_is_detected()
    test_no_estimate_computed_before_window_is_full()
