"""
Тесты для signal/tracker.py (задача 12): стабильное отображаемое число,
полученное ТОЛЬКО из реальных измерений — без выдумывания (см. докстринг
модуля про удалённый BestEffortConfig, задача 8).

Запуск: PYTHONPATH=src python3 -m pytest tests/test_tracker.py -v
"""

from __future__ import annotations

import numpy as np

from rppg.signal.tracker import PulseTracker, TrackerOutput, _weighted_median


def test_weighted_median_basic():
    print("\n=== Тест: взвешенная медиана — базовые случаи ===")
    # Равные веса -> обычная медиана.
    assert _weighted_median([(70.0, 1.0), (72.0, 1.0), (74.0, 1.0)]) == 72.0
    # Один сильно перевешивающий элемент "тянет" медиану к себе.
    assert _weighted_median([(70.0, 0.01), (72.0, 0.01), (150.0, 10.0)]) == 150.0
    # Единственный элемент.
    assert _weighted_median([(80.0, 1.0)]) == 80.0


def test_tracker_returns_last_valid_when_all_windows_zero_weight():
    """Критерий приёмки задачи 12: если подать на вход только окна с
    confidence=0, трекер НЕ выдаёт нового числа, а возвращает возраст
    последнего реального измерения."""
    print("\n=== Тест: confidence=0 на всех окнах -> (last_valid_bpm, age) без выдумывания ===")
    # window_size=1: скользящий буфер держит ТОЛЬКО последний вызов, поэтому
    # is_fresh отражает вес ИМЕННО текущего окна, а не "память" о старом
    # валидном измерении, которое технически ещё не выпало из окна большего
    # размера (см. отдельный test_tracker_weight_zero_window_does_not_pollute_weighted_median
    # про то, что окно size>1 ЗАКОНОМЕРНО удерживает недавние валидные точки).
    tracker = PulseTracker(window_size=1)

    out0 = tracker.update(timestamp_s=0.0, bpm=72.0, confidence=0.8)
    assert out0.is_fresh and out0.bpm is not None
    print(f"  t=0.0: bpm={out0.bpm:.2f}, is_fresh={out0.is_fresh}")

    # Дальше — только окна с нулевой уверенностью (например, NaN bpm при
    # status="no_signal", см. задачу 11).
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        out = tracker.update(timestamp_s=t, bpm=float("nan"), confidence=0.0)
        assert not out.is_fresh, f"t={t}: окно с confidence=0 не должно давать is_fresh=True"
        assert out.bpm == out0.bpm, "bpm должен оставаться last_valid_bpm, а не меняться/обнуляться"
        assert out.age_seconds == t - 0.0, f"age_seconds должен расти вместе с реальным временем (t={t})"
        print(f"  t={t}: bpm={out.bpm:.2f} (заморожено), age_seconds={out.age_seconds:.1f}")


def test_tracker_returns_none_before_any_real_measurement():
    """Если реальных измерений не было НИКОГДА (не просто 'сейчас нулевой
    вес') — bpm=None и age_seconds=None, а не 0.0/выдуманное число."""
    print("\n=== Тест: до первого реального измерения — bpm=None, age=None ===")
    tracker = PulseTracker(window_size=5)
    out = tracker.update(timestamp_s=0.0, bpm=float("nan"), confidence=0.0)
    assert out.bpm is None
    assert out.age_seconds is None
    assert not out.is_fresh
    print(f"  bpm={out.bpm}, age_seconds={out.age_seconds}, is_fresh={out.is_fresh}")


def test_tracker_does_not_clip_to_old_best_effort_range():
    """Задача 12: никакого клиппинга в [55, 120] — только физиологическая
    полоса FilterConfig (по умолчанию 42-240 BPM). 140 BPM (тахикардия,
    ЗА пределами старого диапазона best_effort) должен приниматься как есть."""
    print("\n=== Тест: 140 BPM (тахикардия) НЕ обрезается старым диапазоном [55,120] ===")
    tracker = PulseTracker(window_size=1, max_change_per_step_bpm=None)  # без сглаживания/памяти окна, чтобы видеть сырое значение
    out = tracker.update(timestamp_s=0.0, bpm=140.0, confidence=0.9)
    assert out.is_fresh
    assert abs(out.bpm - 140.0) < 1e-9, f"140 BPM должен пройти как есть, получено {out.bpm}"
    print(f"  bpm={out.bpm:.1f} (принято без обрезки)")

    # А вот 300 BPM (вне физиологической полосы 42-240) — уже не валидное
    # измерение, вес 0.
    out_invalid = tracker.update(timestamp_s=1.0, bpm=300.0, confidence=0.9)
    assert not out_invalid.is_fresh, "300 BPM вне физиологической полосы -> нулевой вес, не новое отображаемое значение"
    print(f"  bpm=300 (вне полосы): is_fresh={out_invalid.is_fresh}, bpm остаётся {out_invalid.bpm}")


def test_tracker_max_change_per_step_smooths_display_not_measurement():
    """max_change_per_step_bpm — сглаживание ОТОБРАЖЕНИЯ поверх реальных
    измерений (не подмена их константой): резкий, но РЕАЛЬНЫЙ скачок
    (72 -> 120) сглаживается по шагам, но в итоге тracker сходится к
    измеренному значению, а не к чему-то постороннему."""
    print("\n=== Тест: max_change_per_step_bpm сглаживает отображение, не искажая источник ===")
    tracker = PulseTracker(window_size=1, max_change_per_step_bpm=5.0)
    tracker.update(timestamp_s=0.0, bpm=72.0, confidence=0.9)

    outputs = []
    for i in range(1, 15):
        out = tracker.update(timestamp_s=float(i), bpm=120.0, confidence=0.9)
        outputs.append(out.bpm)
    print(f"  bpm по шагам: {[round(v, 1) for v in outputs]}")

    assert outputs[0] == 77.0, "первый шаг должен сдвинуться ровно на max_change_per_step_bpm=5.0"
    assert all(b <= 120.0 + 1e-9 for b in outputs), "сглаженное значение не должно ПЕРЕПРЫГИВАТЬ измеренное"
    assert abs(outputs[-1] - 120.0) < 1e-6, "после достаточного числа шагов должно сойтись к реально измеренному 120.0"


def test_tracker_weight_zero_window_does_not_pollute_weighted_median():
    """Окно с confidence=0 должно иметь РОВНО нулевой вес в скользящем
    буфере — не участвовать во взвешенной медиане наравне с валидными, даже
    если оно физически осталось в буфере (окно "no_signal" между двумя
    реальными измерениями не должно тянуть медиану к NaN/произвольному
    значению)."""
    print("\n=== Тест: окно с confidence=0 не искажает взвешенную медиану соседних окон ===")
    tracker = PulseTracker(window_size=3, max_change_per_step_bpm=None)
    tracker.update(timestamp_s=0.0, bpm=70.0, confidence=1.0)
    tracker.update(timestamp_s=1.0, bpm=float("nan"), confidence=0.0)  # no_signal посередине
    out = tracker.update(timestamp_s=2.0, bpm=74.0, confidence=1.0)
    print(f"  bpm={out.bpm:.2f} (ожидание: медиана двух валидных 70 и 74 = 72, NaN-окно не в счёт)")
    assert abs(out.bpm - 72.0) < 1e-9


if __name__ == "__main__":
    test_weighted_median_basic()
    test_tracker_returns_last_valid_when_all_windows_zero_weight()
    test_tracker_returns_none_before_any_real_measurement()
    test_tracker_does_not_clip_to_old_best_effort_range()
    test_tracker_max_change_per_step_smooths_display_not_measurement()
    test_tracker_weight_zero_window_does_not_pollute_weighted_median()
    print("\nOK")
