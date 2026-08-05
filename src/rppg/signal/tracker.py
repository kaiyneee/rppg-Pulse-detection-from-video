"""
Задача 12: трекер поверх пооконных измерений — стабильное отображаемое
число, полученное ТОЛЬКО из реальных измерений.

Контекст: BestEffortConfig/best_effort_bpm (задача 8, теперь удалён)
пытался решить ту же практическую задачу ("нужно стабильное число на
экране/в интеграции, а не дрожащее пооконное значение") ВЫДУМЫВАНИЕМ —
fallback_bpm=75 при полном отсутствии сигнала, клиппинг в произвольный
"правдоподобный" диапазон [55, 120], который к тому же обрезал ровно ту
тахикардию, ради которой FilterConfig расширил полосу до 4.0 Гц.

PulseTracker решает ТУ ЖЕ задачу честно:
  - взвешенная по confidence медиана последних window_size окон — вес
    окна с confidence=0 РАВЕН НУЛЮ (окно просто не участвует), а не
    заменяется константой;
  - если ВСЕ окна в буфере имеют нулевой вес — возвращается
    (last_valid_bpm, age_seconds) — последнее РЕАЛЬНО измеренное значение
    и его возраст, а не выдуманное число;
  - единственное ограничение диапазона — физиологическая полоса, уже
    заданная FilterConfig (по умолчанию 42-240 BPM), а не отдельный более
    узкий "правдоподобный" диапазон;
  - необязательный max_change_per_step_bpm — ЧЕСТНО сглаживание
    ОТОБРАЖЕНИЯ (см. докстринг PulseTracker), применяется к УЖЕ ИЗМЕРЕННЫМ
    значениям, а не к выдуманной константе.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerOutput:
    """Результат одного PulseTracker.update().

    bpm: отображаемое значение. None ТОЛЬКО если ни одного реального
    измерения не было НИКОГДА (буфер пуст с самого начала, а не просто
    "сейчас нулевой вес").
    is_fresh: True, если ТЕКУЩЕЕ окно (и/или недавние окна в буфере) реально
    внесли вклад (ненулевой суммарный вес); False, если весь буфер сейчас
    нулевого веса и bpm — это last_valid_bpm, "замороженный" с прошлого
    реального измерения.
    age_seconds: возраст last_valid_bpm в секундах. 0.0, если is_fresh=True
    (последнее измерение — прямо сейчас). None, если реальных измерений не
    было вообще (bpm тоже None в этом случае).
    """

    bpm: float | None
    is_fresh: bool
    age_seconds: float | None


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Взвешенная медиана: значение, у которого сумма весов элементов по
    обе стороны не превышает половины суммарного веса. Вырождается в
    обычную медиану при равных весах — В ТОМ ЧИСЛЕ для чётного числа
    элементов (например, ровно два равных по весу измерения 70 и 74 должны
    дать 72, а не "нижний" элемент 70) — если накопленный вес попадает
    РОВНО на половину, берётся среднее текущего и следующего значения."""
    if len(pairs) == 1:
        return pairs[0][0]
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    half = total / 2.0
    cum = 0.0
    for i, (value, weight) in enumerate(ordered):
        cum += weight
        if cum > half:
            return value
        if cum == half:
            if i + 1 < len(ordered):
                return (value + ordered[i + 1][0]) / 2.0
            return value
    return ordered[-1][0]  # sanity fallback (не должен достигаться при total > 0)


class PulseTracker:
    """Взвешенная по confidence скользящая медиана последних window_size
    окон + опциональный слew-rate limiter отображаемого значения.

    Использование — один объект на живую сессию (аналог RPPGPipeline),
    один вызов update() на каждый PTSDPulseFeatures (задача 11: bpm/
    confidence оттуда передаются как есть, без промежуточной интерпретации)."""

    def __init__(
        self,
        window_size: int = 5,
        min_bpm: float = 42.0,
        max_bpm: float = 240.0,
        max_change_per_step_bpm: float | None = 5.0,
    ):
        """min_bpm/max_bpm по умолчанию — ФИЗИОЛОГИЧЕСКАЯ полоса FilterConfig
        (0.7-4.0 Гц -> 42-240 BPM), а НЕ отдельный более узкий "правдоподобный"
        диапазон (см. докстринг модуля про старый [55, 120]). При нестандартной
        полосе фильтра передайте cfg.filt.low_hz*60/cfg.filt.high_hz*60 явно —
        значения здесь НЕ читаются из FilterConfig автоматически, чтобы
        tracker.py не зависел от config.py."""
        self.window_size = window_size
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        # Задача 12: ЧЕСТНОЕ сглаживание ОТОБРАЖЕНИЯ — ограничивает, на
        # сколько BPM отображаемое число может измениться ЗА ОДИН ВЫЗОВ
        # update() (обычно раз в WindowConfig.step_seconds), потому что
        # реальный пульс физически не прыгает на десятки ударов за секунду.
        # Применяется к уже взвешенно-усреднённому РЕАЛЬНОМУ измерению, а
        # НЕ к выдуманной константе (в отличие от прежнего
        # BestEffortConfig.max_change_per_step_bpm, который сглаживал шаги
        # ВОКРУГ fallback_bpm=75). None -> без сглаживания, отображается
        # взвешенная медиана как есть.
        self.max_change_per_step_bpm = max_change_per_step_bpm

        self._buffer: deque = deque(maxlen=window_size)  # [(bpm, weight), ...]
        self._last_valid_bpm: float | None = None
        self._last_valid_timestamp_s: float | None = None
        self._last_displayed_bpm: float | None = None

    def update(self, timestamp_s: float, bpm: float, confidence: float) -> TrackerOutput:
        """Один вызов на одно окно оценки. bpm может быть NaN (см.
        PTSDPulseFeatures.status == "no_signal", задача 11) — тогда вес
        автоматически 0, как и при confidence <= 0 или bpm вне
        [min_bpm, max_bpm] (октавная ошибка/гармоника — типичный признак
        нефизиологичного выброса, см. старую логику BestEffortConfig,
        которая по той же причине ПОЛНОСТЬЮ игнорировала такие кандидаты,
        а не сглаживала их наравне с валидными)."""
        is_valid = (
            bpm is not None
            and not np.isnan(bpm)
            and confidence > 0.0
            and self.min_bpm <= bpm <= self.max_bpm
        )
        weight = float(confidence) if is_valid else 0.0
        self._buffer.append((bpm if is_valid else float("nan"), weight))
        if is_valid:
            self._last_valid_bpm = float(bpm)
            self._last_valid_timestamp_s = float(timestamp_s)

        total_weight = sum(w for _, w in self._buffer)
        if total_weight <= 0.0:
            # Весь буфер сейчас нулевого веса — задача 12: возвращаем
            # last_valid_bpm/age, а НЕ выдуманное число. self._last_displayed_bpm
            # НЕ обновляется здесь (остаётся "заморожен"), чтобы возобновление
            # реальных измерений продолжало сглаживание с того же места, а не
            # скачком от какого-то промежуточного состояния.
            age = (
                None if self._last_valid_timestamp_s is None
                else max(0.0, timestamp_s - self._last_valid_timestamp_s)
            )
            return TrackerOutput(bpm=self._last_valid_bpm, is_fresh=False, age_seconds=age)

        target = _weighted_median([(b, w) for b, w in self._buffer if w > 0.0])

        if self.max_change_per_step_bpm is not None and self._last_displayed_bpm is not None:
            delta = float(np.clip(
                target - self._last_displayed_bpm, -self.max_change_per_step_bpm, self.max_change_per_step_bpm
            ))
            displayed = self._last_displayed_bpm + delta
        else:
            displayed = target

        self._last_displayed_bpm = displayed
        return TrackerOutput(bpm=displayed, is_fresh=True, age_seconds=0.0)
