"""Тесты офлайн-цепочки PRV: перенос MSPTD на синтетике с известными ударами,
сопоставление ударов и ряд интервалов со смежностью."""

from __future__ import annotations

import numpy as np
import pytest

from rppg.hrv.offline_prv import (
    _msptd_window, ibi_series_from_beats, match_beats, msptd_beat_detector, prv_metrics,
)


def synthetic_ppg(fs: float, duration_s: float = 60.0, seed: int = 0, noise: float = 0.03):
    """Пульсовая волна с известными ударами: систолический гауссов пик и
    дикротическая волна (0,35 высоты через 0,3 интервала), интервалы с
    дыхательной аритмией и случайным разбросом, медленный дрейф, шум."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    beats, tb = [], 0.5
    while tb < duration_s - 0.5:
        beats.append(tb)
        ibi = 0.8 + 0.06 * np.sin(2 * np.pi * 0.25 * tb) + rng.normal(0, 0.02)
        tb += ibi
    beats = np.array(beats)
    x = np.zeros_like(t)
    for i, b in enumerate(beats):
        ibi = beats[i + 1] - b if i + 1 < len(beats) else 0.8
        x += np.exp(-0.5 * ((t - b) / 0.06) ** 2)
        x += 0.35 * np.exp(-0.5 * ((t - b - 0.3 * ibi) / 0.08) ** 2)
    x += 0.3 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, noise, len(t))
    return t, x, beats


@pytest.mark.parametrize("fs", [30.0, 62.5, 28.67])
def test_msptd_finds_known_beats(fs):
    t, x, beats = synthetic_ppg(fs)
    peaks, _ = msptd_beat_detector(x, fs)
    pairs = match_beats(t[peaks] * 1000, beats * 1000, tol_ms=150)
    tp = len(pairs)
    sens, ppv = tp / len(beats), tp / max(len(peaks), 1)
    err = np.array([abs(t[peaks][i] - beats[j]) for i, j in pairs]) * 1000
    assert sens >= 0.99 and ppv >= 0.99, (sens, ppv)
    assert np.median(err) <= 1000 / fs, np.median(err)  # не грубее одного отсчёта


def test_msptd_window_matches_definition_on_simple_signal():
    """Строго периодический сигнал: пики совпадают с максимумами синусоиды.
    По определению MSPTD пик требует сравнений на всех масштабах до λ
    (здесь λ = период, 24 отсчёта), поэтому максимумы ближе λ к краям окна
    не отмечаются — это свойство алгоритма, а не ошибка переноса."""
    fs, f0 = 30.0, 1.25
    n = int(6 * fs)
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f0 * t)
    p, _ = _msptd_window(x)
    expected = np.round((np.arange(0, 6, 1 / f0) + 0.25 / f0) * fs).astype(int)
    period = int(round(fs / f0))
    interior = expected[(expected >= period) & (expected <= n - 1 - period)]
    assert set(p) == set(interior)


def test_match_beats_is_one_to_one():
    pairs = match_beats(np.array([0.0, 100.0, 1000.0]), np.array([50.0, 980.0]), tol_ms=150)
    assert pairs == [(0, 0), (2, 1)] or pairs == [(1, 0), (2, 1)]
    assert len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})


def test_ibi_series_breaks_adjacency_on_rejected_interval():
    beats = np.array([0, 800, 1600, 4000, 4800, 5600], dtype=float)  # 2400 мс > 1500 — вне диапазона
    s = ibi_series_from_beats(beats)
    assert list(s.ibi_ms) == [800, 800, 800, 800]
    assert list(s.adjacent) == [False, True, False, True]
    m = prv_metrics(s)
    assert m["rmssd"] == 0.0  # разность через дыру не учитывается
