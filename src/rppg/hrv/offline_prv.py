"""
Офлайн-цепочка вариабельности пульса (PRV): удары ищутся ОДИН раз на одном
непрерывном сигнале записи из одной фиксированной области, а не склеиваются
из перекрывающихся 10-секундных окон, как в онлайн-цепочке
(RPPGPipeline._update_ibi_log).

Модуль не меняет и не вызывает онлайн-цепочку оценки пульса.

Содержимое:
  * msptd_beat_detector — перенос на Python детектора MSPTD из набора
    ppg-beats (P. H. Charlton, source/msptd_beat_detector.m, коммит
    a1f3befeb218daa14591463ffe1e50f6e3a6b7b3; doi:10.5281/zenodo.6037646),
    лучшего в сравнении 15 детекторов (Charlton et al., Physiol. Meas. 2022,
    43(8), 085007). Алгоритм — Bishop & Ercole 2018.
  * continuous_roi_signal — непрерывный сигнал области на всей записи: тот же
    POS с реконструкцией наложением, та же предобработка, фильтрация один раз
    с нулевой фазой.
  * BeatSeries / prv_metrics — ряд интервалов с признаком смежности; RMSSD и
    pNN50 только по разностям смежных интервалов.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend as _linear_detrend

from rppg.hrv.features import detect_pulse_peaks, refine_peaks_subsample
from rppg.signal.methods import SignalWindow, get_method
from rppg.signal.preprocessing import fill_gaps, preprocess_signal

# --------------------------------------------------------------------------
# MSPTD (перенос msptd_beat_detector.m)
# --------------------------------------------------------------------------
MSPTD_WIN_LEN_S = 6.0      # win_len
MSPTD_OVERLAP = 0.2        # overlap
MSPTD_TOL_S = 0.05         # tol = ceil(fs*0.05)


def _msptd_window(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """detect_peaks_and_onsets_using_msptd: скалограммы локальных максимумов и
    минимумов, масштаб с наибольшим числом экстремумов, пик — столбец, где
    отсчёт — максимум на ВСЕХ масштабах до найденного. Индексы 0-based."""
    x = _linear_detrend(np.asarray(x, dtype=float))   # detrend(x): минус МНК-прямая
    n = len(x)
    L = int(np.ceil(n / 2)) - 1
    if L < 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    m_max = np.zeros((L, n), dtype=bool)
    m_min = np.zeros((L, n), dtype=bool)
    for k in range(1, L + 1):
        # MATLAB: i = k+2 .. N-k+1, центр c = i-1 (1-based) = k+1 .. N-k
        # -> 0-based центры k .. n-1-k, сравнение с x[c-k] и x[c+k]
        c = x[k:n - k]
        m_max[k - 1, k:n - k] = (c > x[0:n - 2 * k]) & (c > x[2 * k:n])
        m_min[k - 1, k:n - k] = (c < x[0:n - 2 * k]) & (c < x[2 * k:n])
    lam_max = int(np.argmax(m_max.sum(axis=1))) + 1   # max(): первое вхождение
    lam_min = int(np.argmax(m_min.sum(axis=1))) + 1
    p = np.where(m_max[:lam_max].all(axis=0))[0]
    t = np.where(m_min[:lam_min].all(axis=0))[0]
    return p, t


def msptd_beat_detector(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Пики и впадины пульсовой волны, 0-based индексы.

    Отличия от исходника, неизбежные при переносе (больше никаких):
      * no_samps_in_win = round(6*fs): в Matlab при нецелом 6*fs индексация
        падает, исходник рассчитан на целую частоту;
      * окно коррекции пика ±tol обрезается границами окна сигнала: в Matlab
        выход за границы — ошибка индексации;
      * понижение частоты в исходнике выключено (do_ds = 0) — здесь его нет.
    """
    sig = np.asarray(sig, dtype=float)
    n_sig = len(sig)
    no_samps = int(round(MSPTD_WIN_LEN_S * fs))
    if n_sig <= no_samps:
        bounds = [(0, n_sig)]
    else:
        offset = int(round(no_samps * (1 - MSPTD_OVERLAP)))
        # Matlab: win_starts = 1:offset:len-no_samps; win_ends = starts+no_samps (включительно)
        starts = list(range(0, n_sig - no_samps, offset))
        bounds = [(s, s + no_samps + 1) for s in starts]
        if bounds[-1][1] < n_sig:
            bounds.append((n_sig - no_samps - 1, n_sig))
    tol = int(np.ceil(fs * MSPTD_TOL_S))
    peaks, onsets = [], []
    for s, e in bounds:
        win = sig[s:e]
        p, t = _msptd_window(win)
        for idx_list, out, fn in ((p, peaks, np.argmax), (t, onsets, np.argmin)):
            for cur in idx_list:
                lo, hi = max(cur - tol, 0), min(cur + tol, len(win) - 1)
                out.append(s + lo + int(fn(win[lo:hi + 1])))
    return np.unique(np.array(peaks, dtype=int)), np.unique(np.array(onsets, dtype=int))


# --------------------------------------------------------------------------
# Непрерывный сигнал области на всей записи
# --------------------------------------------------------------------------
@dataclass
class ContinuousSignal:
    roi: str
    timestamps_ms: np.ndarray   # метка каждого кадра
    fps: float                  # (N-1)/span по меткам записи
    signal: np.ndarray          # после предобработки
    bad: np.ndarray             # кадр внутри недопустимо длинного провала


MAX_GAP_FRAMES = 15  # тот же предел, что preprocessing.is_gap_acceptable


def _long_gap_mask(valid: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    bad = np.zeros(len(valid), dtype=bool)
    i = 0
    while i < len(valid):
        if not valid[i]:
            j = i
            while j < len(valid) and not valid[j]:
                j += 1
            if j - i > max_gap:
                bad[i:j] = True
            i = j
        else:
            i += 1
    return bad


def continuous_roi_signal(timestamps_ms, rgb: np.ndarray, valid: np.ndarray, roi: str, cfg) -> ContinuousSignal:
    """POS (реконструкция наложением с перекрытием, окно cfg — 1,6 с) и
    preprocess_signal с параметрами cfg.filt — на всей записи сразу.
    Пропуски заполняются fill_gaps; кадры внутри провалов длиннее
    MAX_GAP_FRAMES помечаются в .bad — удары там не используются."""
    ts = np.asarray(timestamps_ms, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    fps = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0)
    filled = np.stack([fill_gaps(np.asarray(rgb[:, ch], dtype=float), valid) for ch in range(3)], axis=1)
    win = SignalWindow(rgb_traces={roi: filled}, fps=fps, landmark_trajectories=None,
                       valid_mask=valid, hr_band_hz=(cfg.filt.low_hz, cfg.filt.high_hz))
    kwargs = {"use_numba": cfg.accel.use_numba} if cfg.method.value == "pos" else {}
    raw = get_method(cfg.method.value, **kwargs).extract(win, roi)
    sig = preprocess_signal(raw, fps, cfg.filt.low_hz, cfg.filt.high_hz, order=cfg.filt.filter_order,
                            detrend_method=cfg.filt.detrend_method, tarvainen_lambda=cfg.filt.tarvainen_lambda,
                            normalize_method=cfg.filt.normalize_method)
    return ContinuousSignal(roi=roi, timestamps_ms=ts, fps=fps, signal=sig, bad=_long_gap_mask(valid))


def beats_v1(cs: ContinuousSignal) -> np.ndarray:
    """V1: текущий detect_pulse_peaks + refine_peaks_subsample, как есть.
    Возвращает дробные индексы."""
    p = detect_pulse_peaks(cs.signal, cs.fps)
    return refine_peaks_subsample(cs.signal, p)


def beats_v2(cs: ContinuousSignal) -> np.ndarray:
    """V2: MSPTD + то же параболическое уточнение, что у V1."""
    p, _ = msptd_beat_detector(cs.signal, cs.fps)
    return refine_peaks_subsample(cs.signal, p)


def index_to_time_ms(cs: ContinuousSignal, idx: np.ndarray, edge_trim_s: float) -> np.ndarray:
    """Дробные индексы -> мс по меткам кадров; удары в кадрах-провалах и в
    первых/последних edge_trim_s секундах записи (краевой эффект фильтрации)
    отбрасываются."""
    idx = np.asarray(idx, dtype=float)
    if len(idx) == 0:
        return idx
    keep = ~cs.bad[np.clip(np.round(idx).astype(int), 0, len(cs.bad) - 1)]
    t = np.interp(idx, np.arange(len(cs.timestamps_ms)), cs.timestamps_ms)
    t0, t1 = cs.timestamps_ms[0] + edge_trim_s * 1000, cs.timestamps_ms[-1] - edge_trim_s * 1000
    return t[keep & (t >= t0) & (t <= t1)]


# --------------------------------------------------------------------------
# Ряд интервалов и метрики
# --------------------------------------------------------------------------
MIN_IBI_MS = 60000.0 / 220.0
MAX_IBI_MS = 60000.0 / 40.0


@dataclass
class IBISeries:
    t_ms: np.ndarray        # время удара, которым заканчивается интервал
    ibi_ms: np.ndarray
    adjacent: np.ndarray    # интервал смежен с предыдущим (общий удар, оба допустимы)


def ibi_series_from_beats(beat_t_ms: np.ndarray, bad_spans_ms: list[tuple[float, float]] = ()) -> IBISeries:
    """Интервалы между соседними ударами. Интервал допустим, если лежит в
    [60000/220, 60000/40] мс и не пересекает провал сигнала; два соседних
    интервала смежны, если оба допустимы (у них общий удар)."""
    t = np.sort(np.asarray(beat_t_ms, dtype=float))
    if len(t) < 2:
        return IBISeries(np.array([]), np.array([]), np.array([], dtype=bool))
    ibi = np.diff(t)
    ok = (ibi >= MIN_IBI_MS) & (ibi <= MAX_IBI_MS)
    for a, b in bad_spans_ms:
        ok &= ~((t[:-1] < b) & (t[1:] > a))
    t_end, ibi = t[1:][ok], ibi[ok]
    # смежность: предыдущий допустимый интервал заканчивается там, где начинается этот
    starts = t_end - ibi
    adjacent = np.zeros(len(ibi), dtype=bool)
    adjacent[1:] = np.abs(starts[1:] - t_end[:-1]) < 1e-6
    return IBISeries(t_end, ibi, adjacent)


def prv_metrics(series: IBISeries, use_adjacency: bool = True, pnn_ms: float = 50.0) -> dict:
    """SDNN — по всем допустимым интервалам; RMSSD и pNN50 — по разностям
    соседних элементов ряда, при use_adjacency только смежных."""
    ibi = series.ibi_ms
    if len(ibi) < 2:
        return {"n_ibi": int(len(ibi)), "rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan}
    d = np.diff(ibi)
    if use_adjacency:
        d = d[series.adjacent[1:]]
    return {
        "n_ibi": int(len(ibi)),
        "sdnn": float(np.std(ibi, ddof=1)),
        "rmssd": float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan,
        "pnn50": float(np.mean(np.abs(d) > pnn_ms) * 100) if len(d) else np.nan,
    }


def match_beats(a_ms: np.ndarray, b_ms: np.ndarray, tol_ms: float = 150.0) -> list[tuple[int, int]]:
    """Сопоставление один к одному: пары по возрастанию |Δt|, каждая сторона
    участвует не больше одного раза, |Δt| <= tol_ms. Детерминировано."""
    a, b = np.asarray(a_ms, float), np.asarray(b_ms, float)
    if len(a) == 0 or len(b) == 0:
        return []
    cand = []
    for i, ta in enumerate(a):
        lo, hi = np.searchsorted(b, ta - tol_ms), np.searchsorted(b, ta + tol_ms, side="right")
        for j in range(lo, hi):
            cand.append((abs(ta - b[j]), i, j))
    cand.sort()
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); pairs.append((i, j))
    return sorted(pairs)
