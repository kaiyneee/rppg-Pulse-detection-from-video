"""
Офлайн-оценщики частоты пульса над набором spectral_dataset.

Всё здесь — чистая обработка массивов над сигналами, которые пайплайн уже
посчитал (см. spectral_dataset.py), без видео и без MediaPipe. Цель —
быстро сравнить способы оценки частоты, объединения ROI и отслеживания
пика на РАЗРАБОТОЧНОЙ части, прежде чем переносить победителя в пайплайн.

Все спектры переводятся на общую сетку в уд/мин (BPM_GRID): после
исправления _estimate_fps частота дискретизации чуть различается от окна к
окну, а для усреднения спектров и отслеживания через окна нужна единая
сетка.

Два эталона:
  ref_paper — эталон первой статьи (Уэлч на контактной волне; как и оценка
              пайплайна, использует первые 8 с из 10-секундного окна);
  ref_beats — эталон, не зависящий от спектрального оценщика: удары
              контактной волны (scipy.signal.find_peaks с параметрами из
              задания 3) на ПОЛНОМ 10-секундном окне. Нужен, чтобы честно
              сравнивать оценщики с разным временным охватом.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.signal.windows import dpss

BPM_GRID = np.arange(40.0, 240.0 + 1e-9, 0.1)


# --------------------------------------------------------------------------
# Спектры
# --------------------------------------------------------------------------
def spec_pipeline_welch(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Точная копия estimate_hr_welch: один 8-секундный сегмент Ханна."""
    nperseg = min(len(x), max(64, int(fps * 8)))
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    f, p = welch(x, fs=fps, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft)
    return f, p


def spec_periodogram(x: np.ndarray, fps: float, nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Периодограмма ВСЕГО окна с окном Ханна (без отбрасывания хвоста)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fps)
    return f, (np.abs(X) ** 2) / np.sum(w ** 2)


def spec_multitaper(x: np.ndarray, fps: float, nw: float = 2.0, k: int | None = None,
                    nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Многооконная оценка (DPSS) по всему окну: меньше дисперсия, чем у
    периодограммы, при контролируемом расширении пика (2·NW/T Гц)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    k = k or max(1, int(2 * nw) - 1)
    tapers = dpss(n, nw, Kmax=k)
    f = np.fft.rfftfreq(nfft, d=1.0 / fps)
    p = np.zeros(len(f))
    for t in tapers:
        p += np.abs(np.fft.rfft(x * t, n=nfft)) ** 2
    return f, p / k


def on_grid(f: np.ndarray, p: np.ndarray, grid: np.ndarray = BPM_GRID) -> np.ndarray:
    return np.interp(grid / 60.0, f, p)


def normalize(s: np.ndarray) -> np.ndarray:
    total = float(np.sum(s))
    return s / total if total > 0 else np.full_like(s, 1.0 / len(s))


# --------------------------------------------------------------------------
# Выбор пика
# --------------------------------------------------------------------------
def band_mask(band: tuple[float, float], grid: np.ndarray = BPM_GRID) -> np.ndarray:
    return (grid >= band[0] * 60.0) & (grid <= band[1] * 60.0)


def argmax_bpm(s: np.ndarray, mask: np.ndarray, refine: bool = False,
               grid: np.ndarray = BPM_GRID) -> float:
    s_m = np.where(mask, s, -np.inf)
    i = int(np.argmax(s_m))
    if not refine or i == 0 or i == len(s) - 1 or not (mask[i - 1] and mask[i + 1]):
        return float(grid[i])
    # параболическая интерполяция по логарифму мощности (гауссов пик)
    y0, y1, y2 = np.log(s[i - 1] + 1e-300), np.log(s[i] + 1e-300), np.log(s[i + 1] + 1e-300)
    denom = y0 - 2 * y1 + y2
    if denom >= 0:
        return float(grid[i])
    delta = 0.5 * (y0 - y2) / denom
    step = grid[1] - grid[0]
    return float(grid[i] + np.clip(delta, -1.0, 1.0) * step)


def viterbi_track(spectra: list[np.ndarray], mask: np.ndarray, sigma_bpm: float,
                  emission_power: float = 1.0, grid: np.ndarray = BPM_GRID,
                  coarse: int = 5) -> np.ndarray:
    """Наиболее вероятная траектория частоты через последовательность окон.

    Эмиссия — логарифм нормированного спектра окна (в степени
    emission_power), переход — гауссов штраф за изменение пульса между
    соседними окнами со шкалой sigma_bpm (уд/мин за шаг окна). Офлайн:
    решение принимается по всей записи сразу, как и второй проход.

    Для скорости решётка прореживается в `coarse` раз (шаг 0,5 уд/мин при
    coarse=5); итоговая частота уточняется argmax-ом спектра окна в
    окрестности найденного узла.
    """
    idx = np.where(mask)[0][::coarse]
    states = grid[idx]
    T, S = len(spectra), len(states)
    log_e = np.empty((T, S))
    for t, s in enumerate(spectra):
        sn = normalize(np.where(mask, s, 0.0))
        log_e[t] = emission_power * np.log(sn[idx] + 1e-12)
    d = states[:, None] - states[None, :]
    log_tr = -0.5 * (d / sigma_bpm) ** 2
    score = log_e[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        cand = score[None, :] + log_tr          # [куда, откуда]
        back[t] = np.argmax(cand, axis=1)
        score = cand[np.arange(S), back[t]] + log_e[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    out = np.empty(T)
    step = grid[1] - grid[0]
    half = int(round(coarse * 1.5))
    for t in range(T):
        centre = idx[path[t]]
        lo, hi = max(centre - half, 0), min(centre + half + 1, len(grid))
        local = np.where(mask[lo:hi], spectra[t][lo:hi], -np.inf)
        out[t] = grid[lo + int(np.argmax(local))]
    return out


# --------------------------------------------------------------------------
# Эталон по ударам контактной волны
# --------------------------------------------------------------------------
def ref_beats(t_ref: np.ndarray, wave: np.ndarray, t0: float, t1: float,
              min_beats: int = 3) -> float | None:
    """Пульс на [t0, t1] по ударам контактной волны.

    Приведение к равномерной сетке — как в основной работе (границы и число
    отсчётов сохраняются); пики — scipy.signal.find_peaks с distance = 0,33 с
    (максимум 180 уд/мин) и prominence = 0,3·СКО, как в задании 3.
    """
    m = (t_ref >= t0) & (t_ref <= t1)
    tw, xw = t_ref[m], np.asarray(wave, dtype=float)[m]
    if len(tw) < 16:
        return None
    span = float(tw[-1] - tw[0])
    if span <= 0:
        return None
    fs = (len(tw) - 1) / span
    tu = np.linspace(tw[0], tw[-1], len(tw))
    x = np.interp(tu, tw, xw)
    peaks, _ = find_peaks(x, distance=max(int(round(0.33 * fs)), 1), prominence=0.3 * float(np.std(x)))
    if len(peaks) < min_beats:
        return None
    ibi = np.diff(tu[peaks])
    return float(60.0 / np.mean(ibi))


# --------------------------------------------------------------------------
# Эталон статьи, но по ПОЛНОМУ окну
# --------------------------------------------------------------------------
def ref_spectral_full(t_ref: np.ndarray, wave: np.ndarray, t0: float, t1: float, cfg,
                      nfft: int = 8192) -> float | None:
    """Та же процедура, что reference_bpm_from_wave (равномерная сетка,
    перенос частоты среза детренда, preprocess_signal), но спектр — по
    ВСЕМУ окну (периодограмма Ханна), а не по одному 8-секундному сегменту
    Уэлча. Единственное отличие от эталона статьи — временной охват."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from rppg.signal.preprocessing import (
        preprocess_signal, tarvainen_cutoff_hz, tarvainen_lambda_for_cutoff,
    )

    m = (t_ref >= t0) & (t_ref <= t1)
    tw, xw = t_ref[m], np.asarray(wave, dtype=float)[m]
    if len(tw) < 16:
        return None
    span = float(tw[-1] - tw[0])
    if span <= 0:
        return None
    fs = (len(tw) - 1) / span
    tu = np.linspace(tw[0], tw[-1], len(tw))
    x = np.interp(tu, tw, xw)
    band = (cfg.filt.low_hz, cfg.filt.high_hz)
    lam = cfg.filt.tarvainen_lambda
    if cfg.filt.detrend_method == "tarvainen":
        lam = tarvainen_lambda_for_cutoff(
            tarvainen_cutoff_hz(cfg.filt.tarvainen_lambda, cfg.window.assumed_fps), fs,
        )
    y = preprocess_signal(x, fs, band[0], band[1], order=cfg.filt.filter_order,
                          detrend_method=cfg.filt.detrend_method, tarvainen_lambda=lam,
                          normalize_method=cfg.filt.normalize_method)
    f, pw = spec_periodogram(y, fs, nfft=nfft)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return None
    return float(f[mask][np.argmax(pw[mask])] * 60.0)
