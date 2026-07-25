"""
Numba-ускорение (пункт "Производительность" ТЗ).

Важное инженерное уточнение, которое стоит проговорить явно (подробнее —
docs/research_report.md, раздел 6): большая часть пайплайна НЕ выигрывает
от Numba, потому что:

  - инференс MediaPipe уже выполняется скомпилированным TFLite-графом
    (XNNPACK/GPU-делегат) — векторизация в Python его не касается;
  - усреднение пикселей по ROI-маске (cv2 + np.mean) — уже C-векторизовано;
  - CHROM/GREEN/PCA/ICA работают на векторах из нескольких сотен отсчётов —
    накладные расходы JIT-компиляции сопоставимы с самим вычислением.

Единственное узкое место, где явный Python-цикл действителен и заметен —
skользящее окно POS с overlap-add (methods.PosMethod): на видео 30 fps и
окне анализа 10 с это ~300 итераций скользящего окна на КАЖДОЕ 10-секундное
окно анализа, повторяющихся при каждом шаге пайплайна (step_seconds=1s).
Здесь JIT даёт реальное ускорение (~10-30x на CPU по опыту аналогичных
циклов), поэтому Numba применена именно тут — предметно, а не "для галочки".

Модуль опционален: если numba не установлена, methods.PosMethod автоматически
использует эталонную numpy-реализацию (идентичный результат, медленнее).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - ветка для окружений без numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator


@njit(cache=True)
def _pos_core_numba(rgb: np.ndarray, ws: int, proj: np.ndarray) -> np.ndarray:
    t = rgb.shape[0]
    h = np.zeros(t)
    for n in range(t - ws + 1):
        c = rgb[n : n + ws]

        means = np.zeros(3)
        for ch in range(3):
            s = 0.0
            for i in range(ws):
                s += c[i, ch]
            m = s / ws
            means[ch] = m if abs(m) > 1e-8 else 1e-8

        cn = np.empty((ws, 3))
        for i in range(ws):
            for ch in range(3):
                cn[i, ch] = c[i, ch] / means[ch]

        s1 = np.empty(ws)
        s2 = np.empty(ws)
        for i in range(ws):
            s1[i] = proj[0, 0] * cn[i, 0] + proj[0, 1] * cn[i, 1] + proj[0, 2] * cn[i, 2]
            s2[i] = proj[1, 0] * cn[i, 0] + proj[1, 1] * cn[i, 1] + proj[1, 2] * cn[i, 2]

        std1 = np.std(s1)
        std2 = np.std(s2)
        alpha = std1 / std2 if std2 > 1e-8 else 1.0

        mean_hw = 0.0
        h_window = np.empty(ws)
        for i in range(ws):
            h_window[i] = s1[i] + alpha * s2[i]
            mean_hw += h_window[i]
        mean_hw /= ws

        for i in range(ws):
            h[n + i] += h_window[i] - mean_hw

    return h


def pos_overlap_add_numba(rgb: np.ndarray, ws: int, proj: np.ndarray) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is not installed")
    return _pos_core_numba(np.ascontiguousarray(rgb, dtype=np.float64), int(ws), proj)
