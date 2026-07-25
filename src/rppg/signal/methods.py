"""
Методы извлечения пульсового сигнала (пункт "Извлечение сигнала" ТЗ).

Реализовано 6 методов с единым интерфейсом PulseExtractionMethod.extract(),
что позволяет переключаться между ними без изменения остального пайплайна
(Strategy pattern). Регистрация в METHOD_REGISTRY даёт runtime-переключение
по значению ExtractionMethod из config.py.

Ключевое разделение (важно для понимания, почему методы вообще сравнимы):

* GREEN / CHROM / POS / PCA(цвет) / ICA(цвет) — работают с усреднённым по ROI
  цветом кожи (photoplethysmographic principle: поглощение света гемоглобином
  меняется на каждом сердечном цикле).
* HEAD_MOTION — работает не с цветом, а с траекториями landmark-точек
  (ballistocardiographic principle: механическая отдача от притока крови
  вызывает микродвижения головы, Balakrishnan et al., CVPR 2013 — метод
  оригинального репозитория, здесь модернизирован под MediaPipe-трекинг).

Источники алгоритмов (для воспроизводимости и корректной атрибуции):
  GREEN — Verkruysse, Svaasand, Nelson (2008), Optics Express.
  CHROM — de Haan & Jeanne (2013), IEEE TBME.
  POS   — Wang, den Brinker, Stuijk, de Haan (2016), IEEE TBME
          ("Algorithmic Principles of Remote PPG").
  ICA   — Poh, McDuff, Picard (2010/2011), Optics Express / IEEE TBME.
  PCA   — Lewandowska, Rumiński, Kowalik, Nowak (2011), FedCSIS.
  Head-motion — Balakrishnan, Durand, Guttag (2013), CVPR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA, FastICA

from rppg.signal.preprocessing import detrend as _detrend
from rppg.signal.quality import select_best_component_by_periodicity


@dataclass
class SignalWindow:
    """Единица данных, которую видит каждый метод извлечения."""

    # roi_name -> массив (T, 3) средних значений R,G,B по ROI за окно.
    rgb_traces: dict[str, np.ndarray]
    fps: float
    # (T, N, 2) координаты N стабильных landmark-точек, для HEAD_MOTION.
    landmark_trajectories: np.ndarray | None = None
    valid_mask: np.ndarray | None = None  # (T,) bool — кадр не окклюдирован
    hr_band_hz: tuple[float, float] = (0.7, 4.0)


class PulseExtractionMethod(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, window: SignalWindow, roi_name: str) -> np.ndarray:
        """Возвращает 1D пульсовой сигнал длины T (ещё без bandpass-фильтра)."""
        raise NotImplementedError


def _temporal_normalize(rgb: np.ndarray) -> np.ndarray:
    """Cn = C / mean(C) по каждому каналу — общий шаг для CHROM/POS/PCA/ICA."""
    means = np.mean(rgb, axis=0)
    means = np.where(np.abs(means) < 1e-8, 1e-8, means)
    return rgb / means


class GreenMethod(PulseExtractionMethod):
    """Простейший baseline: зелёный канал наиболее чувствителен к объёмному
    пульсу крови из-за пика поглощения гемоглобином в этом диапазоне длин волн."""

    name = "green"

    def extract(self, window: SignalWindow, roi_name: str) -> np.ndarray:
        rgb = window.rgb_traces[roi_name]
        g = rgb[:, 1]
        return _detrend(g, method="linear")


class ChromMethod(PulseExtractionMethod):
    """CHROM (de Haan & Jeanne, 2013).

    Строит две хроминансные комбинации, ортогональные направлению кожного тона,
    что подавляет специфичную для конкретной кожи составляющую и даёт
    устойчивость к умеренному движению по сравнению с GREEN/PCA.
    """

    name = "chrom"

    def extract(self, window: SignalWindow, roi_name: str) -> np.ndarray:
        rgb = window.rgb_traces[roi_name]
        cn = _temporal_normalize(rgb)
        rn, gn, bn = cn[:, 0], cn[:, 1], cn[:, 2]

        xs = 3 * rn - 2 * gn
        ys = 1.5 * rn + gn - 1.5 * bn

        std_xs = np.std(xs)
        std_ys = np.std(ys)
        alpha = std_xs / std_ys if std_ys > 1e-8 else 1.0

        return xs - alpha * ys


class PosMethod(PulseExtractionMethod):
    """POS — Plane-Orthogonal-to-Skin (Wang et al., 2016).

    В большинстве независимых бенчмарков (в т.ч. в оригинальной статье и в
    rPPG-Toolbox, NeurIPS'23) POS показывает лучшую или сопоставимую с CHROM
    точность и лучшую устойчивость к движению среди классических
    unsupervised-методов — поэтому выбран как метод по умолчанию в пайплайне
    (см. config.PipelineConfig.method).

    Использует скользящее окно с overlap-add реконструкцией (Algorithm 1
    оригинальной статьи), а не однократную проекцию на весь трейс — это и
    есть источник его помехоустойчивости: проекционная плоскость
    пересчитывается локально, подстраиваясь под медленные изменения
    освещения/тона кожи внутри длинного окна.
    """

    name = "pos"
    projection_matrix = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

    def __init__(self, window_seconds: float = 1.6, use_numba: bool = True):
        self.window_seconds = window_seconds
        self.use_numba = use_numba

    def extract(self, window: SignalWindow, roi_name: str) -> np.ndarray:
        rgb = window.rgb_traces[roi_name]
        fps = window.fps
        ws = max(3, int(round(self.window_seconds * fps)))

        if self.use_numba:
            try:
                from rppg.accel.fast_ops import pos_overlap_add_numba

                return pos_overlap_add_numba(rgb, ws, self.projection_matrix)
            except ImportError:
                pass  # Numba недоступна — используем эталонную реализацию ниже.

        return self._pos_overlap_add_reference(rgb, ws)

    def _pos_overlap_add_reference(self, rgb: np.ndarray, ws: int) -> np.ndarray:
        t = rgb.shape[0]
        h = np.zeros(t)
        for n in range(t - ws + 1):
            c = rgb[n : n + ws]
            cn = _temporal_normalize(c)
            s = cn @ self.projection_matrix.T  # (ws, 2)
            s1, s2 = s[:, 0], s[:, 1]
            std1, std2 = np.std(s1), np.std(s2)
            alpha = std1 / std2 if std2 > 1e-8 else 1.0
            h_window = s1 + alpha * s2
            h_window = h_window - np.mean(h_window)
            h[n : n + ws] += h_window
        return h


class PcaColorMethod(PulseExtractionMethod):
    """PCA на нормализованных цветовых каналах (Lewandowska et al., 2011).

    Раскладывает [Rn,Gn,Bn] на ортогональные компоненты и выбирает ту, чей
    спектр наиболее "пульсоподобен" (см. select_best_component_by_periodicity) —
    тот же критерий отбора, что и в оригинальном head-motion методе, но
    применённый к цвету, а не к движению.
    """

    name = "pca"

    def extract(self, window: SignalWindow, roi_name: str) -> np.ndarray:
        rgb = window.rgb_traces[roi_name]
        cn = _temporal_normalize(rgb)
        cn = cn - cn.mean(axis=0, keepdims=True)

        pca = PCA(n_components=3, random_state=0)
        components = pca.fit_transform(cn)  # (T, 3)

        best = select_best_component_by_periodicity(
            components.T, window.fps, window.hr_band_hz
        )
        return components[:, best]


class IcaColorMethod(PulseExtractionMethod):
    """ICA на нормализованных цветовых каналах (Poh, McDuff, Picard, 2010/2011).

    В отличие от PCA (ортогональность по дисперсии) ICA ищет статистически
    независимые источники — теоретически лучше подходит, когда пульсовая
    составляющая и артефакты движения имеют независимую, а не просто
    некоррелированную природу.
    """

    name = "ica"

    def extract(self, window: SignalWindow, roi_name: str) -> np.ndarray:
        rgb = window.rgb_traces[roi_name]
        cn = _temporal_normalize(rgb)
        cn = cn - cn.mean(axis=0, keepdims=True)

        ica = FastICA(n_components=3, random_state=0, max_iter=500, whiten="unit-variance")
        try:
            components = ica.fit_transform(cn)  # (T, 3)
        except Exception:
            # FastICA может не сойтись на вырожденных/слишком коротких окнах —
            # деградируем до PCA, а не роняем пайплайн.
            return PcaColorMethod().extract(window, roi_name)

        best = select_best_component_by_periodicity(
            components.T, window.fps, window.hr_band_hz
        )
        return components[:, best]


class HeadMotionMethod(PulseExtractionMethod):
    """Head-motion метод (Balakrishnan, Durand, Guttag, CVPR 2013) —
    алгоритм оригинального репозитория, модернизированный:

    * точки для трекинга берутся не через Haar/dlib + ручной goodFeaturesToTrack,
      а напрямую из стабильных MediaPipe landmarks (устраняет дрейф "плохих"
      углов Shi-Tomasi на фоне/волосах);
    * PCA-компонента выбирается тем же единым критерием периодичности,
      что и в PcaColorMethod/IcaColorMethod (в оригинале — упрощённая
      версия того же критерия, реализованная отдельно и только для
      Haar/dlib-режима).
    """

    name = "head_motion"

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def extract(self, window: SignalWindow, roi_name: str | None = None) -> np.ndarray:
        if window.landmark_trajectories is None:
            raise ValueError(
                "HeadMotionMethod требует landmark_trajectories в SignalWindow"
            )
        traj = window.landmark_trajectories  # (T, N, 2)
        y = traj[:, :, 1]  # используем вертикальную компоненту, как в оригинале

        y_detrended = np.apply_along_axis(lambda col: _detrend(col, method="linear"), 0, y)

        n_components = min(self.n_components, y_detrended.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        components = pca.fit_transform(y_detrended)  # (T, n_components)

        best = select_best_component_by_periodicity(
            components.T, window.fps, window.hr_band_hz
        )
        return components[:, best]


METHOD_REGISTRY: dict[str, type[PulseExtractionMethod]] = {
    "green": GreenMethod,
    "chrom": ChromMethod,
    "pos": PosMethod,
    "pca": PcaColorMethod,
    "ica": IcaColorMethod,
    "head_motion": HeadMotionMethod,
}


def get_method(name: str, **kwargs) -> PulseExtractionMethod:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown extraction method '{name}'. Options: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](**kwargs)
