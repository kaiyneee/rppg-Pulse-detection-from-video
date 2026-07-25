"""
Динамические ROI: лоб, левая щека, правая щека (пункт "Отслеживание ROI" ТЗ).

ВАЖНО про индексы landmark-точек ниже: они НЕ придуманы и не взяты "по
памяти" — это реальные индексы, извлечённые программным интроспектированием
установленного пакета mediapipe==0.10.33 (mediapipe.tasks.python.vision.
face_landmarker.FaceLandmarksConnections), а именно из множеств
FACE_LANDMARKS_FACE_OVAL / _LEFT_EYE / _RIGHT_EYE / _LEFT_EYEBROW /
_RIGHT_EYEBROW / _NOSE / _LIPS. См. docs/research_report.md, раздел 3.1,
для деталей и способа воспроизведения.

Из этих проверенных "опорных" точек ROI лба и щёк строятся геометрически, а
не как отдельный "магический" список из полусотни неопределённо взятых
индексов — так проще проверить и скорректировать полигоны при необходимости.

Соглашение об именовании MediaPipe: LEFT_* / RIGHT_* — это левый/правый
относительно самого человека (анатомически), т.е. на "незеркальном"
фронтальном кадре RIGHT_EYE находится в левой части изображения.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- Проверенные множества индексов (см. docstring выше) --------------------

FACE_OVAL = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150,
             152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356,
             361, 365, 377, 378, 379, 389, 397, 400, 454]

LEFT_EYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
RIGHT_EYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220,
        275, 278, 294, 326, 327, 344, 440]
LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
        146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314,
        317, 318, 321, 324, 375, 402, 405, 409, 415]

# --- ROI-полигоны, построенные из опорных точек выше -------------------------
#
# Лоб: снизу ограничен верхними точками бровей (105/107 — правая бровь,
# 336/334 — левая), сверху — верхними точками овала лица около висков
# (109, 10, 338). Это анатомически лоб над бровями и под линией роста волос,
# которую MediaPipe не размечает напрямую (её нет в топологии как таковой),
# поэтому верхней границей служит верх овала лица.
FOREHEAD_IDX = [105, 107, 336, 334, 338, 10, 109]

# Правая щека (в анатомическом смысле = левая часть кадра): между внешним
# краем овала (127/234), нижним веком (155), крылом носа (98) и уголком рта (61).
RIGHT_CHEEK_IDX = [127, 234, 61, 98, 155]

# Левая щека — зеркальный аналог (356/454, 291, 327, 382).
LEFT_CHEEK_IDX = [356, 454, 291, 327, 382]

ROI_LANDMARK_INDICES: dict[str, list[int]] = {
    "forehead": FOREHEAD_IDX,
    "left_cheek": LEFT_CHEEK_IDX,
    "right_cheek": RIGHT_CHEEK_IDX,
}

# Расширенный "стабильный" набор точек для head-motion метода: контур лица +
# нос, БЕЗ точек рта/бровей (их движение при разговоре/мимике — не сердечный
# ballistocardiographic сигнал, а посторонний источник дисперсии, который
# в оригинальном репозитории устранялся вычитанием областей глаз/рта).
STABLE_TRACKING_IDX = sorted(set(FACE_OVAL) | set(NOSE))


def landmarks_to_pixels(landmarks_norm: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """landmarks_norm: (N,2) или (N,3) нормализованные [0,1] координаты
    (как их отдаёт MediaPipe) -> пиксельные координаты (N,2) int32."""
    h, w = frame_shape[:2]
    px = landmarks_norm[:, :2] * np.array([w, h])
    return px.astype(np.int32)


def _shrink_polygon(points: np.ndarray, factor: float) -> np.ndarray:
    """Сжимает полигон к его центроиду. factor<1 отодвигает границу внутрь,
    снижая риск захвата фона/волос/ушей на краю ROI при повороте головы."""
    centroid = points.mean(axis=0, keepdims=True)
    return (centroid + (points - centroid) * factor).astype(np.int32)


def build_roi_mask(
    landmarks_px: np.ndarray,
    roi_name: str,
    frame_shape: tuple[int, int],
    shrink_factor: float = 0.9,
) -> np.ndarray:
    """Строит бинарную маску ROI (H,W) uint8 {0,1} через выпуклую оболочку
    опорных точек региона + инородное сжатие к центроиду."""
    idx = ROI_LANDMARK_INDICES[roi_name]
    pts = landmarks_px[idx]
    hull = cv2.convexHull(pts)
    hull_shrunk = _shrink_polygon(hull.reshape(-1, 2), shrink_factor)

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_shrunk, 1)
    return mask


def roi_mean_rgb(frame_bgr: np.ndarray, mask: np.ndarray, min_valid_fraction: float = 0.6) -> tuple[np.ndarray | None, bool]:
    """Среднее R,G,B по маске. Возвращает (rgb[3] | None, valid).

    valid=False, если после исключения переэкспонированных/недоэкспонированных
    пикселей (типичный артефакт при бликах/тенях) валидных пикселей осталось
    меньше min_valid_fraction от площади ROI — тогда кадр не должен тянуть
    среднее ROI в сторону артефакта, а должен считаться отсутствующим
    (см. preprocessing.interpolate_missing).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, False

    pixels = frame_bgr[ys, xs].astype(np.float32)  # (K,3) BGR
    brightness = pixels.mean(axis=1)
    good = (brightness > 10) & (brightness < 250)  # отсекаем клиппинг

    if good.sum() < min_valid_fraction * len(ys):
        return None, False

    mean_bgr = pixels[good].mean(axis=0)
    mean_rgb = mean_bgr[::-1].copy()  # BGR -> RGB
    return mean_rgb, True


@dataclass
class ROIExtractionResult:
    rgb_by_roi: dict[str, np.ndarray | None]
    valid_by_roi: dict[str, bool]
    landmarks_px: np.ndarray
    stable_tracking_points: np.ndarray  # (len(STABLE_TRACKING_IDX), 2)


def extract_rois(
    frame_bgr: np.ndarray,
    landmarks_norm: np.ndarray,
    roi_names: tuple[str, ...] = ("forehead", "left_cheek", "right_cheek"),
    shrink_factor: float = 0.9,
    min_valid_fraction: float = 0.6,
) -> ROIExtractionResult:
    landmarks_px = landmarks_to_pixels(landmarks_norm, frame_bgr.shape)

    rgb_by_roi: dict[str, np.ndarray | None] = {}
    valid_by_roi: dict[str, bool] = {}
    for name in roi_names:
        mask = build_roi_mask(landmarks_px, name, frame_bgr.shape, shrink_factor)
        rgb, valid = roi_mean_rgb(frame_bgr, mask, min_valid_fraction)
        rgb_by_roi[name] = rgb
        valid_by_roi[name] = valid

    stable_points = landmarks_px[STABLE_TRACKING_IDX].astype(np.float64)

    return ROIExtractionResult(
        rgb_by_roi=rgb_by_roi,
        valid_by_roi=valid_by_roi,
        landmarks_px=landmarks_px,
        stable_tracking_points=stable_points,
    )
