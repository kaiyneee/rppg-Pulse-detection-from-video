"""
Объективная оценка тона кожи по ITA (Individual Typology Angle) — п.29
требований: "Анализ по тону кожи — самое востребованное сейчас направление
в rPPG, и в большинстве работ он до сих пор отсутствует."

Публичные rPPG-датасеты (UBFC-rPPG, PURE, COHFACE, MMSE-HR) почти никогда не
поставляются с разметкой тона кожи по Fitzpatrick — она либо отсутствует
вовсе, либо требует ручной дерматологической оценки, которая в этой среде
недоступна. ITA — стандартный ОБЪЕКТИВНЫЙ, полностью воспроизводимый и не
требующий ручной разметки прокси, вычисляемый прямо по пикселям кадра:

    ITA° = arctan((L* - 50) / b*) * 180/π

(Chardon, Cretois, Hourseau, 1991, "Skin colour typology and suntanning
pathways", International Journal of Cosmetic Science). Пороги бакетов —
Del Bino & Bernerd, 2013, British Journal of Dermatology.

ВАЖНАЯ ОГОВОРКА: ITA — прокси, а НЕ замена клинической шкалы Fitzpatrick
(которая дополнительно учитывает историю реакции кожи на солнце, а не
только текущий цвет, и зависит от освещения при съёмке). Соответствие
ITA-бакетов номерам Fitzpatrick I-VI ниже — общепринятое приближение,
используемое в CV/дерматологической литературе при отсутствии клинической
разметки, но не идентичное ей. Если у датасета ЕСТЬ настоящая
Fitzpatrick-разметка (изредка в демографических CSV) — используйте ЕЁ,
а не ITA.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Del Bino & Bernerd (2013) пороги ITA° -> бакет; последняя колонка —
# приближённое соответствие Fitzpatrick I-VI (см. модульный docstring).
_ITA_BUCKETS: list[tuple[float, str, str]] = [
    (55.0, "very_light", "I"),
    (41.0, "light", "II"),
    (28.0, "intermediate", "III"),
    (10.0, "tan", "IV"),
    (-30.0, "brown", "V"),
    (-np.inf, "dark", "VI"),
]


@dataclass
class SkinToneEstimate:
    ita_degrees: float
    bucket_name: str        # very_light|light|intermediate|tan|brown|dark
    fitzpatrick_proxy: str  # "I".."VI" — ПРИБЛИЖЕНИЕ, см. модульный docstring
    n_pixels: int


def _ita_to_bucket(ita_degrees: float) -> tuple[str, str]:
    for threshold, name, fitzpatrick in _ITA_BUCKETS:
        if ita_degrees > threshold:
            return name, fitzpatrick
    return _ITA_BUCKETS[-1][1], _ITA_BUCKETS[-1][2]


def estimate_skin_tone(
    frame_bgr: np.ndarray, skin_mask: np.ndarray, min_pixels: int = 50
) -> SkinToneEstimate | None:
    """
    frame_bgr: (H,W,3) uint8 BGR-кадр.
    skin_mask: (H,W) бинарная маска пикселей кожи — например,
    face.roi.build_skin_mask(frame_bgr) пересечённая с объединением ROI
    лица (см. estimate_skin_tone_from_landmarks ниже для готовой обёртки).

    Возвращает None, если валидных пикселей кожи меньше min_pixels — тогда
    вызывающий код должен попробовать другой кадр, а не тянуть в среднее
    ненадёжную оценку по единичным пикселям.
    """
    ys, xs = np.where(skin_mask > 0)
    if len(ys) < min_pixels:
        return None

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    # OpenCV хранит 8-битный Lab как L в [0,255] (не [0,100]) и a,b со
    # сдвигом +128 — приводим к стандартной шкале CIE L*a*b* перед ITA.
    l_star = lab[ys, xs, 0] * (100.0 / 255.0)
    b_star = lab[ys, xs, 2] - 128.0

    # Медиана, а не среднее — та же логика устойчивости к бликам/теням, что
    # и в roi.roi_mean_rgb (единичные экстремальные пиксели не должны
    # определять ITA всего лица).
    mean_l = float(np.median(l_star))
    mean_b = float(np.median(b_star))
    if abs(mean_b) < 1e-6:
        mean_b = 1e-6 if mean_b >= 0 else -1e-6

    ita = float(np.degrees(np.arctan((mean_l - 50.0) / mean_b)))
    bucket_name, fitzpatrick_proxy = _ita_to_bucket(ita)

    return SkinToneEstimate(
        ita_degrees=ita,
        bucket_name=bucket_name,
        fitzpatrick_proxy=fitzpatrick_proxy,
        n_pixels=len(ys),
    )


def estimate_skin_tone_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
    shrink_factor: float = 0.9,
) -> SkinToneEstimate | None:
    """Удобная обёртка: строит маску кожи по ОБЪЕДИНЕНИЮ всех ROI лица (лоб
    + обе щеки) на одном кадре — больше пикселей для устойчивой оценки, чем
    один ROI. Предназначено для РАЗОВОЙ оценки на подходящем (лицо анфас,
    без сильных теней) кадре каждого испытуемого при подготовке
    metadata_by_subject для evaluate.stratified_report (п.29), а НЕ для
    покадрового вызова внутри RPPGPipeline (это отдельная, offline-стадия
    подготовки датасета, не часть онлайн-пайплайна)."""
    from rppg.face import roi as roi_module

    skin_mask = roi_module.build_skin_mask(frame_bgr)
    combined_roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for name in roi_module.ROI_LANDMARK_INDICES:
        combined_roi_mask |= roi_module.build_roi_mask(landmarks_px, name, frame_bgr.shape, shrink_factor)

    return estimate_skin_tone(frame_bgr, combined_roi_mask & skin_mask)
