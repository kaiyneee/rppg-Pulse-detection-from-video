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

# Наружные уголки глаз (canthus) — уже входят в проверенные множества
# RIGHT_EYE/LEFT_EYE выше. Используются как стандартный прокси межзрачкового
# расстояния (interocular distance) для АБСОЛЮТНОЙ нормировки джиттера
# landmark-точек в quality.landmark_stability_score (п.19 требований):
# расстояние между наружными уголками глаз даёт масштаб лица в пикселях
# (инвариантный к разрешению кадра/расстоянию до камеры), не требуя
# iris-landmarks, которые не всегда есть в выводе модели.
RIGHT_EYE_OUTER_IDX = 33
LEFT_EYE_OUTER_IDX = 263


def landmarks_to_pixels(landmarks_norm: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """landmarks_norm: (N,2) или (N,3) нормализованные [0,1] координаты
    (как их отдаёт MediaPipe) -> пиксельные координаты (N,2) int32."""
    h, w = frame_shape[:2]
    px = landmarks_norm[:, :2] * np.array([w, h])
    return px.astype(np.int32)


def interocular_distance_px(landmarks_px: np.ndarray) -> float:
    """Расстояние между наружными уголками глаз в пикселях — прокси
    межзрачкового расстояния для абсолютного масштабирования лица (п.19)."""
    p_right = landmarks_px[RIGHT_EYE_OUTER_IDX].astype(np.float64)
    p_left = landmarks_px[LEFT_EYE_OUTER_IDX].astype(np.float64)
    return float(np.linalg.norm(p_left - p_right))


def build_background_roi_mask(
    frame_shape: tuple[int, int],
    landmarks_px: np.ndarray,
    patch_fraction: float = 0.12,
    margin_fraction: float = 0.20,
) -> np.ndarray | None:
    """
    Фоновый ROI (п.22 требований) — фиксированный квадратный патч в одном из
    углов кадра, заведомо ВНЕ ограничивающего прямоугольника лица (+запас
    margin_fraction). Используется quality.detect_illumination_flicker, чтобы
    отличить реальный пульс от синхронного мерцания освещения: у фона (стены
    за головой) нет кровоснабжения, поэтому любая узкополосная периодичность
    в его яркости на той же частоте, что и "пульс" на лице, — почти наверняка
    внешний источник (мерцание/PWM), а не сердцебиение.

    Это дешёвая эвристика на основе bounding box'а лица, а не сегментация
    сцены — не гарантирует, что патч это именно стена, а не волосы/одежда/
    другой объект, но для детектора мерцания важно только то, что патч НЕ
    является кожей лица.

    ВАЖНО (задача 7, риск ложных срабатываний детектора мерцания):
    landmarks_px — это ТОЛЬКО face mesh (478 точек лица от MediaPipe), в нём
    физически нет точек волос/ушей/плеч/одежды — bounding box по этим
    точкам систематически ТЕСНЕЕ реального силуэта человека в кадре.
    margin_fraction — ЕДИНСТВЕННАЯ защита от захвата волос (особенно
    длинных/причёсок, выступающих далеко за овал лица) или плеч (которые в
    bbox не входят вообще и могут начинаться сразу под подбородком при
    крупном плане). Раньше margin_fraction=0.03 (3% от меньшей стороны
    кадра) — при типичном фронтальном крупном плане веб-камеры это буквально
    несколько пикселей, заведомо недостаточно: если фоновый патч задевает
    то, что физически движется ВМЕСТЕ с человеком, он коррелирует с лицом
    по построению, и детектор мерцания получает систематическую ложную
    тревогу на каждом окне. 0.20 — с существенным запасом; ЦЕНА — угол чаще
    признаётся "занятым" и bg_mask чаще возвращает None (нет данных для
    детектора мерцания), что в духе общего принципа проекта "лучше
    промолчать, чем соврать" (см. QualityConfig.flicker_min_raw_std рядом).

    Возвращает None, если ни один угол кадра не гарантированно свободен от
    лица (лицо занимает весь кадр вплотную к краям).
    """
    h, w = frame_shape[:2]
    x_min, y_min = float(landmarks_px[:, 0].min()), float(landmarks_px[:, 1].min())
    x_max, y_max = float(landmarks_px[:, 0].max()), float(landmarks_px[:, 1].max())

    patch = int(round(patch_fraction * min(h, w)))
    margin = int(round(margin_fraction * min(h, w)))
    if patch < 4:
        return None

    face_box = (x_min - margin, y_min - margin, x_max + margin, y_max + margin)
    candidates = [
        (0, 0),                  # верхний левый угол
        (w - patch, 0),          # верхний правый угол
        (0, h - patch),          # нижний левый угол
        (w - patch, h - patch),  # нижний правый угол
    ]

    for px0, py0 in candidates:
        px1, py1 = px0 + patch, py0 + patch
        overlaps_face = not (
            px1 <= face_box[0] or px0 >= face_box[2] or py1 <= face_box[1] or py0 >= face_box[3]
        )
        if not overlaps_face:
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            mask[py0:py1, px0:px1] = 1
            return mask
    return None


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


def build_skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Пороговая маска кожи в YCrCb (Chai & Ngan, 1999 — стандартный порог,
    воспроизводится в большинстве rPPG-репозиториев): 133 <= Cr <= 173,
    77 <= Cb <= 127.

    Зачем нужна ПОВЕРХ полигона ROI: FOREHEAD_IDX упирается в точку 10 у
    верха овала лица — это уже линия роста волос, MediaPipe её отдельно не
    размечает. shrink_factor=0.9 не спасает при чёлке, а яркостный порог
    в roi_mean_rgb (10-250) пропускает светлые/русые волосы, у которых
    яркость похожа на кожу. Без skin-маски такие пиксели попадают в
    оценку ROI и загрязняют пульсовой сигнал не связанной с пульсом
    составляющей (цвет и текстура волос, а не кожи).
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return mask.astype(np.uint8)


def roi_mean_rgb(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    min_valid_fraction: float = 0.6,
    trim_fraction: float = 0.10,
) -> tuple[np.ndarray | None, bool, int, str | None]:
    """Робастная оценка R,G,B по маске: усечённое среднее (truncated mean),
    НЕ медиана. Возвращает (rgb[3] | None, valid, n_valid_pixels).

    Задача 31 (2026-08-20) отменяет более раннее решение в пользу медианы,
    закреплённое в предыдущей версии этой функции — тот выбор был ошибкой.
    Пульсовая модуляция яркости кожи — это ~0.1-0.5% DC-компоненты
    (~0.15-0.75 уровня из 256 при 8-битном АЦП). Пространственное
    усреднение по тысячам пикселей ROI снижает шум квантования в sqrt(N)
    раз, поднимая сигнал далеко выше шумового порога отдельного пикселя —
    но у МЕДИАНЫ пиксельных значений на целочисленной сетке нет этого
    преимущества: она сама квантована на ту же целочисленную сетку и
    отражает плавный сдвиг среднего на 0.3-0.5 уровня СТУПЕНЬКОЙ (меняется
    только когда пересекает целочисленную границу), уничтожая как раз тот
    непрерывный сигнал, ради сохранения которого делалось усреднение по
    площади. Плюс медиана асимптотически ~1.57x менее эффективна среднего
    для нормального шума (ARE = 2/pi). Усечённое среднее убирает выбросы
    (блики/тени/клиппинг сенсора) почти так же надёжно, как медиана, но
    остаётся НЕПРЕРЫВНОЙ функцией входа.

    trim_fraction — ФИКСИРОВАННАЯ доля пикселей, отбрасываемых с каждого
    конца по яркости (не зависит от яркостного порога клиппинга ниже) —
    иначе эффективная доля усечения плавает вместе с составом кадра, и
    возвращается ровно та же "смена состава выборки двигает оценку"
    проблема, только на другом уровне. Усечение общее для R/G/B (по рангу
    ЯРКОСТИ пикселя, не по рангу каждого канала отдельно) — так триплет
    BGR одного физического пикселя либо весь остаётся в выборке, либо весь
    из неё выбрасывается.

    Считается в float64 (не float32) — усечённое среднее по тысячам
    пикселей само по себе устойчиво к суммарной ошибке округления float32,
    но не даёт повода терять точность там, где она достаётся бесплатно.

    valid=False, если после исключения переэкспонированных/недоэкспонированных
    пикселей (типичный артефакт при бликах/тенях) валидных пикселей осталось
    меньше min_valid_fraction от площади ROI — тогда кадр не должен тянуть
    оценку ROI в сторону артефакта, а должен считаться отсутствующим
    (см. preprocessing.interpolate_missing).

    n_valid_pixels отдаётся наружу отдельно: резкое изменение состава ROI
    между кадрами (окклюзия, поворот головы) — самостоятельный повод
    пометить кадр невалидным, ДАЖЕ если min_valid_fraction формально
    пройдена (см. RPPGPipeline._check_valid_pixel_count_stability, задача 31).

    Задача 52: четвёртый элемент возврата — rejection_cause (None при
    valid=True), различающий ДВЕ разные причины valid=False, до этого
    слитые в один и тот же булев флаг и неразличимые в логе:
      "mask_empty"         — mask (обычно полигон ROI ∩ маска кожи, см.
                              extract_rois) пуст ЦЕЛИКОМ — для реальных
                              лицевых ROI геометрический полигон почти
                              никогда не пуст сам по себе (лицо уже
                              обнаружено), так что это практически всегда
                              означает "маска кожи отклонила весь полигон"
                              (например, чёлка перекрывает лоб — см.
                              extract_rois, где вызывающий код переименовывает
                              эту причину в "skin_mask_empty" для лицевых ROI).
      "brightness_clipping" — mask непустая, но доля пикселей, прошедших
                              порог отсечения блика/тени (10 < яркость <
                              250), не набрала min_valid_fraction —
                              освещение, а не маска кожи.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, False, 0, "mask_empty"

    pixels = frame_bgr[ys, xs].astype(np.float64)  # (K,3) BGR, float64
    brightness = pixels.mean(axis=1)
    good = (brightness > 10) & (brightness < 250)  # отсекаем клиппинг
    n_valid = int(good.sum())

    if n_valid < min_valid_fraction * len(ys):
        return None, False, n_valid, "brightness_clipping"

    good_pixels = pixels[good]
    good_brightness = brightness[good]
    order = np.argsort(good_brightness)
    trim_n = int(len(order) * trim_fraction)
    trimmed_idx = order[trim_n: len(order) - trim_n] if 2 * trim_n < len(order) else order

    mean_bgr = good_pixels[trimmed_idx].mean(axis=0)
    mean_rgb = mean_bgr[::-1].copy()  # BGR -> RGB
    return mean_rgb, True, n_valid, None


@dataclass
class ROIExtractionResult:
    rgb_by_roi: dict[str, np.ndarray | None]
    valid_by_roi: dict[str, bool]
    valid_pixel_count_by_roi: dict[str, int]  # задача 31 — для проверки стабильности состава ROI
    # Задача 52: причина valid_by_roi[name]=False на уровне ОДНОГО кадра
    # (None при valid=True) — "skin_mask_empty"/"brightness_clipping" из
    # roi_mean_rgb (переименовано из общего "mask_empty", см. extract_rois
    # ниже); "valid_pixel_count_unstable" добавляется ПОСЛЕ, вызывающим
    # кодом (RPPGPipeline.process_frame), если валидный по этой функции
    # кадр забраковала _check_valid_pixel_count_stability.
    rejection_cause_by_roi: dict[str, str | None]
    landmarks_px: np.ndarray
    stable_tracking_points: np.ndarray  # (len(STABLE_TRACKING_IDX), 2)
    interocular_distance_px: float  # п.19 — абсолютный масштаб лица
    background_rgb: np.ndarray | None  # п.22 — фон вне лица, для детектора мерцания
    background_valid: bool


def extract_rois(
    frame_bgr: np.ndarray,
    landmarks_norm: np.ndarray,
    roi_names: tuple[str, ...] = ("forehead", "left_cheek", "right_cheek"),
    shrink_factor: float = 0.9,
    min_valid_fraction: float = 0.6,
    trim_fraction: float = 0.10,
    skin_mask: np.ndarray | None = None,
) -> ROIExtractionResult:
    """skin_mask: если задан, используется вместо пересчёта build_skin_mask
    на каждый кадр (задача 34) — маска кожи, посчитанная по одному кадру,
    имеет ту же проблему "меняющегося состава выборки", что и медиана в
    задаче 31: шум/блик/лёгкий поворот головы меняют бинарное решение
    "кожа/не кожа" по краю маски кадр к кадру, и это никак не связано с
    пульсом. RPPGPipeline кэширует маску, посчитанную по медианному кадру
    последнего шага (см. RPPGPipeline._refresh_skin_mask_cache), и
    передаёт её сюда фиксированной на весь шаг. None (по умолчанию)
    сохраняет прежнее поведение "посчитать заново по этому кадру" — нужно
    для прямых юнит-тестов extract_rois без пайплайна."""
    landmarks_px = landmarks_to_pixels(landmarks_norm, frame_bgr.shape)
    if skin_mask is None:
        skin_mask = build_skin_mask(frame_bgr)

    rgb_by_roi: dict[str, np.ndarray | None] = {}
    valid_by_roi: dict[str, bool] = {}
    valid_pixel_count_by_roi: dict[str, int] = {}
    rejection_cause_by_roi: dict[str, str | None] = {}
    for name in roi_names:
        mask = build_roi_mask(landmarks_px, name, frame_bgr.shape, shrink_factor)
        mask = mask & skin_mask
        rgb, valid, n_valid, cause = roi_mean_rgb(frame_bgr, mask, min_valid_fraction, trim_fraction)
        rgb_by_roi[name] = rgb
        valid_by_roi[name] = valid
        valid_pixel_count_by_roi[name] = n_valid
        # Задача 52: здесь mask = полигон ROI ∩ маска кожи — геометрический
        # полигон сам по себе почти никогда не пуст (лицо уже обнаружено),
        # поэтому "mask_empty" для лицевого ROI практически всегда означает
        # именно "маска кожи отклонила весь полигон" — переименовываем для
        # точности лога (см. docstring roi_mean_rgb).
        rejection_cause_by_roi[name] = "skin_mask_empty" if cause == "mask_empty" else cause

    stable_points = landmarks_px[STABLE_TRACKING_IDX].astype(np.float64)
    ipd = interocular_distance_px(landmarks_px)

    background_rgb: np.ndarray | None = None
    background_valid = False
    bg_mask = build_background_roi_mask(frame_bgr.shape, landmarks_px)
    if bg_mask is not None:
        # Фон намеренно БЕЗ skin_mask — это не кожа, и порог кожи отсеял бы его целиком.
        background_rgb, background_valid, _, _ = roi_mean_rgb(frame_bgr, bg_mask, min_valid_fraction, trim_fraction)

    return ROIExtractionResult(
        rgb_by_roi=rgb_by_roi,
        valid_by_roi=valid_by_roi,
        valid_pixel_count_by_roi=valid_pixel_count_by_roi,
        rejection_cause_by_roi=rejection_cause_by_roi,
        landmarks_px=landmarks_px,
        stable_tracking_points=stable_points,
        interocular_distance_px=ipd,
        background_rgb=background_rgb,
        background_valid=background_valid,
    )
