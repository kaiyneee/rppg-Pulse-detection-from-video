"""
Вырожденные входы для проверки поведения SQI (задача «сбор данных по
поведению SQI на вырожденных входах»).

Модуль НИЧЕГО не меняет в пайплайне: он только подменяет КАДРЫ на входе,
кадр за кадром, в момент чтения видео. Ни одна деградация не пишет
промежуточных видеофайлов на диск (11 условий x 50 записей x ~2 ГБ — это
сотня терабайт, которой нет и которая не нужна: деградация дешевле
декодирования).

КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ЧИСЛЕННОЙ ЧАСТИ. Реальная пульсовая модуляция
составляет 0.1-0.5% от постоянной составляющей яркости, то есть МЕНЬШЕ
одного младшего разряда 8-битного видео, и восстанавливается только
пространственным усреднением по тысячам пикселей. Отсюда два правила,
нарушение любого из которых превращает эксперимент в пустой прогон:

1. Модуляция накладывается в float32 и квантуется в uint8 ОДИН раз, в
   самом конце. Умножение uint8-кадра на (1 + 0.003*sin) округлится в
   тождественное преобразование: 0.3% от 128 — это 0.38 младшего разряда.
2. Мерцание накладывается на кадр, УЖЕ содержащий шум сенсора (условие
   A2). Шум работает дизерингом: без него квантование уничтожает
   субразрядную модуляцию даже при аккуратной арифметике, потому что
   округление одинаково для всех кадров. Поэтому B1w/B1f/B2f строятся
   поверх A2, а не поверх чистого замороженного кадра.

Проверка обоих правил — scripts/check_degrade_dither.py (там же показано,
что после квантования модуляция реально присутствует в среднем по ROI).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from rppg.face.roi import build_skin_mask

# Полный список условий. Порядок ЗАФИКСИРОВАН — в нём пишутся строки
# summary.csv, чтобы повторный прогон давал побитово тот же файл.
CONDITIONS: tuple[str, ...] = (
    "G0", "A1", "A2", "B1w", "B1f", "B2f", "B3", "B4", "C1", "C2", "D1",
)

# Условия, где ПОДЛИННЫЙ физиологический сигнал сохранён (эталон по
# контактной волне осмыслен и должен быть выгружен).
CONDITIONS_WITH_GENUINE_SIGNAL: frozenset[str] = frozenset({"G0", "B3", "C1", "C2", "D1"})

# Условия с НАЛОЖЕННОЙ периодичностью (заполняется bpm_injected).
CONDITIONS_WITH_INJECTION: frozenset[str] = frozenset({"B1w", "B1f", "B2f", "B3"})

# Условия, которым нужен замороженный медианный кадр записи.
CONDITIONS_FROZEN: frozenset[str] = frozenset({"A1", "A2", "B1w", "B1f", "B2f"})

# Параметры мерцания по условиям: (частота Гц | None -> берётся из эталона,
# амплитуда, только по лицу?).
FLICKER_SPECS: dict[str, tuple[float | None, float, bool]] = {
    "B1w": (1.2, 0.003, False),
    "B1f": (1.2, 0.003, True),
    "B2f": (1.2, 0.015, True),
    "B3": (None, 0.015, True),
}

# Ширина растушёвки маски лица, пиксели. Жёсткий край маски кожи — это
# ступенька, которой в реальной сцене не бывает: она сама по себе даёт
# широкополосный отклик на границе и делает условие нерепрезентативным.
FACE_MASK_FEATHER_PX: float = 18.0

# Шаг прореживания кадров при оценке медианного кадра и σ шума. Полная
# запись — 2000 кадров 640x480x3, то есть ~1.8 ГБ в памяти на один стек;
# каждый 10-й кадр даёт ~200 отсчётов на пиксель, чего более чем достаточно
# и для медианы, и для временно́го СКО.
PREP_SUBSAMPLE: int = 10

BLUR_SIGMA_PX: float = 8.0  # C1
LOOP_SECONDS: float = 3.0   # B4

# C2: H.264 с CRF 45. Кодировщик берётся из пакета imageio-ffmpeg (в
# системе ffmpeg нет) — это статический бинарник ffmpeg 7.1 с libx264.
H264_CRF: int = 45
# Число потоков x264 ЗАФИКСИРОВАНО: при автоопределении оно зависит от
# загрузки машины, а разбиение на слайсы между потоками меняет битовый
# результат кодирования — повторный прогон давал бы другой файл и, строго
# говоря, другие кадры (требование «побитово те же CSV»).
H264_THREADS: int = 4


def transcode_h264(video_path: str, out_path: str, crf: int = H264_CRF) -> str:
    """Условие C2: перекодировать запись в H.264 с заданным CRF.

    Кадры НЕ пересчитываются по времени (-fps_mode passthrough): число
    кадров и их порядок обязаны совпасть с исходной записью, иначе окна
    перестанут соответствовать другим условиям.
    """
    import subprocess

    import imageio_ffmpeg

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", video_path,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
        "-threads", str(H264_THREADS),
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg не смог перекодировать {video_path}: {proc.stderr.strip()[:500]}")
    return out_path


def condition_seed(record_id: str, condition: str) -> int:
    """Детерминированный seed на пару (запись, условие).

    zlib.crc32, а НЕ встроенный hash(): hash() строк рандомизируется
    PYTHONHASHSEED при каждом запуске интерпретатора, и повторный прогон
    давал бы другой шум — прямое нарушение требования «повторный запуск
    даёт побитово те же CSV».
    """
    return zlib.crc32(f"{record_id}|{condition}".encode("utf-8"))


def soft_face_mask(frame_bgr: np.ndarray, feather_px: float = FACE_MASK_FEATHER_PX) -> np.ndarray:
    """Маска кожи с мягким краем, float32 в [0, 1], форма (H, W, 1).

    Используется ШТАТНАЯ roi.build_skin_mask (YCrCb, Chai & Ngan) — та же
    маска, которой пользуется сам пайплайн; здесь она только растушёвывается.
    σ гауссиана = feather_px/3, поэтому переход занимает ~±3σ = feather_px
    пикселей.
    """
    mask = build_skin_mask(frame_bgr).astype(np.float32)
    sigma = max(feather_px / 3.0, 1e-6)
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(soft, 0.0, 1.0)[:, :, None]


@dataclass
class RecordPrep:
    """Предрассчитанные по ИСХОДНОЙ записи величины для деградаций."""

    median_frame: np.ndarray | None = None       # uint8, A1/A2/B1*/B2f
    noise_sigma: float | None = None             # A2 и всё, что поверх него
    face_mask: np.ndarray | None = None          # float32 (H,W,1), мерцание по лицу
    loop_frames: list[np.ndarray] = field(default_factory=list)  # B4
    n_frames_seen: int = 0


def prepare_record(
    video_path: str, condition: str, fps: float, subsample: int = PREP_SUBSAMPLE,
) -> RecordPrep:
    """Один проход по видео за тем, что нужно данному условию.

    Для условий, которым ничего не нужно (G0/B3/C1/C2/D1 — они работают с
    подлинными кадрами), проход не выполняется вовсе.
    """
    prep = RecordPrep()
    needs_frozen = condition in CONDITIONS_FROZEN
    needs_loop = condition == "B4"
    if not (needs_frozen or needs_loop):
        return prep

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео для подготовки: {video_path}")
    n_loop = int(round(LOOP_SECONDS * fps))
    sampled: list[np.ndarray] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        take_loop = needs_loop and idx < n_loop
        take_sample = needs_frozen and idx % subsample == 0
        if take_loop or take_sample:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if take_loop:
                prep.loop_frames.append(frame.copy())
            if take_sample:
                sampled.append(frame)
        idx += 1
    cap.release()
    prep.n_frames_seen = idx

    if needs_frozen:
        if not sampled:
            raise RuntimeError(f"нет кадров для медианного кадра: {video_path}")
        stack = np.stack(sampled).astype(np.float32)
        prep.median_frame = np.median(stack, axis=0).astype(np.uint8)
        # σ подбирается по ФАКТИЧЕСКОМУ временно́му СКО яркости пикселей
        # исходной записи: для каждого пикселя и канала берём СКО по
        # времени, затем медиану по всем пикселям. Медиана, а не среднее —
        # среднее утягивают вверх пиксели на границе лица и волос, где
        # временна́я дисперсия отражает движение головы, а не шум сенсора.
        prep.noise_sigma = float(np.median(np.std(stack, axis=0)))
        prep.face_mask = soft_face_mask(prep.median_frame)
    return prep


class FrameDegrader:
    """Деградация кадров на лету. Вызывается как degrader(frame, idx).

    Возвращает НОВЫЙ кадр uint8 той же формы. Число кадров и их временны́е
    метки деградация не меняет НИКОГДА — иначе окна перестали бы совпадать
    между условиями (критерий приёмки 3).
    """

    def __init__(
        self, condition: str, record_id: str, fps: float, prep: RecordPrep,
        injected_hz: float | None = None,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"неизвестное условие: {condition}")
        self.condition = condition
        self.fps = float(fps)
        self.prep = prep
        self.rng = np.random.default_rng(condition_seed(record_id, condition))
        self.flicker_hz: float | None = None
        self.flicker_amp: float = 0.0
        self.flicker_face_only: bool = False
        if condition in FLICKER_SPECS:
            spec_hz, amp, face_only = FLICKER_SPECS[condition]
            self.flicker_hz = spec_hz if spec_hz is not None else injected_hz
            if self.flicker_hz is None:
                raise ValueError(f"условию {condition} нужна частота наложения (injected_hz)")
            self.flicker_amp = amp
            self.flicker_face_only = face_only

    @property
    def injected_bpm(self) -> float | None:
        return None if self.flicker_hz is None else self.flicker_hz * 60.0

    def _base_frame_float(self, frame_bgr: np.ndarray, idx: int) -> np.ndarray:
        """Кадр-основа в float32 ДО наложения мерцания."""
        cond = self.condition
        if cond in CONDITIONS_FROZEN:
            base = self.prep.median_frame.astype(np.float32)
            if cond != "A1":
                # A2 и всё, что строится поверх него: гауссов шум сенсора.
                # Он же служит дизерингом для субразрядного мерцания ниже.
                base = base + self.rng.normal(0.0, self.prep.noise_sigma, base.shape).astype(np.float32)
            return base
        if cond == "B4":
            frames = self.prep.loop_frames
            return frames[idx % len(frames)].astype(np.float32)
        if cond == "C1":
            return cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=BLUR_SIGMA_PX, sigmaY=BLUR_SIGMA_PX).astype(np.float32)
        # G0, B3, C2, D1 — подлинный кадр. C2 подменяется на этапе чтения
        # (перекодированный файл), D1 — на этапе выбора ROI, не здесь.
        return frame_bgr.astype(np.float32)

    def __call__(self, frame_bgr: np.ndarray, idx: int) -> np.ndarray:
        base = self._base_frame_float(frame_bgr, idx)
        if self.flicker_hz is not None:
            t = idx / self.fps
            wave = self.flicker_amp * np.sin(2.0 * np.pi * self.flicker_hz * t)
            if self.flicker_face_only:
                # Для замороженных условий маска посчитана один раз по
                # медианному кадру; для B3 (подлинная запись, лицо движется)
                # маска пересчитывается покадрово.
                mask = self.prep.face_mask if self.condition in CONDITIONS_FROZEN else soft_face_mask(frame_bgr)
                base = base * (1.0 + wave * mask)
            else:
                base = base * (1.0 + wave)
        # Округление, а не усечение: astype отбрасывает дробную часть и
        # систематически занижал яркость на ~0.5 уровня (см. задание 2,
        # раздел «требует решения»). Реальное квантование в камере округляет.
        return np.clip(np.round(base), 0.0, 255.0).astype(np.uint8)
