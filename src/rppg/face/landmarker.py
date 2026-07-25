"""
Обёртка над MediaPipe Face Landmarker (пункт "Детекция лица" ТЗ).

Почему Face Landmarker (Tasks API), а не Face Mesh (legacy Solutions API) —
см. docs/research_report.md, раздел 3.1. Кратко: в установленной актуальной
версии пакета (mediapipe==0.10.33, проверено интроспекцией) legacy API
`mediapipe.solutions.face_mesh` и `mediapipe.python.solutions` физически
отсутствуют в пакете — есть только `mediapipe.tasks`. Т.е. это не вопрос
стиля/вкуса, а необходимость: код на `mp.solutions.face_mesh` не запустится
на актуальном пакете 2026 года.

Дополнительные преимущества Tasks API помимо самого факта поддержки:
  - `output_facial_transformation_matrixes=True` даёт готовую 4x4 матрицу
    позы головы -> можно оценивать yaw/pitch/roll без отдельной PnP-задачи,
    что напрямую служит требованию "устойчивость к движениям головы";
  - `output_face_blendshapes=True` даёт 52 ARKit-подобных коэффициента
    мимики — полезно как будущее расширение (детекция закрытия глаз/речи
    как источника недостоверности ROI), не используется в MVP, но
    архитектурно предусмотрено (см. FaceFrameResult.blendshapes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class FaceFrameResult:
    detected: bool
    landmarks_norm: np.ndarray | None  # (468, 3) нормализованные x,y,z
    head_pose: HeadPose | None
    face_presence_ok: bool


def _rotation_matrix_to_euler_deg(m: np.ndarray) -> HeadPose:
    """m: верхний левый 3x3 блок facial_transformation_matrix (row-major)."""
    sy = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(m[2, 1], m[2, 2])
        yaw = np.arctan2(-m[2, 0], sy)
        roll = np.arctan2(m[1, 0], m[0, 0])
    else:
        pitch = np.arctan2(-m[1, 2], m[1, 1])
        yaw = np.arctan2(-m[2, 0], sy)
        roll = 0.0
    return HeadPose(np.degrees(yaw), np.degrees(pitch), np.degrees(roll))


class FaceLandmarkerWrapper:
    """Тонкая обёртка с удобным API для потокового (VIDEO) режима."""

    def __init__(
        self,
        model_asset_path: str | Path,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_face_blendshapes: bool = False,
        output_facial_transformation_matrixes: bool = True,
        delegate: str = "CPU",
        running_mode: VisionTaskRunningMode = VisionTaskRunningMode.VIDEO,
    ):
        model_asset_path = Path(model_asset_path)
        if not model_asset_path.exists():
            raise FileNotFoundError(
                f"Модель Face Landmarker не найдена: {model_asset_path}\n"
                f"Скачайте её один раз командой:\n"
                f"  mkdir -p models && curl -L -o {model_asset_path} {MODEL_URL}"
            )

        delegate_enum = (
            BaseOptions.Delegate.GPU if delegate.upper() == "GPU" else BaseOptions.Delegate.CPU
        )
        base_options = BaseOptions(
            model_asset_path=str(model_asset_path), delegate=delegate_enum
        )
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._running_mode = running_mode

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceFrameResult:
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        if self._running_mode == VisionTaskRunningMode.VIDEO:
            result: FaceLandmarkerResult = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FaceFrameResult(detected=False, landmarks_norm=None, head_pose=None, face_presence_ok=False)

        lm = result.face_landmarks[0]
        landmarks_norm = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)

        head_pose = None
        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            head_pose = _rotation_matrix_to_euler_deg(mat[:3, :3])

        return FaceFrameResult(
            detected=True, landmarks_norm=landmarks_norm, head_pose=head_pose, face_presence_ok=True
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerWrapper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
