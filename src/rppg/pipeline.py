"""
RPPGPipeline — верхнеуровневая оркестрация всего модуля.

Поток данных на каждый кадр:

    frame -> FaceLandmarkerWrapper.detect
          -> roi.extract_rois (лоб/щёки, с occlusion-гейтом на пиксельном уровне)
          -> буферизация в скользящее окно (WindowConfig)
          -> [раз в step_seconds] для каждого валидного ROI:
                 preprocessing.preprocess_signal
                 methods.<Extraction>.extract
                 frequency.estimate_hr
          -> quality.assess_quality (спектральный SNR + межзонное согласие
             + стабильность landmark-точек)
          -> hrv.extract_hrv_features (на сигнале лучшего по SQI ROI)
          -> gate: если SQI ниже порога, PTSDPulseFeatures.publishable=False
             и BPM/HRV не должны использоваться потребителем дальше по системе.

Это единственное место, где реализовано жёсткое требование ТЗ:
"если качество сигнала низкое — не передавать значение BPM в систему ПТСР".
Мы не удаляем результат физически (это усложнило бы отладку/логирование),
а помечаем его `publishable=False`; интеграция с системой ПТСР обязана
проверять именно этот флаг, а не просто наличие числа в bpm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from rppg.config import PipelineConfig
from rppg.face.landmarker import FaceLandmarkerWrapper
from rppg.face.roi import extract_rois, STABLE_TRACKING_IDX
from rppg.signal.preprocessing import preprocess_signal, interpolate_missing
from rppg.signal.methods import get_method, SignalWindow
from rppg.signal.frequency import estimate_hr
from rppg.signal.quality import assess_quality, dominant_frequency_and_snr
from rppg.hrv.features import extract_hrv_features, HRVFeatures


@dataclass
class PTSDPulseFeatures:
    """Итоговая структура, которая уходит в систему анализа ПТСР."""

    timestamp_ms: int
    bpm: float
    hrv: HRVFeatures | None
    sqi_score: float
    sqi_level: str
    publishable: bool
    warnings: list[str] = field(default_factory=list)
    method_used: str = ""
    frequency_method_used: str = ""
    per_roi_bpm: dict[str, float] = field(default_factory=dict)


class _RingBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.frames_bgr_ts: deque = deque(maxlen=maxlen)
        self.rgb_by_roi: dict[str, deque] = {}
        self.valid_by_roi: dict[str, deque] = {}
        self.landmark_traj: deque = deque(maxlen=maxlen)
        self.timestamps_ms: deque = deque(maxlen=maxlen)

    def ensure_roi(self, roi_name: str) -> None:
        if roi_name not in self.rgb_by_roi:
            self.rgb_by_roi[roi_name] = deque(maxlen=self.maxlen)
            self.valid_by_roi[roi_name] = deque(maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self.timestamps_ms)


class RPPGPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self._landmarker = FaceLandmarkerWrapper(
            model_asset_path=self.cfg.face.model_asset_path,
            num_faces=self.cfg.face.num_faces,
            min_face_detection_confidence=self.cfg.face.min_face_detection_confidence,
            min_face_presence_confidence=self.cfg.face.min_face_presence_confidence,
            min_tracking_confidence=self.cfg.face.min_tracking_confidence,
            output_face_blendshapes=self.cfg.face.output_face_blendshapes,
            output_facial_transformation_matrixes=self.cfg.face.output_facial_transformation_matrixes,
            delegate=self.cfg.face.delegate,
        )

        maxlen = int(round(self.cfg.window.window_seconds * self.cfg.window.assumed_fps)) + 5
        self._buf = _RingBuffer(maxlen=maxlen)
        for roi in self.cfg.roi.enabled_rois:
            self._buf.ensure_roi(roi.value)

        self._last_estimate_ms: int | None = None
        self._method = get_method(self.cfg.method.value)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "RPPGPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PTSDPulseFeatures | None:
        face = self._landmarker.detect(frame_bgr, timestamp_ms)

        if not face.detected:
            self._push_invalid_frame(timestamp_ms)
        else:
            roi_result = extract_rois(
                frame_bgr,
                face.landmarks_norm,
                roi_names=tuple(r.value for r in self.cfg.roi.enabled_rois),
                shrink_factor=self.cfg.roi.shrink_factor,
                min_valid_fraction=self.cfg.roi.min_valid_pixel_fraction,
            )
            self._push_frame(timestamp_ms, roi_result)

        if len(self._buf) < self.cfg.window.min_seconds_before_estimate * self.cfg.window.assumed_fps:
            return None

        if (
            self._last_estimate_ms is not None
            and timestamp_ms - self._last_estimate_ms < self.cfg.window.step_seconds * 1000
        ):
            return None

        self._last_estimate_ms = timestamp_ms
        return self._compute_estimate(timestamp_ms)

    # ------------------------------------------------------------------ #
    def _push_frame(self, timestamp_ms: int, roi_result) -> None:
        self._buf.timestamps_ms.append(timestamp_ms)
        self._buf.landmark_traj.append(roi_result.stable_tracking_points)
        for roi in self.cfg.roi.enabled_rois:
            name = roi.value
            rgb = roi_result.rgb_by_roi.get(name)
            valid = roi_result.valid_by_roi.get(name, False)
            self._buf.rgb_by_roi[name].append(rgb if rgb is not None else np.zeros(3))
            self._buf.valid_by_roi[name].append(bool(valid))

    def _push_invalid_frame(self, timestamp_ms: int) -> None:
        self._buf.timestamps_ms.append(timestamp_ms)
        n_stable = len(STABLE_TRACKING_IDX)
        last_traj = self._buf.landmark_traj[-1] if self._buf.landmark_traj else np.zeros((n_stable, 2))
        self._buf.landmark_traj.append(last_traj)  # держим позицию, не телепортируем в (0,0)
        for roi in self.cfg.roi.enabled_rois:
            name = roi.value
            self._buf.rgb_by_roi[name].append(np.zeros(3))
            self._buf.valid_by_roi[name].append(False)

    # ------------------------------------------------------------------ #
    def _estimate_fps(self) -> float:
        ts = np.array(self._buf.timestamps_ms, dtype=float)
        if len(ts) < 2:
            return self.cfg.window.assumed_fps
        dt = np.diff(ts) / 1000.0
        dt = dt[dt > 1e-3]
        if len(dt) == 0:
            return self.cfg.window.assumed_fps
        return float(1.0 / np.median(dt))

    def _compute_estimate(self, timestamp_ms: int) -> PTSDPulseFeatures:
        fps = self._estimate_fps()
        band = (self.cfg.filt.low_hz, self.cfg.filt.high_hz)

        rgb_traces: dict[str, np.ndarray] = {}
        per_roi_bpm: dict[str, float] = {}
        per_roi_signal: dict[str, np.ndarray] = {}
        warnings: list[str] = []

        for roi in self.cfg.roi.enabled_rois:
            name = roi.value
            raw = np.array(self._buf.rgb_by_roi[name])
            valid_mask = np.array(self._buf.valid_by_roi[name], dtype=bool)

            fixed_channels = []
            ok = True
            for ch in range(3):
                fixed, ch_ok = interpolate_missing(raw[:, ch], valid_mask)
                fixed_channels.append(fixed)
                ok = ok and ch_ok
            rgb = np.stack(fixed_channels, axis=1)
            rgb_traces[name] = rgb

            if not ok:
                warnings.append(f"ROI '{name}': слишком длинный провал трекинга/окклюзии в окне.")
                continue

            window = SignalWindow(
                rgb_traces={name: rgb},
                fps=fps,
                landmark_trajectories=None,
                valid_mask=valid_mask,
                hr_band_hz=band,
            )
            try:
                raw_signal = self._method.extract(window, name)
            except Exception as exc:  # noqa: BLE001 - деградация вместо падения пайплайна
                warnings.append(f"ROI '{name}': ошибка метода извлечения ({exc}).")
                continue

            processed = preprocess_signal(
                raw_signal, fps, band[0], band[1],
                order=self.cfg.filt.filter_order,
                detrend_method=self.cfg.filt.detrend_method,
                tarvainen_lambda=self.cfg.filt.tarvainen_lambda,
                normalize_method=self.cfg.filt.normalize_method,
            )
            per_roi_signal[name] = processed

            freq_est = estimate_hr(processed, fps, band, method=self.cfg.frequency_method.value)
            per_roi_bpm[name] = freq_est.bpm

        if not per_roi_signal:
            return PTSDPulseFeatures(
                timestamp_ms=timestamp_ms, bpm=float("nan"), hrv=None,
                sqi_score=0.0, sqi_level="low", publishable=False,
                warnings=warnings + ["Ни один ROI не дал валидного сигнала в этом окне."],
                method_used=self._method.name,
                frequency_method_used=self.cfg.frequency_method.value,
                per_roi_bpm=per_roi_bpm,
            )

        # ROI с максимальным spectral SNR используется как основной источник BPM/HRV.
        best_roi = max(
            per_roi_signal,
            key=lambda n: dominant_frequency_and_snr(per_roi_signal[n], fps, band)[1],
        )
        best_signal = per_roi_signal[best_roi]
        _, best_snr_db = dominant_frequency_and_snr(best_signal, fps, band)

        landmark_traj = np.array(self._buf.landmark_traj)

        sqi = assess_quality(
            spectral_snr_db=best_snr_db,
            bpm_by_roi=per_roi_bpm,
            landmark_trajectories=landmark_traj,
            min_spectral_snr_db=self.cfg.quality.min_spectral_snr_db,
            max_cross_roi_bpm_diff=self.cfg.quality.max_cross_roi_bpm_diff,
            min_landmark_stability=self.cfg.quality.min_landmark_stability,
            min_overall_score_to_publish=self.cfg.quality.min_overall_score_to_publish,
        )

        hrv = extract_hrv_features(
            best_signal, fps,
            pulse_signal_quality_score=sqi.overall_score,
            pnn_threshold_ms=self.cfg.hrv.pnn_threshold_ms,
            pnn20_threshold_ms=self.cfg.hrv.pnn20_threshold_ms,
            compute_frequency_domain=self.cfg.hrv.compute_frequency_domain,
        )

        return PTSDPulseFeatures(
            timestamp_ms=timestamp_ms,
            bpm=per_roi_bpm.get(best_roi, float("nan")),
            hrv=hrv,
            sqi_score=sqi.overall_score,
            sqi_level=sqi.level.value,
            publishable=sqi.is_reliable,
            warnings=warnings + sqi.warnings + hrv.warnings,
            method_used=self._method.name,
            frequency_method_used=self.cfg.frequency_method.value,
            per_roi_bpm=per_roi_bpm,
        )
