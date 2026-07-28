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
          -> respiration.estimate_respiration_rate (по огибающей сигнала
             лучшего по SQI ROI, на каждом шаге)
          -> _update_ibi_log: суб-сэмпловые пики лучшего по SQI ROI
             пополняют ОТДЕЛЬНЫЙ накопитель IBI поверх перекрывающихся
             BPM-окон -> _maybe_compute_hrv публикует hrv.extract_hrv_features_from_ibi
             не чаще, чем раз в HRVConfig.step_seconds, и только когда
             накоплено >= HRVConfig.min_accumulation_seconds ряда IBI —
             BPM и HRV разнесены по окнам намеренно (см. HRVConfig).
          -> gate: если SQI ниже порога, PTSDPulseFeatures.publishable=False
             и BPM не должен использоваться потребителем дальше по системе;
             HRVFeatures.publishable — отдельный гейт по доле артефактных
             интервалов, который потребитель обязан проверять сам.

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
from rppg.signal.preprocessing import preprocess_signal, interpolate_missing, detrend
from rppg.signal.methods import get_method, SignalWindow
from rppg.signal.frequency import estimate_hr
from rppg.signal.quality import assess_quality, dominant_frequency_and_snr
from rppg.signal.respiration import estimate_respiration_rate
from rppg.hrv.features import (
    detect_pulse_peaks,
    refine_peaks_subsample,
    extract_hrv_features_from_ibi,
    HRVFeatures,
)


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
    # Частота дыхания по амплитудной модуляции пульсовой волны (п.18
    # требований) — считается на КАЖДОМ BPM-шаге (в отличие от hrv, который
    # копится и публикуется намного реже, см. HRVConfig).
    respiration_rate_bpm: float | None = None


class _RingBuffer:
    """Буфер скользящего окна, обрезаемый по РЕАЛЬНОМУ времени (timestamps_ms),
    а не по количеству кадров — иначе длина окна в секундах зависит от
    фактического fps камеры (см. WindowConfig.assumed_fps)."""

    def __init__(self):
        self.frames_bgr_ts: deque = deque()
        self.rgb_by_roi: dict[str, deque] = {}
        self.valid_by_roi: dict[str, deque] = {}
        self.landmark_traj: deque = deque()
        self.face_valid: deque = deque()  # True, если лицо было обнаружено в кадре
        self.timestamps_ms: deque = deque()

    def ensure_roi(self, roi_name: str) -> None:
        if roi_name not in self.rgb_by_roi:
            self.rgb_by_roi[roi_name] = deque()
            self.valid_by_roi[roi_name] = deque()

    def __len__(self) -> int:
        return len(self.timestamps_ms)

    def trim_older_than(self, window_seconds: float) -> None:
        if not self.timestamps_ms:
            return
        cutoff = self.timestamps_ms[-1] - window_seconds * 1000.0
        while self.timestamps_ms and self.timestamps_ms[0] < cutoff:
            self.timestamps_ms.popleft()
            self.landmark_traj.popleft()
            self.face_valid.popleft()
            for dq in self.rgb_by_roi.values():
                dq.popleft()
            for dq in self.valid_by_roi.values():
                dq.popleft()


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

        self._buf = _RingBuffer()
        for roi in self.cfg.roi.enabled_rois:
            self._buf.ensure_roi(roi.value)

        self._last_estimate_ms: int | None = None
        self._method = get_method(self.cfg.method.value)

        # Накопитель IBI для HRV — ОТДЕЛЬНЫЙ от скользящего BPM-окна (см.
        # HRVConfig и hrv/features.py, п.14 требований): пополняется
        # инкрементально из каждого перекрывающегося BPM-окна, хранит
        # (время_пика_мс, IBI_мс), обрезается по времени до
        # target_accumulation_seconds.
        self._ibi_log: deque = deque()
        self._last_peak_time_ms: float | None = None
        self._last_hrv_ms: int | None = None
        self._cached_hrv: HRVFeatures | None = None

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

        if len(self._buf) < 2:
            return None
        elapsed_s = (self._buf.timestamps_ms[-1] - self._buf.timestamps_ms[0]) / 1000.0
        if elapsed_s < self.cfg.window.min_seconds_before_estimate:
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
        self._buf.face_valid.append(True)
        for roi in self.cfg.roi.enabled_rois:
            name = roi.value
            rgb = roi_result.rgb_by_roi.get(name)
            valid = roi_result.valid_by_roi.get(name, False)
            self._buf.rgb_by_roi[name].append(rgb if rgb is not None else np.zeros(3))
            self._buf.valid_by_roi[name].append(bool(valid))
        self._buf.trim_older_than(self.cfg.window.window_seconds)

    def _push_invalid_frame(self, timestamp_ms: int) -> None:
        self._buf.timestamps_ms.append(timestamp_ms)
        n_stable = len(STABLE_TRACKING_IDX)
        last_traj = self._buf.landmark_traj[-1] if self._buf.landmark_traj else np.zeros((n_stable, 2))
        self._buf.landmark_traj.append(last_traj)  # держим позицию, не телепортируем в (0,0)
        self._buf.face_valid.append(False)
        for roi in self.cfg.roi.enabled_rois:
            name = roi.value
            self._buf.rgb_by_roi[name].append(np.zeros(3))
            self._buf.valid_by_roi[name].append(False)
        self._buf.trim_older_than(self.cfg.window.window_seconds)

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

    def _estimate_frequency(
        self,
        raw_signal: np.ndarray,
        valid_mask: np.ndarray,
        processed_signal: np.ndarray,
        fps: float,
        band: tuple[float, float],
        timestamps_sec: np.ndarray,
    ) -> float:
        """BPM по одному сигналу. Для lomb_scargle — отдельная ветка: LS
        работает по НЕинтерполированным валидным отсчётам и их реальным
        меткам времени (см. docstring frequency.py), а не по сигналу,
        уже "залеченному" interpolate_missing и пропущенному через
        filtfilt на равномерной сетке — иначе LS вырождается в обычную
        периодограмму на том же равномерном сигнале, что и FFT/Welch.
        """
        if self.cfg.frequency_method.value != "lomb_scargle":
            return estimate_hr(processed_signal, fps, band, method=self.cfg.frequency_method.value).bpm

        trend_removed = detrend(raw_signal, method=self.cfg.filt.detrend_method, lam=self.cfg.filt.tarvainen_lambda)
        ls_signal = trend_removed[valid_mask]
        ls_timestamps = timestamps_sec[valid_mask]
        return estimate_hr(ls_signal, fps, band, method="lomb_scargle", timestamps_sec=ls_timestamps).bpm

    def _estimate_color_roi(
        self,
        name: str,
        fps: float,
        band: tuple[float, float],
        timestamps_sec: np.ndarray,
        per_roi_bpm: dict[str, float],
        per_roi_signal: dict[str, np.ndarray],
        warnings: list[str],
    ) -> None:
        raw = np.array(self._buf.rgb_by_roi[name])
        valid_mask = np.array(self._buf.valid_by_roi[name], dtype=bool)

        fixed_channels = []
        ok = True
        for ch in range(3):
            fixed, ch_ok = interpolate_missing(raw[:, ch], valid_mask)
            fixed_channels.append(fixed)
            ok = ok and ch_ok
        rgb = np.stack(fixed_channels, axis=1)

        if not ok:
            warnings.append(f"ROI '{name}': слишком длинный провал трекинга/окклюзии в окне.")
            return

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
            return

        processed = preprocess_signal(
            raw_signal, fps, band[0], band[1],
            order=self.cfg.filt.filter_order,
            detrend_method=self.cfg.filt.detrend_method,
            tarvainen_lambda=self.cfg.filt.tarvainen_lambda,
            normalize_method=self.cfg.filt.normalize_method,
        )
        per_roi_signal[name] = processed
        per_roi_bpm[name] = self._estimate_frequency(raw_signal, valid_mask, processed, fps, band, timestamps_sec)

    def _estimate_head_motion(
        self,
        landmark_traj: np.ndarray,
        fps: float,
        band: tuple[float, float],
        timestamps_sec: np.ndarray,
        per_roi_bpm: dict[str, float],
        per_roi_signal: dict[str, np.ndarray],
        warnings: list[str],
    ) -> None:
        """HEAD_MOTION не зависит от ROI (он использует движение головы целиком),
        поэтому считается один раз на окно, а не по разу на каждый ROI —
        иначе cross_roi_agreement сравнивал бы три идентичных числа и всегда
        давал бы agreement = 1.0, искусственно завышая SQI."""
        face_valid = np.array(self._buf.face_valid, dtype=bool)
        if not face_valid.any():
            warnings.append("HEAD_MOTION: лицо не было обнаружено ни в одном кадре окна.")
            return

        window = SignalWindow(
            rgb_traces={},
            fps=fps,
            landmark_trajectories=landmark_traj,
            valid_mask=face_valid,
            hr_band_hz=band,
        )
        try:
            raw_signal = self._method.extract(window, None)
        except Exception as exc:  # noqa: BLE001 - деградация вместо падения пайплайна
            warnings.append(f"HEAD_MOTION: ошибка метода извлечения ({exc}).")
            return

        processed = preprocess_signal(
            raw_signal, fps, band[0], band[1],
            order=self.cfg.filt.filter_order,
            detrend_method=self.cfg.filt.detrend_method,
            tarvainen_lambda=self.cfg.filt.tarvainen_lambda,
            normalize_method=self.cfg.filt.normalize_method,
        )
        per_roi_signal["face"] = processed
        per_roi_bpm["face"] = self._estimate_frequency(raw_signal, face_valid, processed, fps, band, timestamps_sec)

    # ------------------------------------------------------------------ #
    def _update_ibi_log(self, signal: np.ndarray, fps: float, timestamps_ms) -> None:
        """Инкрементально пополняет накопитель IBI новыми (ранее не
        учтёнными) пиками из текущего скользящего BPM-окна.

        Окна перекрываются (шаг WindowConfig.step_seconds << window_seconds),
        поэтому один и тот же пик виден в нескольких последовательных
        вызовах — дедуплицируем по абсолютному времени последнего уже
        учтённого пика (self._last_peak_time_ms), а не по индексу в массиве
        (индексы каждый раз разные, т.к. окно скользит). Так по множеству
        коротких перекрывающихся BPM-окон строится один непрерывный ряд IBI
        длиной в минуты — то, что реально нужно для SDNN/RMSSD/LF/HF
        (см. HRVConfig, п.14 требований).
        """
        ts = np.asarray(timestamps_ms, dtype=float)
        n = len(signal)
        trim = int(round(self.cfg.hrv.edge_trim_seconds * fps))
        min_len = 2 * trim + int(round(fps * 2))
        if n <= min_len:
            return  # окно ещё недостаточно длинное, чтобы безопасно (без краевых артефактов) искать пики

        lo, hi = trim, n - trim
        interior = signal[lo:hi]
        peaks = detect_pulse_peaks(interior, fps)
        if len(peaks) == 0:
            return
        peaks_sub = refine_peaks_subsample(interior, peaks) + lo

        idx = np.arange(n, dtype=float)
        peak_times_ms = np.interp(peaks_sub, idx, ts)

        min_ibi_ms = 60000.0 / 220.0
        max_ibi_ms = 60000.0 / 40.0
        max_gap_ms = 3000.0  # больше этого — разрыв непрерывности (окклюзия), не IBI

        for pt in peak_times_ms:
            pt = float(pt)
            if self._last_peak_time_ms is None:
                self._last_peak_time_ms = pt
                continue
            gap = pt - self._last_peak_time_ms
            if gap <= 1e-6:
                continue  # уже учтён на предыдущем шаге (окна перекрываются)
            self._last_peak_time_ms = pt
            if gap > max_gap_ms:
                continue
            if min_ibi_ms <= gap <= max_ibi_ms:
                self._ibi_log.append((pt, gap))

        self._trim_ibi_log()

    def _trim_ibi_log(self) -> None:
        if not self._ibi_log:
            return
        cutoff = self._ibi_log[-1][0] - self.cfg.hrv.target_accumulation_seconds * 1000.0
        while self._ibi_log and self._ibi_log[0][0] < cutoff:
            self._ibi_log.popleft()

    def _maybe_compute_hrv(
        self, timestamp_ms: int, pulse_signal_quality_score: float, warnings: list[str]
    ) -> HRVFeatures | None:
        accumulated_seconds = 0.0
        if len(self._ibi_log) >= 2:
            accumulated_seconds = (self._ibi_log[-1][0] - self._ibi_log[0][0]) / 1000.0

        if accumulated_seconds < self.cfg.hrv.min_accumulation_seconds:
            if self._cached_hrv is None:
                warnings.append(
                    f"HRV ещё не накоплен: {accumulated_seconds:.0f}с из минимум "
                    f"{self.cfg.hrv.min_accumulation_seconds:.0f}с ряда межпульсовых "
                    "интервалов (см. HRVConfig.min_accumulation_seconds)."
                )
            return self._cached_hrv

        if (
            self._last_hrv_ms is not None
            and timestamp_ms - self._last_hrv_ms < self.cfg.hrv.step_seconds * 1000
        ):
            return self._cached_hrv

        ibi_ms = np.array([ibi for _, ibi in self._ibi_log])
        hrv = extract_hrv_features_from_ibi(
            ibi_ms,
            pulse_signal_quality_score=pulse_signal_quality_score,
            pnn_threshold_ms=self.cfg.hrv.pnn_threshold_ms,
            pnn20_threshold_ms=self.cfg.hrv.pnn20_threshold_ms,
            compute_frequency_domain=self.cfg.hrv.compute_frequency_domain,
            max_artifact_fraction=self.cfg.hrv.max_artifact_fraction,
            ectopic_max_relative_change=self.cfg.hrv.ectopic_max_relative_change,
            lf_min_duration_seconds=self.cfg.hrv.lf_min_duration_seconds,
            hf_min_duration_seconds=self.cfg.hrv.hf_min_duration_seconds,
        )
        self._last_hrv_ms = timestamp_ms
        self._cached_hrv = hrv
        return hrv

    def _compute_estimate(self, timestamp_ms: int) -> PTSDPulseFeatures:
        fps = self._estimate_fps()
        band = (self.cfg.filt.low_hz, self.cfg.filt.high_hz)
        timestamps_sec = np.array(self._buf.timestamps_ms, dtype=float) / 1000.0
        landmark_traj = np.array(self._buf.landmark_traj)

        per_roi_bpm: dict[str, float] = {}
        per_roi_signal: dict[str, np.ndarray] = {}
        warnings: list[str] = []

        if self._method.name == "head_motion":
            self._estimate_head_motion(landmark_traj, fps, band, timestamps_sec, per_roi_bpm, per_roi_signal, warnings)
        else:
            for roi in self.cfg.roi.enabled_rois:
                self._estimate_color_roi(roi.value, fps, band, timestamps_sec, per_roi_bpm, per_roi_signal, warnings)

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

        sqi = assess_quality(
            spectral_snr_db=best_snr_db,
            bpm_by_roi=per_roi_bpm,
            landmark_trajectories=landmark_traj,
            min_spectral_snr_db=self.cfg.quality.min_spectral_snr_db,
            max_cross_roi_bpm_diff=self.cfg.quality.max_cross_roi_bpm_diff,
            min_landmark_stability=self.cfg.quality.min_landmark_stability,
            min_overall_score_to_publish=self.cfg.quality.min_overall_score_to_publish,
        )

        self._update_ibi_log(best_signal, fps, self._buf.timestamps_ms)
        hrv_warnings: list[str] = []
        hrv = self._maybe_compute_hrv(timestamp_ms, sqi.overall_score, hrv_warnings)

        respiration_rate_bpm, _ = estimate_respiration_rate(best_signal, fps)

        return PTSDPulseFeatures(
            timestamp_ms=timestamp_ms,
            bpm=per_roi_bpm.get(best_roi, float("nan")),
            hrv=hrv,
            sqi_score=sqi.overall_score,
            sqi_level=sqi.level.value,
            publishable=sqi.is_reliable,
            warnings=warnings + sqi.warnings + hrv_warnings + (hrv.warnings if hrv is not None else []),
            method_used=self._method.name,
            frequency_method_used=self.cfg.frequency_method.value,
            per_roi_bpm=per_roi_bpm,
            respiration_rate_bpm=respiration_rate_bpm,
        )
