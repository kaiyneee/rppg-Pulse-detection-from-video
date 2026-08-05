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
          -> quality.assess_quality (спектральный SNR со штрафом за
             гармоники/субгармоники + межзонное согласие + стабильность
             landmark-точек, нормированная на межзрачковое расстояние +
             temporal consistency BPM между соседними окнами; жёсткий гейт
             дополнительно проверяет фоновый ROI на мерцание освещения)
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
from pathlib import Path

import numpy as np

from rppg.config import PipelineConfig, ExtractionMethod, QualityConfig
from rppg.face.landmarker import FaceLandmarkerWrapper
from rppg.face.roi import extract_rois, ROIExtractionResult, STABLE_TRACKING_IDX
from rppg.signal.preprocessing import preprocess_signal, interpolate_missing, detrend, is_gap_acceptable, fill_gaps
from rppg.signal.methods import get_method, SignalWindow
from rppg.signal.frequency import estimate_hr
from rppg.signal.quality import assess_quality, dominant_frequency_and_snr, compute_psd, SQIInputs, SQIResult
from rppg.signal.fusion import fuse_signals_by_sqi, snr_db_to_weight
from rppg.signal.respiration import estimate_respiration_rate
from rppg.structured_log import StructuredWindowLogger, WindowLogRecord
from rppg.hrv.features import (
    detect_pulse_peaks,
    refine_peaks_subsample,
    extract_hrv_features_from_ibi,
    HRVFeatures,
)


@dataclass
class DebugInfo:
    """Данные для отладочного оверлея (scripts/run_webcam.py --debug),
    п.44 требований: "глядя на экран 10с, видно, какая компонента мешает
    публикации прямо сейчас, без логов".

    Собирается ТОЛЬКО когда RPPGPipeline создан с debug=True — Welch PSD
    здесь пересчитывать не нужно (est.assess_quality её уже считает
    внутри dominant_frequency_and_snr), но ХРАНИТЬ freqs/psd массив на
    каждый шаг ради оверлея, который никто не смотрит в обычном режиме,
    было бы неоправданным по памяти для длинных офлайн-прогонов видео."""

    sqi: SQIResult
    qcfg: QualityConfig
    band_hz: tuple[float, float]
    best_source: str
    peak_freq_hz: float
    psd_freqs: np.ndarray | None
    psd_values: np.ndarray | None


@dataclass
class PTSDPulseFeatures:
    """Итоговая структура, которая уходит в систему анализа ПТСР."""

    timestamp_ms: int
    # Задача 11: заполняется ВСЕГДА, когда было хоть одно валидное окно
    # (т.е. хотя бы один ROI дал сигнал) — bpm приходит НАПРЯМУЮ из спектра
    # этого окна (см. _compute_estimate), независимо от confidence/
    # publishable. NaN — ТОЛЬКО когда валидных окон не было вообще
    # (status="no_signal" ниже), никогда как "тихая" замена низкой
    # уверенности. Другими словами: bpm отвечает на вопрос "что показал
    # спектр", а publishable/confidence — на вопрос "можно ли этому верить".
    bpm: float
    hrv: HRVFeatures | None
    sqi_score: float
    sqi_level: str
    # Задача 11: ПРОИЗВОДНЫЙ удобный флаг для потребителя, который не хочет
    # разбираться в компонентах SQI: publishable = confidence >=
    # QualityConfig.min_overall_score_to_publish — простое пороговое
    # правило, НЕ единственный интерфейс. В частности, ОНО НЕ учитывает
    # flicker_suspected и жёсткий пол по spectral_snr_db по отдельности
    # (см. confidence_breakdown ниже) — тот более строгий рейл, который
    # структурный JSONL-лог (structured_log.py) логирует под тем же именем
    # "publishable", здесь СОЗНАТЕЛЬНО не воспроизведён 1:1: цель этого поля
    # — дать разумный дефолт с порогом "из коробки", а не заменить собой
    # анализ confidence_breakdown для тех, кому нужна строгая семантика
    # (см. пример строгого правила в docstring confidence_breakdown).
    publishable: bool
    # Задача 11: непрерывная (НЕ бинарная) оценка доверия в [0, 1] — то же
    # самое SQIResult.overall_score, что и sqi_score выше (после
    # перекалибровки компонент, см. задачу 13), но ВСЕГДА определена: 0.0,
    # если валидных окон не было вообще (status="no_signal"), а не просто
    # "берётся откуда-то ещё". sqi_score оставлен для обратной совместимости
    # (то же значение) — confidence самим именем прямо говорит, что это
    # ответ на вопрос "насколько доверять", а не сырой служебный термин SQI.
    confidence: float = 0.0
    # Задача 11: ВСЕ компоненты SQI по отдельности (в дополнение к
    # свёрнутому confidence) — чтобы потребитель мог построить СВОЁ правило,
    # а не зависеть от того, как именно устроен publishable выше. Пример
    # более строгого правила (эквивалент старого SQIResult.is_reliable):
    #   confidence >= cfg.quality.min_overall_score_to_publish
    #   and breakdown["spectral_snr_db"] >= cfg.quality.min_spectral_snr_db
    #   and breakdown["flicker_suspected"] < 0.5
    # Пустой словарь при status="no_signal" (компоненты SQI не считались).
    confidence_breakdown: dict[str, float] = field(default_factory=dict)
    # Задача 11: "ok" — было валидное окно (пусть и с низким confidence);
    # "no_signal" — ни один ROI не дал сигнала в этом окне вообще (см.
    # _compute_estimate). Отдельно от publishable/confidence: "низкая
    # уверенность в реальном измерении" (status="ok", confidence=0.1) и
    # "измерения не было" (status="no_signal", confidence=0.0) — РАЗНЫЕ
    # ситуации, которые нельзя различить по одному только confidence=0..0.1,
    # если не смотреть status отдельно.
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    method_used: str = ""
    frequency_method_used: str = ""
    per_roi_bpm: dict[str, float] = field(default_factory=dict)
    # Частота дыхания по амплитудной модуляции пульсовой волны (п.18
    # требований) — считается на КАЖДОМ BPM-шаге (в отличие от hrv, который
    # копится и публикуется намного реже, см. HRVConfig).
    respiration_rate_bpm: float | None = None
    # Задача 8, вариант А: раньше здесь было best_effort_bpm — сглаженное
    # значение, которое ВСЕГДА было реальным числом в правдоподобном
    # диапазоне, даже при полном отсутствии сигнала (стартовало с
    # fallback_bpm=75 и медленно "уплывало" на шуме). Проблема не в точности
    # — это было НЕ измерение, а генератор правдоподобного числа человеческого
    # пульса по требованию, что на экране/в интеграции неотличимо от
    # настоящего измерения. См. историю в git log за "best_effort_bpm" и
    # обсуждение в задаче 8.
    #
    # Вместо этого наружу отдаётся ЧЕСТНАЯ пара: последнее РЕАЛЬНО
    # измеренное (т.е. publishable=True, прошедшее SQI-гейт — см.
    # _compute_estimate) значение BPM и его возраст в секундах. Оба None,
    # если ни одного publishable-окна ещё не было. Принимающая система
    # получает значение на каждом шаге (как и раньше), но решение "не
    # слишком ли оно устарело" принимает САМА по last_valid_bpm_age_s, а не
    # получает фальшивое число, замаскированное под непрерывный поток данных.
    last_valid_bpm: float | None = None
    last_valid_bpm_age_s: float | None = None
    # Только при RPPGPipeline(debug=True) — см. DebugInfo. None в обычном
    # режиме, чтобы не тащить лишние массивы/объекты по умолчанию.
    debug: DebugInfo | None = None


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
        # п.19: масштаб лица (межзрачковое/межугловое расстояние) за кадр,
        # для абсолютной нормировки джиттера в landmark_stability_score.
        self.interocular_dist: deque = deque()
        # п.22: фоновый ROI (вне лица), для детектора мерцания освещения.
        self.background_rgb: deque = deque()
        self.background_valid: deque = deque()

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
            self.interocular_dist.popleft()
            self.background_rgb.popleft()
            self.background_valid.popleft()
            for dq in self.rgb_by_roi.values():
                dq.popleft()
            for dq in self.valid_by_roi.values():
                dq.popleft()


class RPPGPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        log_path: str | Path | None = None,
        debug: bool = False,
    ):
        """log_path: если задан, каждое оценённое окно дополнительно
        пишется структурированной JSONL-строкой (см. structured_log.py,
        п.43 требований) — timestamp, BPM по каждому ROI, ВСЕ компоненты
        SQI по отдельности, warnings. Опционально: без log_path поведение
        не отличается от прежнего.

        debug: если True, PTSDPulseFeatures.debug и last_roi_result
        дополнительно заполняются (полный SQIResult, Welch-спектр лучшего
        источника, landmark-точки текущего кадра) — материал для
        отладочного оверлея scripts/run_webcam.py --debug (п.44
        требований). По умолчанию False — эти данные никому не нужны в
        обычном режиме и стоят лишних вычислений/памяти."""
        self.cfg = config or PipelineConfig()
        self._debug = debug
        self._window_logger = StructuredWindowLogger(log_path) if log_path is not None else None
        # Только для --debug оверлея: полный ROIExtractionResult ТЕКУЩЕГО
        # кадра (landmarks_px и т.п.), чтобы отрисовать контуры ROI на
        # экране КАЖДЫЙ кадр — независимо от того, готова ли уже
        # BPM-оценка (см. WindowConfig.step_seconds, она реже кадров).
        self.last_roi_result: ROIExtractionResult | None = None
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
        # AccelerationConfig.use_numba раньше был объявлен в конфиге, но
        # никуда не передавался — get_method() вызывался без kwargs, и
        # PosMethod всегда получал свой дефолт use_numba=True независимо от
        # конфига (п.32: без этого сравнение "с Numba и без" через
        # PipelineConfig было бы невозможно сделать, не трогая код). kwarg
        # передаётся только методам, которые его принимают (сейчас — только
        # POS; GreenMethod/ChromMethod/... не имеют параметров __init__).
        self._auto_mode = self.cfg.method == ExtractionMethod.AUTO
        if self._auto_mode:
            # Адаптация к разным камерам: не фиксируем один метод, а
            # держим ВСЕ 5 цветовых методов наготове — какой из них
            # использовать в конкретном окне решает _select_auto_method
            # (см. _compute_estimate) заново каждое окно по измеренному
            # SNR. HEAD_MOTION сюда не входит — это отдельная
            # (motion-based, не цветовая) модальность, за неё отвечает
            # FusionConfig.include_head_motion, а не AUTO.
            self._auto_candidate_methods: dict[str, object] = {
                name: get_method(name, **({"use_numba": self.cfg.accel.use_numba} if name == "pos" else {}))
                for name in ("green", "chrom", "pos", "pca", "ica")
            }
            self._method = self._auto_candidate_methods["pos"]  # дефолт до первого окна
        else:
            self._auto_candidate_methods = {}
            method_kwargs = {}
            if self.cfg.method == ExtractionMethod.POS:
                method_kwargs["use_numba"] = self.cfg.accel.use_numba
            self._method = get_method(self.cfg.method.value, **method_kwargs)

        # п.20: история BPM лучшего ROI по последним оценкам (перекрывающиеся
        # окна) — для temporal_consistency_score. maxlen соответствует
        # нескольким window.step_seconds шагам назад; не привязана к
        # window.window_seconds, т.к. это отдельная, более короткая шкала.
        self._recent_bpm_history: deque = deque(maxlen=6)

        # Задача 8: последнее РЕАЛЬНО измеренное (publishable=True) значение
        # BPM и время его получения — см. PTSDPulseFeatures.last_valid_bpm/
        # last_valid_bpm_age_s. Оба None, пока не было ни одного publishable
        # окна (никакого fallback-числа, в отличие от прежнего best_effort_bpm).
        self._last_valid_bpm: float | None = None
        self._last_valid_bpm_timestamp_ms: int | None = None

        # Задача 7: история ОДНОконных candidate-флагов detect_illumination_
        # flicker с предыдущих окон — требование устойчивости во времени
        # (см. quality.assess_quality, SQIInputs.recent_flicker_candidates).
        # maxlen с запасом над QualityConfig.flicker_min_consecutive_windows,
        # чтобы смена конфига не требовала менять этот buffer.
        self._recent_flicker_candidates: deque = deque(maxlen=10)

        # Накопитель IBI для HRV — ОТДЕЛЬНЫЙ от скользящего BPM-окна (см.
        # HRVConfig и hrv/features.py, п.14 требований): пополняется
        # инкрементально из каждого перекрывающегося BPM-окна, хранит
        # (время_пика_мс, IBI_мс), обрезается по времени до
        # target_accumulation_seconds.
        self._ibi_log: deque = deque()
        self._last_peak_time_ms: float | None = None
        self._last_hrv_ms: int | None = None
        self._cached_hrv: HRVFeatures | None = None

    @property
    def buffered_seconds(self) -> float:
        """Сколько секунд реального времени сейчас накоплено в скользящем
        окне — п.6 требований (задача 6): с тех пор как
        min_seconds_before_estimate по умолчанию равен window_seconds,
        process_frame() возвращает None (без обратной связи) всю
        буферизацию/прогрев целиком. Это единственный способ для UI
        (scripts/run_webcam.py) показать прогресс ("буферизация 6.2 / 10.0
        с") вместо того, чтобы отсутствие числа в первые секунды выглядело
        как поломка."""
        if len(self._buf.timestamps_ms) < 2:
            return 0.0
        return (self._buf.timestamps_ms[-1] - self._buf.timestamps_ms[0]) / 1000.0

    def close(self) -> None:
        self._landmarker.close()
        if self._window_logger is not None:
            self._window_logger.close()

    def __enter__(self) -> "RPPGPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PTSDPulseFeatures | None:
        face = self._landmarker.detect(frame_bgr, timestamp_ms)

        if not face.detected:
            self._push_invalid_frame(timestamp_ms)
            if self._debug:
                self.last_roi_result = None
        else:
            roi_result = extract_rois(
                frame_bgr,
                face.landmarks_norm,
                roi_names=tuple(r.value for r in self.cfg.roi.enabled_rois),
                shrink_factor=self.cfg.roi.shrink_factor,
                min_valid_fraction=self.cfg.roi.min_valid_pixel_fraction,
            )
            self._push_frame(timestamp_ms, roi_result)
            if self._debug:
                self.last_roi_result = roi_result

        if len(self._buf) < 2:
            return None
        elapsed_s = (self._buf.timestamps_ms[-1] - self._buf.timestamps_ms[0]) / 1000.0
        # ВАЖНО (найдено на реальной записи телефона, fps=60.0143): _RingBuffer.
        # trim_older_than жёстко поддерживает инвариант elapsed_s <= window_seconds
        # (обрезает буфер до latest - window_seconds*1000 КАЖДЫЙ кадр) — из этого
        # математически следует, что elapsed_s может достичь window_seconds
        # ТОЧНО только если временная метка какого-то кадра ровно совпадёт с
        # cutoff. При РЕАЛЬНОМ джиттере веб-камеры (или "круглых" fps/window в
        # синтетических тестах, например fps=30 и window=10с, где int(300/30*1000)
        # == 10000 ровно) такое совпадение обычно случается. Но при идеально
        # равномерной сетке кадров с "некруглым" fps (типично для видеофайла
        # с телефона, например 60.0143) elapsed_s стабилизируется РОВНО на
        # (window_seconds - один период кадра) и остаётся там НАВСЕГДА — окно
        # физически никогда не "дозаполнялось" бы без этого допуска, и
        # process_frame молча не возвращал бы ни одной оценки ВЕСЬ файл
        # (см. также аналогичный допуск в scripts/finger_ppg.py::process_finger_video).
        # Доказуемая верхняя граница нехватки — РОВНО один период кадра (см.
        # анализ инварианта trim_older_than), поэтому допуск 1.5/fps с запасом
        # покрывает её в любом (в т.ч. наихудшем идеально равномерном) случае.
        fps_estimate = self._estimate_fps()
        readiness_tolerance_s = 1.5 / fps_estimate if fps_estimate > 0 else 0.0
        if elapsed_s < self.cfg.window.min_seconds_before_estimate - readiness_tolerance_s:
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
        self._buf.interocular_dist.append(roi_result.interocular_distance_px)
        bg_rgb = roi_result.background_rgb
        self._buf.background_rgb.append(bg_rgb if bg_rgb is not None else np.zeros(3))
        self._buf.background_valid.append(bool(roi_result.background_valid))
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
        self._buf.interocular_dist.append(np.nan)  # нет landmark'ов -> нет масштаба лица за этот кадр
        self._buf.background_rgb.append(np.zeros(3))
        self._buf.background_valid.append(False)
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

    def _update_last_valid_bpm(self, timestamp_ms: int, current_bpm: float, is_reliable: bool) -> None:
        """Задача 8: обновляет last_valid_bpm/age ТОЛЬКО когда окно реально
        прошло SQI-гейт (is_reliable=True) — в отличие от старого
        _update_best_effort_bpm, здесь принципиально нет "правдоподобного
        диапазона" для сырых кандидатов и нет fallback-значения: либо это
        настоящее измерение, либо состояние не меняется вообще."""
        if is_reliable and not np.isnan(current_bpm):
            self._last_valid_bpm = current_bpm
            self._last_valid_bpm_timestamp_ms = timestamp_ms

    def _last_valid_bpm_age_s(self, timestamp_ms: int) -> float | None:
        if self._last_valid_bpm_timestamp_ms is None:
            return None
        return max(0.0, (timestamp_ms - self._last_valid_bpm_timestamp_ms) / 1000.0)

    def _select_auto_method(self, fps: float, band: tuple[float, float]):
        """method=AUTO: оценивает КАЖДЫЙ из 5 цветовых методов по среднему
        spectral SNR, который он даёт на ВСЕХ включённых ROI ЭТОГО окна, и
        возвращает метод-победитель для использования во ВСЁМ окне целиком.

        Один метод на всё окно (а НЕ по отдельности на каждый ROI) —
        осознанное решение: если бы разные ROI обрабатывались разными
        методами, систематические межметодные различия (разный частотный/
        фазовый отклик POS vs CHROM vs PCA) подмешивались бы в
        cross_roi_agreement наравне с реальным шумом и портили бы этот
        компонент SQI, а не только собственно решали задачу выбора метода.

        Разные камеры/условия освещения объективно благоприятствуют разным
        классическим методам (нет единого метода, лучшего всегда и везде —
        см. CITATION.cff) — этот механизм подбирает метод под ТЕКУЩУЮ
        камеру автоматически, а не требует от пользователя вручную угадывать
        --method под своё конкретное железо.

        Стоимость: до 5x compute стадии "метод + препроцессинг + SNR"
        относительно фиксированного метода — по замерам
        scripts/benchmark_performance.py это <2мс на ROI даже для самого
        дорогого метода (ICA), т.е. пренебрежимо на фоне бюджета в 1с
        между BPM-шагами (WindowConfig.step_seconds)."""
        best_name, best_score = None, -np.inf
        for name, method in self._auto_candidate_methods.items():
            total_snr, n = 0.0, 0
            for roi in self.cfg.roi.enabled_rois:
                raw = np.array(self._buf.rgb_by_roi[roi.value])
                valid_mask = np.array(self._buf.valid_by_roi[roi.value], dtype=bool)
                if not is_gap_acceptable(valid_mask):
                    continue
                rgb = np.stack([fill_gaps(raw[:, ch], valid_mask) for ch in range(3)], axis=1)
                window = SignalWindow(rgb_traces={roi.value: rgb}, fps=fps, valid_mask=valid_mask, hr_band_hz=band)
                try:
                    raw_signal = method.extract(window, roi.value)
                except Exception:  # noqa: BLE001 - этот метод просто не участвует в голосовании
                    continue
                processed = preprocess_signal(
                    raw_signal, fps, band[0], band[1],
                    order=self.cfg.filt.filter_order, detrend_method=self.cfg.filt.detrend_method,
                    tarvainen_lambda=self.cfg.filt.tarvainen_lambda, normalize_method=self.cfg.filt.normalize_method,
                )
                _, snr_db = dominant_frequency_and_snr(
                    processed, fps, band, include_second_harmonic=self.cfg.quality.include_second_harmonic_in_snr
                )
                total_snr += snr_db
                n += 1
            score = (total_snr / n) if n else -np.inf
            if score > best_score:
                best_name, best_score = name, score

        if best_name is None:
            # Ни один метод не дал сигнала ни на одном ROI — оставляем
            # текущий self._method как есть; ниже по _compute_estimate
            # сработает обычная ветка "нет валидного сигнала ни с одного ROI".
            return self._method
        return self._auto_candidate_methods[best_name]

    def _estimate_color_roi(
        self,
        name: str,
        fps: float,
        band: tuple[float, float],
        timestamps_sec: np.ndarray,
        per_roi_bpm: dict[str, float],
        per_roi_signal: dict[str, np.ndarray],
        warnings: list[str],
        per_roi_raw_signal: dict[str, np.ndarray] | None = None,
    ) -> None:
        raw = np.array(self._buf.rgb_by_roi[name])
        valid_mask = np.array(self._buf.valid_by_roi[name], dtype=bool)

        # valid_mask одна и та же для R/G/B (окклюзия определяется на уровне
        # кадра, а не канала) — проверяем допустимость провала ОДИН раз, а
        # не пересчитываем длину провала 3 раза подряд (п.47 требований).
        if not is_gap_acceptable(valid_mask):
            warnings.append(f"ROI '{name}': слишком длинный провал трекинга/окклюзии в окне.")
            return
        rgb = np.stack([fill_gaps(raw[:, ch], valid_mask) for ch in range(3)], axis=1)

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
        if per_roi_raw_signal is not None:
            per_roi_raw_signal[name] = raw_signal
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
        per_roi_raw_signal: dict[str, np.ndarray] | None = None,
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
        if per_roi_raw_signal is not None:
            per_roi_raw_signal["face"] = raw_signal
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

    def _estimate_background_signal(self, fps: float, band: tuple[float, float]) -> np.ndarray | None:
        """Сигнал фонового ROI (вне лица) — для detect_illumination_flicker
        (п.22). Берётся зелёный канал (наибольшая чувствительность к
        яркостной модуляции, см. make_synthetic_rgb docstring в тестах) и
        обрабатывается тем же preprocess_signal, что и лицевые ROI, чтобы
        частоты были сравнимы напрямую.

        Задача 7: дисперсия проверяется на СЫРОМ (ДО preprocess_signal)
        сигнале — preprocess_signal нормализует через z-score, который
        растягивает К ЕДИНИЧНОЙ дисперсии ЛЮБОЙ сигнал, даже почти
        постоянный (ровная стена + шум квантования камеры). Если проверять
        дисперсию уже ПОСЛЕ normalize, разница между "реальным сигналом" и
        "усиленным шумом квантования" стирается по построению — детектор
        мерцания увидел бы обычный на вид сигнал и мог бы дать ложное
        срабатывание на ровном участке без какой-либо реальной модуляции."""
        valid_mask = np.array(self._buf.background_valid, dtype=bool)
        if valid_mask.sum() < 8:
            return None
        green = np.array(self._buf.background_rgb)[:, 1]
        fixed, ok = interpolate_missing(green, valid_mask)
        if not ok:
            return None
        if float(np.std(fixed)) < self.cfg.quality.flicker_min_raw_std:
            return None  # почти постоянный фон -> "нет данных", а не усиленный шум квантования
        return preprocess_signal(
            fixed, fps, band[0], band[1],
            order=self.cfg.filt.filter_order,
            detrend_method=self.cfg.filt.detrend_method,
            tarvainen_lambda=self.cfg.filt.tarvainen_lambda,
            normalize_method=self.cfg.filt.normalize_method,
        )

    def _compute_fusion(
        self,
        fps: float,
        band: tuple[float, float],
        timestamps_sec: np.ndarray,
        landmark_traj: np.ndarray,
        per_roi_bpm: dict[str, float],
        per_roi_signal: dict[str, np.ndarray],
        per_roi_raw_signal: dict[str, np.ndarray],
        warnings: list[str],
    ) -> tuple[str, np.ndarray, float, np.ndarray | None, dict[str, float]]:
        """SQI-взвешенное объединение всех доступных источников (3 цветовых
        ROI + опционально head-motion канал) в ОДИН сигнал, вместо
        argmax-выбора одного ROI (п.34 требований, см. signal/fusion.py).

        Возвращает (label, fused_signal, fused_bpm, harmonic_check_signal,
        snr_by_roi_db) — harmonic_check_signal строится тем же
        fusion-объединением, но по ДЕТРЕНДИРОВАННЫМ (не bandpass-
        отфильтрованным) исходным сигналам — та же логика, что и для
        одиночного best_roi в _compute_estimate, нужна harmonic_plausibility
        (см. её docstring в quality.py). snr_by_roi_db — СОБСТВЕННЫЙ spectral
        SNR каждого источника (задача 13б): cross_roi_agreement_score
        использует его, чтобы не сравнивать оценку заведомо шумного ROI с
        оценкой хорошего наравне."""
        fusion_sources = dict(per_roi_signal)
        if self._method.name != "head_motion" and self.cfg.fusion.include_head_motion:
            self._estimate_head_motion(
                landmark_traj, fps, band, timestamps_sec, per_roi_bpm, fusion_sources, warnings,
                per_roi_raw_signal=per_roi_raw_signal,
            )

        if len(fusion_sources) < 2:
            # Недостаточно независимых источников для осмысленного fusion —
            # деградируем к единственному доступному, а не городим
            # "взвешенное объединение" из одного сигнала.
            only_name = next(iter(fusion_sources))
            harmonic_signal = None
            if only_name in per_roi_raw_signal:
                harmonic_signal = detrend(
                    per_roi_raw_signal[only_name], method=self.cfg.filt.detrend_method, lam=self.cfg.filt.tarvainen_lambda
                )
            _, only_snr_db = dominant_frequency_and_snr(
                fusion_sources[only_name], fps, band, include_second_harmonic=self.cfg.quality.include_second_harmonic_in_snr
            )
            return (
                only_name, fusion_sources[only_name], per_roi_bpm.get(only_name, float("nan")), harmonic_signal,
                {only_name: only_snr_db},
            )

        weights: dict[str, float] = {}
        snr_by_roi_db: dict[str, float] = {}
        for name, sig in fusion_sources.items():
            _, snr_db = dominant_frequency_and_snr(
                sig, fps, band, include_second_harmonic=self.cfg.quality.include_second_harmonic_in_snr
            )
            snr_by_roi_db[name] = snr_db
            weights[name] = snr_db_to_weight(snr_db, self.cfg.fusion.weight_floor_db, self.cfg.fusion.weight_ceil_db)

        fused_signal, _diag = fuse_signals_by_sqi(
            fusion_sources, weights, fps=fps, max_lag_seconds=self.cfg.fusion.max_lag_seconds
        )
        # Валидная маска здесь всегда "всё валидно": каждый исходный сигнал
        # уже прошёл interpolate_missing до попадания в fusion_sources, а
        # lomb_scargle на полностью валидной равномерной сетке эквивалентен
        # обычной периодограмме — не теряет своего смысла, просто не
        # эксплуатирует своё преимущество для НЕравномерной выборки здесь.
        fused_bpm = self._estimate_frequency(
            fused_signal, np.ones(len(fused_signal), dtype=bool), fused_signal, fps, band, timestamps_sec
        )

        harmonic_check_signal = None
        detrended_sources = {
            name: detrend(per_roi_raw_signal[name], method=self.cfg.filt.detrend_method, lam=self.cfg.filt.tarvainen_lambda)
            for name in fusion_sources if name in per_roi_raw_signal
        }
        if detrended_sources:
            harmonic_check_signal, _ = fuse_signals_by_sqi(
                detrended_sources, {n: weights[n] for n in detrended_sources},
                fps=fps, max_lag_seconds=self.cfg.fusion.max_lag_seconds,
            )

        return "fusion", fused_signal, fused_bpm, harmonic_check_signal, snr_by_roi_db

    def _compute_estimate(self, timestamp_ms: int) -> PTSDPulseFeatures:
        fps = self._estimate_fps()
        band = (self.cfg.filt.low_hz, self.cfg.filt.high_hz)
        timestamps_sec = np.array(self._buf.timestamps_ms, dtype=float) / 1000.0
        landmark_traj = np.array(self._buf.landmark_traj)

        if self._auto_mode:
            # Выбираем метод-победитель ДО всего остального — все
            # последующие self._method.name-зависимые ветки (head_motion
            # проверка ниже, fusion в _compute_fusion) должны видеть уже
            # актуальный для этого окна метод.
            self._method = self._select_auto_method(fps, band)

        per_roi_bpm: dict[str, float] = {}
        per_roi_signal: dict[str, np.ndarray] = {}
        per_roi_raw_signal: dict[str, np.ndarray] = {}
        warnings: list[str] = []

        if self._method.name == "head_motion":
            self._estimate_head_motion(
                landmark_traj, fps, band, timestamps_sec, per_roi_bpm, per_roi_signal, warnings,
                per_roi_raw_signal=per_roi_raw_signal,
            )
        else:
            for roi in self.cfg.roi.enabled_rois:
                self._estimate_color_roi(
                    roi.value, fps, band, timestamps_sec, per_roi_bpm, per_roi_signal, warnings,
                    per_roi_raw_signal=per_roi_raw_signal,
                )

        if not per_roi_signal:
            no_signal_warnings = warnings + ["Ни один ROI не дал валидного сигнала в этом окне."]
            no_signal_method = f"auto:{self._method.name}" if self._auto_mode else self._method.name
            if self._window_logger is not None:
                # Тоже логируем (п.43) — "окно без сигнала" такой же
                # материал для анализа failure cases, как и низкий SQI.
                self._window_logger.log(WindowLogRecord(
                    timestamp_ms=timestamp_ms, bpm=float("nan"), per_roi_bpm=dict(per_roi_bpm),
                    publishable=False, method_used=no_signal_method,
                    frequency_method_used=self.cfg.frequency_method.value,
                    sqi_overall_score=0.0, sqi_level="low",
                    sqi_spectral_snr_db=float("-inf"), sqi_cross_roi_agreement=0.0,
                    sqi_landmark_stability=0.0, sqi_temporal_consistency=0.0,
                    sqi_harmonic_score=0.0, sqi_flicker_suspected=False,
                    sqi_flicker_background_freq_hz=0.0, sqi_flicker_background_snr_db=float("-inf"),
                    sqi_subharmonic_check_informative=True,
                    respiration_rate_bpm=None, warnings=no_signal_warnings,
                ))
            return PTSDPulseFeatures(
                timestamp_ms=timestamp_ms, bpm=float("nan"), hrv=None,
                sqi_score=0.0, sqi_level="low", publishable=False,
                confidence=0.0, confidence_breakdown={}, status="no_signal",
                warnings=no_signal_warnings,
                method_used=no_signal_method,
                frequency_method_used=self.cfg.frequency_method.value,
                per_roi_bpm=per_roi_bpm,
                last_valid_bpm=self._last_valid_bpm,
                last_valid_bpm_age_s=self._last_valid_bpm_age_s(timestamp_ms),
            )

        if self.cfg.fusion.enabled:
            best_roi, best_signal, current_bpm, harmonic_check_signal, snr_by_roi_db = self._compute_fusion(
                fps, band, timestamps_sec, landmark_traj, per_roi_bpm, per_roi_signal, per_roi_raw_signal, warnings,
            )
            best_peak_freq_hz, best_snr_db = dominant_frequency_and_snr(
                best_signal, fps, band, include_second_harmonic=self.cfg.quality.include_second_harmonic_in_snr
            )
        else:
            # ROI с максимальным spectral SNR используется как основной источник
            # BPM/HRV. Считаем dominant_frequency_and_snr РОВНО ОДИН раз на ROI
            # (Welch с nfft=2048 не бесплатен, а это выполняется на каждом шаге
            # окна) и переиспользуем для выбранного best_roi, а не пересчитываем
            # его же ещё раз после argmax (п.46 требований).
            snr_by_roi = {
                n: dominant_frequency_and_snr(sig, fps, band, include_second_harmonic=self.cfg.quality.include_second_harmonic_in_snr)
                for n, sig in per_roi_signal.items()
            }
            best_roi = max(snr_by_roi, key=lambda n: snr_by_roi[n][1])
            best_signal = per_roi_signal[best_roi]
            best_peak_freq_hz, best_snr_db = snr_by_roi[best_roi]
            current_bpm = per_roi_bpm.get(best_roi, float("nan"))
            # Задача 13б: собственный SNR каждого ROI — для cross_roi_agreement_score.
            snr_by_roi_db = {n: v[1] for n, v in snr_by_roi.items()}

            # п.21: для проверки на гармоники/субгармоники нужен ДЕТРЕНДИРОВАННЫЙ,
            # но НЕ узкополосно отфильтрованный сигнал — иначе сам bandpass в
            # preprocess_signal обрезает 2f/f/2 ещё до того, как их можно измерить
            # (см. harmonic_plausibility docstring в quality.py).
            harmonic_check_signal = None
            if best_roi in per_roi_raw_signal:
                harmonic_check_signal = detrend(
                    per_roi_raw_signal[best_roi],
                    method=self.cfg.filt.detrend_method,
                    lam=self.cfg.filt.tarvainen_lambda,
                )

        interocular_distances_px = np.array(self._buf.interocular_dist, dtype=float)
        background_signal = self._estimate_background_signal(fps, band)

        recent_bpm_history = list(self._recent_bpm_history) + [current_bpm]

        sqi = assess_quality(
            SQIInputs(
                spectral_snr_db=best_snr_db,
                peak_freq_hz=best_peak_freq_hz,
                fps=fps,
                band_hz=band,
                bpm_by_roi=per_roi_bpm,
                snr_by_roi_db=snr_by_roi_db,
                n_samples=len(best_signal),
                frequency_method=self.cfg.frequency_method.value,
                landmark_trajectories=landmark_traj,
                interocular_distances_px=interocular_distances_px,
                recent_bpm_history=recent_bpm_history,
                harmonic_check_signal=harmonic_check_signal,
                background_signal=background_signal,
                recent_flicker_candidates=list(self._recent_flicker_candidates),
            ),
            self.cfg.quality,
        )
        if not np.isnan(current_bpm):
            self._recent_bpm_history.append(current_bpm)
        # Задача 7: пополняем историю ОДНОконных кандидатов ПОСЛЕ вызова
        # (см. SQIInputs.recent_flicker_candidates — история должна
        # содержать только ПРЕДЫДУЩИЕ окна, текущее assess_quality уже
        # учла сама).
        self._recent_flicker_candidates.append(sqi.flicker_candidate)

        debug_info: DebugInfo | None = None
        if self._debug:
            psd_freqs, psd_values = compute_psd(best_signal, fps)
            debug_info = DebugInfo(
                sqi=sqi,
                qcfg=self.cfg.quality,
                band_hz=band,
                best_source=best_roi,
                peak_freq_hz=best_peak_freq_hz,
                psd_freqs=psd_freqs,
                psd_values=psd_values,
            )

        self._update_ibi_log(best_signal, fps, self._buf.timestamps_ms)
        hrv_warnings: list[str] = []
        hrv = self._maybe_compute_hrv(timestamp_ms, sqi.overall_score, hrv_warnings)

        respiration_rate_bpm, _ = estimate_respiration_rate(best_signal, fps)

        method_used = f"auto:{self._method.name}" if self._auto_mode else self._method.name
        if self.cfg.fusion.enabled:
            method_used = f"fusion({self._method.name}+head_motion)" if best_roi == "fusion" else best_roi

        all_warnings = warnings + sqi.warnings + hrv_warnings + (hrv.warnings if hrv is not None else [])

        # Задача 11: confidence/publishable считаются здесь (а не только в
        # return ниже), т.к. _update_last_valid_bpm тоже должен видеть НОВОЕ
        # (простое, пороговое) значение publishable — оно и есть контракт,
        # который реально виден потребителю на PTSDPulseFeatures, а не
        # скрытый от него sqi.is_reliable.
        confidence = sqi.overall_score
        is_publishable = confidence >= self.cfg.quality.min_overall_score_to_publish

        # Задача 8: обновляется ТОЛЬКО когда окно реально прошло гейт — см.
        # _update_last_valid_bpm. В отличие от прежнего best_effort_bpm,
        # здесь принципиально нет обновления на непубликуемых окнах.
        self._update_last_valid_bpm(timestamp_ms, current_bpm, is_publishable)

        if self._window_logger is not None:
            # п.43: полная разбивка SQI по компонентам (не только overall_score)
            # — материал для анализа failure cases, доступный ЗДЕСЬ (полный
            # sqi: SQIResult), но не в возвращаемом PTSDPulseFeatures, который
            # наружу отдаёт только сводные sqi_score/sqi_level.
            self._window_logger.log(WindowLogRecord(
                timestamp_ms=timestamp_ms,
                bpm=current_bpm,
                per_roi_bpm=dict(per_roi_bpm),
                publishable=sqi.is_reliable,
                method_used=method_used,
                frequency_method_used=self.cfg.frequency_method.value,
                sqi_overall_score=sqi.overall_score,
                sqi_level=sqi.level.value,
                sqi_spectral_snr_db=sqi.spectral_snr_db,
                sqi_cross_roi_agreement=sqi.cross_roi_agreement,
                sqi_landmark_stability=sqi.landmark_stability,
                sqi_temporal_consistency=sqi.temporal_consistency,
                sqi_harmonic_score=sqi.harmonic_score,
                sqi_flicker_suspected=sqi.flicker_suspected,
                sqi_flicker_background_freq_hz=sqi.flicker_background_freq_hz,
                sqi_flicker_background_snr_db=sqi.flicker_background_snr_db,
                sqi_subharmonic_check_informative=sqi.subharmonic_check_informative,
                respiration_rate_bpm=respiration_rate_bpm,
                warnings=list(all_warnings),
            ))

        # confidence/is_publishable уже посчитаны выше (перед _update_last_valid_bpm).
        # confidence_breakdown — ВСЕ компоненты по отдельности (см. docstring
        # PTSDPulseFeatures.confidence_breakdown); НЕ то же самое, что
        # sqi.is_reliable (логируется как есть в JSONL выше — тот учитывает
        # ещё и flicker_suspected/жёсткий пол SNR по отдельности как AND).
        confidence_breakdown = {
            "spectral_snr_db": sqi.spectral_snr_db,
            "harmonic_score": sqi.harmonic_score,
            "cross_roi_agreement": sqi.cross_roi_agreement,
            "landmark_stability": sqi.landmark_stability,
            "temporal_consistency": sqi.temporal_consistency,
            "flicker_suspected": 1.0 if sqi.flicker_suspected else 0.0,
        }
        return PTSDPulseFeatures(
            timestamp_ms=timestamp_ms,
            bpm=current_bpm,
            hrv=hrv,
            sqi_score=sqi.overall_score,
            sqi_level=sqi.level.value,
            publishable=is_publishable,
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
            status="ok",
            warnings=all_warnings,
            method_used=method_used,
            frequency_method_used=self.cfg.frequency_method.value,
            per_roi_bpm=per_roi_bpm,
            respiration_rate_bpm=respiration_rate_bpm,
            last_valid_bpm=self._last_valid_bpm,
            last_valid_bpm_age_s=self._last_valid_bpm_age_s(timestamp_ms),
            debug=debug_info,
        )
