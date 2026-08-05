"""
Запуск пайплайна с веб-камеры в реальном времени.

Использование:
    python scripts/run_webcam.py --camera 0 --method pos --freq welch
    python scripts/run_webcam.py --camera 0 --log session.jsonl   # см. п.43
    python scripts/run_webcam.py --camera 0 --debug                # см. п.44

Перед первым запуском один раз скачайте модель Face Landmarker (версия
зафиксирована — п.41 требований, см. src/rppg/face/landmarker.py::MODEL_URL):
    mkdir -p models
    curl -L -o models/face_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.config_io import load_config
from rppg.pipeline import RPPGPipeline
from rppg.face.roi import build_roi_mask, build_background_roi_mask, ROI_LANDMARK_INDICES

# Сколько последних ПУБЛИКУЕМЫХ (SQI ok) оценок усредняем для отображаемого
# числа. Даже среди publishable=True окон отдельные оценки могут скакать —
# медиана нескольких последних сглаживает это для живого демо, не трогая
# сам pipeline/SQI (это чисто отображение, см. draw_overlay).
DISPLAY_SMOOTHING_WINDOW = 5
# Если дольше этого не было ни одной publishable-оценки — считаем
# сглаженное число устаревшим и не показываем его (человек мог отойти,
# сменить освещение и т.п.), а не морозим старое значение на экране.
STALE_AFTER_MS = 10_000


def draw_overlay(frame, result, smoothed_bpm: float | None,
                  buffered_seconds: float = 0.0, target_seconds: float = 0.0) -> None:
    if result is None:
        # Задача 6: раньше это было просто "Buffering..." без чисел —
        # с тех пор как min_seconds_before_estimate по умолчанию равен
        # window_seconds, process_frame() возвращает None ВСЮ буферизацию
        # целиком (несколько секунд), и голая надпись без прогресса легко
        # читается как "программа зависла", а не "ждёт заполнения окна".
        cv2.putText(frame, f"Buffering... {buffered_seconds:.1f} / {target_seconds:.1f} c",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return

    # ВАЖНО: показываем число ТОЛЬКО когда SQI ему доверяет (smoothed_bpm
    # приходит из окон с publishable=True, см. main()). Раньше здесь
    # показывался result.bpm ЛЮБОГО окна (просто другим цветом, если
    # publishable=False) — визуально выглядело как "пульс скачет 50<->150",
    # хотя на самом деле система КОРРЕКТНО не доверяла этим числам, просто
    # всё равно их печатала. Не публикуемые, но реально существующие числа
    # ещё сильнее шумят, чем можно было бы подумать по (относительно
    # спокойному) SQI-скору — сам SQI не гарантирует, что НЕОПУБЛИКОВАННОЕ
    # число близко к истине, гарантия действует только для publishable=True.
    if smoothed_bpm is not None:
        cv2.putText(frame, f"BPM: {smoothed_bpm:.0f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "Измеряю пульс... (нужно больше света, лицо ближе, поменьше движения)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    color = (0, 200, 0) if result.publishable else (0, 165, 255)
    cv2.putText(frame, f"SQI: {result.sqi_level} ({result.sqi_score:.2f})", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"method: {result.method_used}/{result.frequency_method_used}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Задача 8: раньше здесь был best_effort_bpm — ВСЕГДА реальное число
    # (даже без сигнала, стартуя с fallback=75), неотличимое на глаз от
    # настоящего измерения. last_valid_bpm — ЧЕСТНАЯ альтернатива: либо
    # последнее РЕАЛЬНО измеренное значение с его возрастом (человек сам
    # видит, насколько оно устарело), либо явное "измерений ещё не было" —
    # ни то, ни другое НИКОГДА не рисуется зелёным (тем же цветом, что и
    # доверенный BPM выше), чтобы их нельзя было спутать на экране.
    if result.last_valid_bpm is not None:
        last_valid_text = (
            f"last valid measurement: {result.last_valid_bpm:.0f} BPM "
            f"({result.last_valid_bpm_age_s:.0f}s ago)"
        )
    else:
        last_valid_text = "last valid measurement: ещё не было ни одного измерения"
    cv2.putText(frame, last_valid_text, (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    if result.hrv is not None and result.publishable:
        cv2.putText(frame, f"SDNN: {result.hrv.sdnn_ms:.0f}ms  RMSSD: {result.hrv.rmssd_ms:.0f}ms  "
                            f"pNN50: {result.hrv.pnn50_pct:.0f}%",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    if not result.publishable:
        cv2.putText(frame, "LOW QUALITY - not sent to PTSD pipeline", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# Цвета контуров ROI на кадре в --debug (см. draw_roi_contours). Отдельные
# цвета на ROI, чтобы контуры не сливались друг с другом на кадре; красный
# ЗАРЕЗЕРВИРОВАН под "ROI невалиден в этом кадре" (см. valid_by_roi/
# background_valid) — не переиспользуется как цвет идентичности ROI.
_ROI_CONTOUR_COLORS = {
    "forehead": (255, 200, 0),
    "left_cheek": (0, 200, 255),
    "right_cheek": (200, 0, 255),
}
_INVALID_ROI_COLOR = (0, 0, 255)
_BACKGROUND_ROI_COLOR = (180, 180, 180)


def draw_roi_contours(frame, pipeline: RPPGPipeline) -> None:
    """Контуры ВСЕХ ROI (включая фоновый, вне лица) прямо на кадре — п.44
    требований. Перерисовывается КАЖДЫЙ кадр из pipeline.last_roi_result
    (обновляется в process_frame независимо от того, готова ли уже
    BPM-оценка), а не только на кадрах с новой BPM-оценкой — иначе контуры
    дёргались бы с частотой WindowConfig.step_seconds вместо частоты кадров."""
    roi_result = pipeline.last_roi_result
    if roi_result is None:
        return
    landmarks_px = roi_result.landmarks_px

    for roi in pipeline.cfg.roi.enabled_rois:
        name = roi.value
        if name not in ROI_LANDMARK_INDICES:
            continue  # head_motion не строит цветовой ROI-полигон, рисовать нечего
        mask = build_roi_mask(landmarks_px, name, frame.shape, shrink_factor=pipeline.cfg.roi.shrink_factor)
        valid = roi_result.valid_by_roi.get(name, False)
        color = _ROI_CONTOUR_COLORS.get(name, (255, 255, 255)) if valid else _INVALID_ROI_COLOR
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
        if contours:
            corner = tuple(contours[0][contours[0][:, :, 1].argmin()][0])
            cv2.putText(frame, name, corner, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    bg_mask = build_background_roi_mask(frame.shape, landmarks_px)
    if bg_mask is not None:
        color = _BACKGROUND_ROI_COLOR if roi_result.background_valid else _INVALID_ROI_COLOR
        contours, _ = cv2.findContours(bg_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
        if contours:
            corner = tuple(contours[0][contours[0][:, :, 1].argmin()][0])
            cv2.putText(frame, "background", corner, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# (метка, доступ к значению, порог, диапазон отображения полоски). Пороги
# spectral_snr/overall/landmark_stability берутся из живого QualityConfig
# (передаётся из result.debug.qcfg), а не дублируются здесь захардкоженными
# числами — иначе оверлей и реальный гейт публикации могли бы разойтись при
# смене конфига. cross_roi/temporal НЕ имеют отдельного поля-порога в
# QualityConfig — это soft-warning уровень 0.5, зашитый прямо в
# quality.py::assess_quality (см. "if roi_score < 0.5" / "if temporal_score
# < 0.5" там же), поэтому дублируется здесь тем же литералом с явной
# отсылкой к первоисточнику, а не берётся из конфига, которого для него нет.
_CROSS_ROI_TEMPORAL_WARN_THRESHOLD = 0.5
# harmonic_score: 1.0 = гармоническая структура спектра чиста (см.
# harmonic_plausibility docstring в quality.py); любое значение < 1.0 уже
# означает, что штраф применился (обнаружена конкурирующая энергия на 2f
# или f/2) — здесь это и есть "порог", а не отдельная калиброванная константа.
_HARMONIC_CLEAN_THRESHOLD = 1.0


def draw_sqi_debug_panel(frame, sqi_result, qcfg, start_y: int) -> int:
    """Каждая компонента SQI по отдельности — число + полоска-индикатор,
    подсвечена красным, если НИЖЕ своего порога (п.44: "какая компонента
    мешает публикации прямо сейчас"). Возвращает y для следующего блока."""
    rows = [
        ("overall_score", sqi_result.overall_score, qcfg.min_overall_score_to_publish, 0.0, 1.0),
        ("spectral_snr_db", sqi_result.spectral_snr_db, qcfg.min_spectral_snr_db, -5.0, 15.0),
        ("harmonic_score", sqi_result.harmonic_score, _HARMONIC_CLEAN_THRESHOLD, 0.0, 1.0),
        ("cross_roi_agreement", sqi_result.cross_roi_agreement, _CROSS_ROI_TEMPORAL_WARN_THRESHOLD, 0.0, 1.0),
        ("landmark_stability", sqi_result.landmark_stability, qcfg.min_landmark_stability, 0.0, 1.0),
        ("temporal_consistency", sqi_result.temporal_consistency, _CROSS_ROI_TEMPORAL_WARN_THRESHOLD, 0.0, 1.0),
    ]
    x0, bar_w, bar_h, row_h = 20, 120, 12, 22
    y = start_y
    for name, value, threshold, lo, hi in rows:
        below = value < threshold
        color = (0, 0, 255) if below else (0, 200, 0)
        frac = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (90, 90, 90), 1)
        cv2.rectangle(frame, (x0 + 1, y + 1), (x0 + 1 + int((bar_w - 2) * frac), y + bar_h - 1), color, -1)
        cv2.putText(frame, f"{name}: {value:.2f} (thr {threshold:.2f})", (x0 + bar_w + 8, y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += row_h

    flicker_color = (0, 0, 255) if sqi_result.flicker_suspected else (0, 200, 0)
    cv2.putText(frame, f"flicker_suspected: {sqi_result.flicker_suspected}", (x0, y + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, flicker_color, 1)
    y += row_h
    # Задача 7: candidate — ОДНОконное совпадение ДО требования устойчивости
    # (см. quality.assess_quality) — жёлтым, если кандидат есть, но ещё не
    # подтверждён несколькими окнами подряд (не путать с flicker_suspected
    # выше, который уже прошёл через это требование).
    candidate_color = (0, 165, 255) if (sqi_result.flicker_candidate and not sqi_result.flicker_suspected) else (120, 120, 120)
    cv2.putText(
        frame,
        f"  bg candidate: {sqi_result.flicker_candidate} @ {sqi_result.flicker_background_freq_hz * 60:.0f} bpm "
        f"(snr {sqi_result.flicker_background_snr_db:.1f} dB)",
        (x0, y + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, candidate_color, 1,
    )
    y += row_h + 6

    # Дублирует spectral_snr_db выше, но крупно и в явном человекочитаемом
    # виде "N dB / need M dB" — отдельным требованием п.44, т.к. именно SNR
    # чаще всего оказывается связывающим ограничением (см. scripts/analyze_log.py).
    snr_ok = sqi_result.spectral_snr_db >= qcfg.min_spectral_snr_db
    snr_color = (0, 200, 0) if snr_ok else (0, 0, 255)
    cv2.putText(frame, f"SNR: {sqi_result.spectral_snr_db:.1f} dB / need {qcfg.min_spectral_snr_db:.1f}",
                (x0, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, snr_color, 2)
    return y + 16 + 14


def draw_per_roi_raw_bpm(frame, per_roi_bpm: dict, x0: int, y0: int) -> int:
    """Сырой BPM КАЖДОГО ROI по отдельности (до fusion/выбора best_roi) —
    п.44: расхождение между ROI видно напрямую, не только через свёрнутый
    cross_roi_agreement."""
    cv2.putText(frame, "raw BPM per ROI:", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y = y0 + 20
    for name, bpm in per_roi_bpm.items():
        text = f"  {name}: {bpm:.0f}" if not np.isnan(bpm) else f"  {name}: n/a"
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)
        y += 18
    return y


def draw_psd_panel(frame, debug_info, panel_w: int = 260, panel_h: int = 150, margin: int = 10) -> None:
    """Спектр Уэлча лучшего ROI как маленький график в углу кадра (п.44), с
    отметками найденного пика, 2f/f/2 (см. harmonic_plausibility) и границ
    рабочей полосы (band_hz) — тот же самый PSD, что assess_quality реально
    использовал для этого окна (см. DebugInfo/compute_psd в pipeline.py),
    а не пересчитанный заново с другими параметрами Welch."""
    freqs, psd = debug_info.psd_freqs, debug_info.psd_values
    if freqs is None or psd is None or len(freqs) < 2:
        return
    h, w = frame.shape[:2]
    x0, y0 = w - panel_w - margin, margin
    x1, y1 = x0 + panel_w, y0 + panel_h

    cv2.rectangle(frame, (x0, y0), (x1, y1), (25, 25, 25), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

    band_lo, band_hi = debug_info.band_hz
    peak = debug_info.peak_freq_hz
    f_max = max(band_hi * 1.6, peak * 2.3, 0.1)
    mask = freqs <= f_max
    fs, ps = freqs[mask], psd[mask]
    if len(fs) < 2:
        return
    ps_db = 10.0 * np.log10(ps + 1e-12)
    ps_norm = (ps_db - ps_db.min()) / (ps_db.max() - ps_db.min() + 1e-9)

    def fx(f: float) -> int:
        return int(x0 + np.clip(f / f_max, 0.0, 1.0) * panel_w)

    def fy(v: float) -> int:
        return int(y1 - 6 - v * (panel_h - 22))

    pts = np.array([[fx(f), fy(v)] for f, v in zip(fs, ps_norm)], dtype=np.int32)
    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 1)

    def vline(f: float, color: tuple, thickness: int = 1) -> None:
        x = fx(f)
        cv2.line(frame, (x, y0 + 14), (x, y1), color, thickness, cv2.LINE_AA)

    vline(band_lo, (150, 150, 150))
    vline(band_hi, (150, 150, 150))
    vline(peak, (0, 0, 255), 2)
    if 2 * peak <= f_max:
        vline(2 * peak, (0, 165, 255))
    if peak / 2 >= 0:
        vline(peak / 2, (0, 165, 255))

    cv2.putText(frame, f"Welch PSD: {debug_info.best_source}", (x0 + 4, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"peak {peak * 60:.0f} bpm", (x0 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(frame, "2f/f2", (x0 + panel_w - 42, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)


def draw_debug_overlay(frame, pipeline: RPPGPipeline, result) -> None:
    """Точка входа отладочного оверлея (--debug, п.44 требований): контуры
    всех ROI на кадре ВСЕГДА (даже пока BPM-окно ещё не готово), разбивка
    SQI по компонентам/PSD/raw BPM по ROI — только когда результат окна
    уже посчитан (result.debug заполняется RPPGPipeline(debug=True))."""
    draw_roi_contours(frame, pipeline)
    if result is None or result.debug is None:
        return
    y = draw_sqi_debug_panel(frame, result.debug.sqi, result.debug.qcfg, start_y=190)
    y = draw_per_roi_raw_bpm(frame, result.per_roi_bpm, 20, y + 8)
    draw_psd_panel(frame, result.debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="rPPG realtime demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--config", type=str, default=None, help="Базовый YAML/JSON-конфиг, см. rppg.config_io")
    parser.add_argument("--method", type=str, default=None, choices=[m.value for m in ExtractionMethod])
    parser.add_argument("--freq", type=str, default=None, choices=[m.value for m in FrequencyMethod])
    parser.add_argument(
        "--log", type=str, default=None,
        help="Путь к JSONL-логу по окнам (timestamp/BPM по ROI/компоненты SQI/warnings, см. п.43)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Отладочный оверлей (п.44): компоненты SQI по отдельности с порогами, "
             "Welch-спектр лучшего ROI, контуры всех ROI (включая фоновый), сырой BPM по ROI",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else PipelineConfig()
    config.face.model_asset_path = args.model
    if args.method is not None:
        config.method = ExtractionMethod(args.method)
    if args.freq is not None:
        config.frequency_method = FrequencyMethod(args.freq)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {args.camera}")

    start_ms = time.time() * 1000

    published_bpm_history: deque = deque(maxlen=DISPLAY_SMOOTHING_WINDOW)
    last_published_ms: int | None = None

    with RPPGPipeline(config, log_path=args.log, debug=args.debug) as pipeline:
        last_result = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(time.time() * 1000 - start_ms)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                last_result = result
                if result.publishable:
                    published_bpm_history.append(result.bpm)
                    last_published_ms = timestamp_ms
                if result.warnings:
                    print(f"[{timestamp_ms/1000:.1f}s] " + " | ".join(result.warnings))

            is_stale = last_published_ms is None or timestamp_ms - last_published_ms > STALE_AFTER_MS
            smoothed_bpm = None if is_stale or not published_bpm_history else float(np.median(published_bpm_history))

            draw_overlay(frame, last_result, smoothed_bpm,
                         buffered_seconds=pipeline.buffered_seconds,
                         target_seconds=pipeline.cfg.window.min_seconds_before_estimate)
            if args.debug:
                draw_debug_overlay(frame, pipeline, last_result)
            cv2.imshow("rPPG (ESC to quit)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
