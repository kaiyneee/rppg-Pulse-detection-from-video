"""
Сбор данных по поведению SQI на ВЫРОЖДЕННЫХ входах.

Прогоняет пайплайн БЕЗ ЕДИНОГО изменения его логики по 11 условиям (см.
rppg.degrade.CONDITIONS) на записях UBFC-rPPG и выгружает пооконные
значения ВСЕХ компонент индекса качества по отдельности, не агрегируя.

Что здесь делается поверх штатного прогона и почему это не меняет пайплайн:
  * кадры подменяются на входе (rppg.degrade) — пайплайн не знает, что
    видео деградировано, и обрабатывает его как обычное;
  * для условия D1 подменяется ФУНКЦИЯ выбора ROI (см. _background_rois):
    это обёртка вокруг штатной roi.extract_rois, поставленная на время
    прогона в rppg.pipeline; сам модуль face/roi.py не тронут;
  * логируются ДВА решения о публикации на одном окне: published_gated
    (как решает текущая архитектура: композит + жёсткие гейты SNR и
    мерцания) и published_weighted_only (как решал бы ОДИН композитный
    балл со своим порогом). Это основное сравнение работы.

Запуск (полный прогон, все условия, все записи):
    PYTHONPATH=src venv/bin/python3 benchmark/degenerate.py --data-dir ~/Downloads

Задача 0 (приоритетная проверка детектора мерцания):
    PYTHONPATH=src venv/bin/python3 benchmark/degenerate.py --task0
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import platform
import re
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rppg.pipeline as pipeline_mod  # noqa: E402
from rppg.config import DEFAULT_PRESET, PRESET_OVERRIDES, PipelineConfig, make_pipeline_config  # noqa: E402
from rppg.degrade import (  # noqa: E402
    H264_CRF,
    CONDITIONS,
    CONDITIONS_WITH_GENUINE_SIGNAL,
    CONDITIONS_WITH_INJECTION,
    FLICKER_SPECS,
    FrameDegrader,
    prepare_record,
    transcode_h264,
)
from rppg.face.roi import ROIExtractionResult, roi_mean_rgb  # noqa: E402
from rppg.face.roi import extract_rois as _orig_extract_rois  # noqa: E402
from rppg.pipeline import RPPGPipeline, apply_second_pass  # noqa: E402

import eval_ubfc_subject as ev  # noqa: E402  (переиспользуются загрузчик эталона и поиск файлов)

OUT_DIR_DEFAULT = REPO_ROOT / "results" / "degenerate"
TASK0_RECORDS = ("5", "6", "7", "8", "10")
TASK0_CONDITIONS = ("G0", "B1w", "B1f")

WINDOW_FIELDS = [
    "record_id", "condition", "window_idx", "t_start_s",
    "sqi_composite", "sqi_snr", "snr_db", "sqi_roi_agreement",
    "sqi_landmark_stability", "sqi_temporal_consistency",
    "harmonic_plausibility", "flicker_flag", "flicker_score",
    "hard_snr_gate_pass", "published_gated", "published_weighted_only",
    "bpm_estimate", "bpm_reference", "bpm_injected",
    "method", "freq_method", "fs_used_hz",
]

SUMMARY_FIELDS = [
    "condition", "n_records", "n_windows",
    "publish_rate_gated", "publish_rate_weighted_only",
    "median_sqi_composite", "median_sqi_snr", "median_snr_db",
    "median_sqi_roi_agreement", "median_sqi_landmark_stability",
    "median_sqi_temporal_consistency",
    "flicker_flag_rate", "harmonic_plausibility_fail_rate",
    "share_published_near_injected", "mae_vs_reference", "mae_pooled_windows", "coverage",
]

PERCENTILE_METRICS = [
    "sqi_composite", "sqi_snr", "snr_db", "sqi_roi_agreement",
    "sqi_landmark_stability", "sqi_temporal_consistency",
    "harmonic_plausibility", "flicker_score",
]

NEAR_INJECTED_TOLERANCE_BPM = 5.0


# --------------------------------------------------------------------------
# Частота, фактически полученная оценщиком Уэлча
# --------------------------------------------------------------------------
@contextlib.contextmanager
def trace_estimate_fps(holder: dict):
    """Снимает значение RPPGPipeline._estimate_fps ИМЕННО ТОГО вызова, что
    питает оценку частоты (_compute_estimate), не трогая сам пайплайн.

    _estimate_fps вызывается из двух мест (строки 1012 и 1526); нужно
    второе — то, чей результат уходит в Уэлч. Различаются по имени
    вызывающей функции, поэтому обёртка смотрит на кадр стека, а не
    угадывает по порядку вызовов.
    """
    original = RPPGPipeline._estimate_fps

    def traced(self):
        value = original(self)
        if sys._getframe(1).f_code.co_name == "_compute_estimate":
            holder["fps"] = value
        return value

    RPPGPipeline._estimate_fps = traced
    try:
        yield
    finally:
        RPPGPipeline._estimate_fps = original


# --------------------------------------------------------------------------
# D1: перенос ROI на фон
# --------------------------------------------------------------------------
def _background_patch_masks(
    frame_shape, landmarks_px, n_patches: int,
    patch_fraction: float = 0.12, margin_fraction: float = 0.20,
):
    """Квадратные патчи в СВОБОДНЫХ углах кадра — вне bounding box'а лица.

    Повторяет геометрию roi.build_background_roi_mask (тот же patch_fraction
    и margin_fraction, тот же порядок углов), но возвращает НЕСКОЛЬКО патчей:
    лицевых ROI три, и подменять их все одним и тем же патчем нельзя —
    межзонное согласие тогда было бы тождественно единичным по построению
    обёртки, а не по поведению самого индекса качества. Если свободных углов
    меньше трёх, патчи повторяются по кругу (это фиксируется в отчёте).
    """
    h, w = frame_shape[:2]
    patch = int(round(patch_fraction * min(h, w)))
    margin = int(round(margin_fraction * min(h, w)))
    if patch < 4:
        return [None] * n_patches
    x_min = float(landmarks_px[:, 0].min()) - margin
    y_min = float(landmarks_px[:, 1].min()) - margin
    x_max = float(landmarks_px[:, 0].max()) + margin
    y_max = float(landmarks_px[:, 1].max()) + margin
    free = []
    for px0, py0 in ((0, 0), (w - patch, 0), (0, h - patch), (w - patch, h - patch)):
        px1, py1 = px0 + patch, py0 + patch
        if px1 <= x_min or px0 >= x_max or py1 <= y_min or py0 >= y_max:
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            mask[py0:py1, px0:px1] = 1
            free.append(mask)
    if not free:
        return [None] * n_patches
    return [free[i % len(free)] for i in range(n_patches)]


def _background_rois(
    frame_bgr, landmarks_norm, roi_names=("forehead", "left_cheek", "right_cheek"),
    shrink_factor: float = 0.9, min_valid_fraction: float = 0.6,
    trim_fraction: float = 0.10, skin_mask=None,
) -> ROIExtractionResult:
    """Условие D1: лицевые ROI принудительно перенесены на фон.

    Всё остальное (landmark-точки, межзрачковое расстояние, фоновый ROI для
    детектора мерцания) берётся из ШТАТНОЙ extract_rois без изменений —
    подменяется только то, откуда берётся цвет трёх «лицевых» зон.
    Маска кожи к фоновым патчам НЕ применяется: порог кожи отсеял бы стену
    целиком (так же поступает и сам roi.py со своим фоновым ROI).
    """
    base = _orig_extract_rois(
        frame_bgr, landmarks_norm, roi_names, shrink_factor,
        min_valid_fraction, trim_fraction, skin_mask,
    )
    masks = _background_patch_masks(frame_bgr.shape, base.landmarks_px, len(roi_names))
    rgb_by_roi, valid_by_roi, count_by_roi, cause_by_roi = {}, {}, {}, {}
    for name, mask in zip(roi_names, masks):
        if mask is None:
            rgb_by_roi[name], valid_by_roi[name] = None, False
            count_by_roi[name], cause_by_roi[name] = 0, "mask_empty"
            continue
        rgb, valid, n_valid, cause = roi_mean_rgb(frame_bgr, mask, min_valid_fraction, trim_fraction)
        rgb_by_roi[name], valid_by_roi[name] = rgb, valid
        count_by_roi[name], cause_by_roi[name] = n_valid, cause
    return replace(
        base, rgb_by_roi=rgb_by_roi, valid_by_roi=valid_by_roi,
        valid_pixel_count_by_roi=count_by_roi, rejection_cause_by_roi=cause_by_roi,
    )


@contextlib.contextmanager
def roi_override(condition: str):
    """Подмена выбора ROI на время прогона одного условия."""
    if condition != "D1":
        yield
        return
    saved = pipeline_mod.extract_rois
    pipeline_mod.extract_rois = _background_rois
    try:
        yield
    finally:
        pipeline_mod.extract_rois = saved


# --------------------------------------------------------------------------
# Записи и эталон
# --------------------------------------------------------------------------
def _natural_key(record_id: str):
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", record_id))


def discover_records(data_dir: Path) -> list[tuple[str, Path]]:
    found = {}
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            ev.find_video(sub)
            ev.find_ground_truth(sub)
        except ev.SubjectDataError:
            continue
        found[ev.resolve_subject_id(sub)] = sub
    return sorted(found.items(), key=lambda kv: _natural_key(kv[0]))


def median_reference_bpm(t_ref, wave, cfg: PipelineConfig, window_seconds: float) -> float:
    """Медианный эталонный BPM записи — по тем же окнам и той же процедурой
    (контактная волна -> Уэлч), что и эталон в основной работе. Считается
    по регулярной сетке окон, а не по фактическим меткам пайплайна: нужен
    ДО прогона, чтобы задать частоту наложения для условия B3."""
    t_end = float(t_ref[-1])
    values = []
    t = window_seconds
    while t <= t_end:
        ref, _snr = ev.reference_bpm_from_wave(t_ref, wave, t, window_seconds, cfg)
        if ref is not None and np.isfinite(ref):
            values.append(float(ref))
        t += 1.0
    if not values:
        raise ev.SubjectDataError("эталонная волна не дала ни одного валидного окна")
    return float(np.median(values))


# --------------------------------------------------------------------------
# Прогон одного условия на одной записи
# --------------------------------------------------------------------------
def run_condition(record_id: str, subject_dir: Path, condition: str, verbose: bool = True,
                  preset: str = DEFAULT_PRESET) -> list[dict]:
    video_path = ev.find_video(subject_dir)
    gt_path = ev.find_ground_truth(subject_dir)
    t_ref, wave, _hr_device = ev.load_reference_waveform(gt_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ev.SubjectDataError(f"не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or PipelineConfig().window.assumed_fps

    cfg = make_pipeline_config(preset)
    cfg.face.model_asset_path = str(ev.MODEL_PATH)
    cfg.window.assumed_fps = float(fps)
    window_seconds = cfg.window.window_seconds

    injected_hz = None
    if condition == "B3":
        injected_hz = (median_reference_bpm(t_ref, wave, cfg, window_seconds) + 25.0) / 60.0
    elif condition in FLICKER_SPECS:
        injected_hz = FLICKER_SPECS[condition][0]

    tmp_transcoded = None
    if condition == "C2":
        # Перекодирование во временный файл рядом с выгрузкой результатов,
        # а не в память: 2 ГБ исходника x 50 записей в оперативную память
        # не помещаются, а хранить перекодированные копии после прогона
        # незачем — они удаляются в finally ниже.
        tmp_transcoded = Path(tempfile.gettempdir()) / f"degenerate_c2_{record_id}.mp4"
        transcode_h264(str(video_path), str(tmp_transcoded))
        cap.release()
        cap = cv2.VideoCapture(str(tmp_transcoded))
        if not cap.isOpened():
            raise ev.SubjectDataError(f"не удалось открыть перекодированное видео {tmp_transcoded}")

    prep = prepare_record(str(video_path), condition, float(fps))
    degrader = FrameDegrader(condition, record_id, float(fps), prep, injected_hz=injected_hz)
    injected_bpm = degrader.injected_bpm

    rows = []
    fs_by_timestamp: dict[int, float] = {}
    fps_holder: dict[str, float | None] = {"fps": None}
    t0 = time.time()
    with roi_override(condition), trace_estimate_fps(fps_holder), RPPGPipeline(cfg, debug=True) as pipeline:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            degraded = degrader(frame, frame_idx)
            timestamp_ms = int(frame_idx / fps * 1000)
            fps_holder["fps"] = None
            result = pipeline.process_frame(degraded, timestamp_ms)
            if result is not None:
                rows.append(result)
                if fps_holder["fps"] is not None:
                    fs_by_timestamp[result.timestamp_ms] = float(fps_holder["fps"])
            frame_idx += 1
        # Двухпроходный офлайн-режим — тот же, в котором получены
        # опубликованные числа основной работы (см. критерий приёмки 1).
        apply_second_pass(rows, pipeline._trusted_bpm_history, cfg.frequency.history_min_windows)
    cap.release()
    if tmp_transcoded is not None:
        tmp_transcoded.unlink(missing_ok=True)

    records = _build_rows(rows, record_id, condition, t_ref, wave, cfg, window_seconds, injected_bpm,
                          fs_by_timestamp)
    if verbose:
        pub = sum(1 for r in records if r["published_gated"])
        print(f"  {record_id:16s} {condition:4s} окон {len(records):4d}  опубликовано {pub:4d} "
              f"({pub / max(len(records), 1):5.1%})  {time.time() - t0:5.1f}с")
    return records


def _build_rows(rows, record_id, condition, t_ref, wave, cfg, window_seconds, injected_bpm,
                fs_by_timestamp=None) -> list[dict]:
    qcfg = cfg.quality
    want_reference = condition in CONDITIONS_WITH_GENUINE_SIGNAL
    out = []
    for idx, r in enumerate(rows):
        window_end_s = r.timestamp_ms / 1000.0
        sqi = r.debug.sqi if r.debug is not None else None
        composite = float(r.sqi_score)
        # None -> пустая ячейка: окно не дошло до оценки качества вообще
        # (ранний возврат пайплайна), в отличие от посчитанного -inf.
        snr_db = float(sqi.spectral_snr_db) if sqi is not None else None
        ref = None
        if want_reference:
            ref, _ref_snr = ev.reference_bpm_from_wave(t_ref, wave, window_end_s, window_seconds, cfg)
        out.append({
            "record_id": record_id,
            "condition": condition,
            "window_idx": idx,
            "t_start_s": round(window_end_s - window_seconds, 6),
            "sqi_composite": composite,
            "sqi_snr": sqi.snr_score if sqi is not None else None,
            "snr_db": snr_db,
            "sqi_roi_agreement": sqi.cross_roi_agreement if sqi is not None else None,
            "sqi_landmark_stability": sqi.landmark_stability if sqi is not None else None,
            "sqi_temporal_consistency": sqi.temporal_consistency if sqi is not None else None,
            "harmonic_plausibility": sqi.harmonic_score if sqi is not None else None,
            "flicker_flag": bool(sqi.flicker_suspected) if sqi is not None else False,
            # Непрерывная величина, которую порогует детектор: SNR фонового
            # ROI (дБ) на частоте, совпавшей с лицевой. -inf, когда фон
            # недоступен или неузкополосен — пишется как nan.
            "flicker_score": float(sqi.flicker_background_snr_db) if sqi is not None else None,
            "hard_snr_gate_pass": bool(snr_db is not None and snr_db >= qcfg.min_spectral_snr_db),
            "published_gated": bool(r.publishable),
            "published_weighted_only": bool(composite >= qcfg.min_overall_score_to_publish),
            "bpm_estimate": float(r.bpm),
            "bpm_reference": ref if (want_reference and ref is not None) else None,
            "bpm_injected": injected_bpm if condition in CONDITIONS_WITH_INJECTION else None,
            "method": r.method_used,
            "freq_method": r.frequency_method_used,
            "fs_used_hz": (fs_by_timestamp or {}).get(r.timestamp_ms),
        })
    return out


# --------------------------------------------------------------------------
# Выгрузка
# --------------------------------------------------------------------------
def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # -inf у spectral_snr_db означает «пика в полосе нет вообще» — это
        # ИЗМЕРЕНИЕ, а не пропуск, и оно обязано отличаться в выгрузке от
        # nan («окно вообще не дошло до оценки качества»). Раньше обе
        # ситуации писались как nan и были неразличимы.
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.6g}"
    return str(value)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([_fmt(row.get(f)) for f in fields])


def _median_finite(values) -> float:
    """Медиана только по конечным значениям.

    Для flicker_score это существенно: -inf означает «фонового сигнала у
    детектора не было вообще» (фон отброшен порогом дисперсии либо пик на
    лице не найден). Если включить -inf в медиану, она вырождается в -inf
    у любого условия, где таких окон больше половины, и колонка перестаёт
    различать условия. Доля окон, где данные БЫЛИ, видна по отдельности —
    см. results/degenerate/README.md, раздел «Детектор мерцания».
    """
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def _median(values) -> float:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def summarize(all_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary, percentiles = [], []
    for condition in CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == condition]
        if not rows:
            continue
        n = len(rows)
        gated = [r for r in rows if r["published_gated"]]
        near = ""
        if condition in CONDITIONS_WITH_INJECTION and gated:
            hits = sum(
                1 for r in gated
                if r["bpm_injected"] is not None and r["bpm_estimate"] is not None
                and np.isfinite(r["bpm_estimate"])
                and abs(r["bpm_estimate"] - r["bpm_injected"]) <= NEAR_INJECTED_TOLERANCE_BPM
            )
            near = hits / len(gated)
        mae = mae_pooled = coverage = ""
        if condition in CONDITIONS_WITH_GENUINE_SIGNAL:
            def _errs(subset):
                return [
                    abs(r["bpm_estimate"] - r["bpm_reference"]) for r in subset
                    if r["bpm_reference"] is not None and r["bpm_estimate"] is not None
                    and np.isfinite(r["bpm_estimate"])
                ]
            # Усреднение ПО ЗАПИСЯМ: сперва MAE внутри записи, затем среднее
            # по записям — иначе длинные записи получают больший вес
            # (псевдорепликация, см. benchmark/evaluate.py). Пуловое значение
            # по всем окнам сохраняется отдельной колонкой, чтобы обе
            # величины были видны, а не подменяли друг друга.
            per_record_mae, per_record_cov = [], []
            for rid in sorted({r["record_id"] for r in rows}):
                rec_rows = [r for r in rows if r["record_id"] == rid]
                rec_gated = [r for r in rec_rows if r["published_gated"]]
                rec_errs = _errs(rec_gated)
                if rec_errs:
                    per_record_mae.append(float(np.mean(rec_errs)))
                per_record_cov.append(len(rec_gated) / len(rec_rows))
            pooled = _errs(gated)
            mae = float(np.mean(per_record_mae)) if per_record_mae else float("nan")
            mae_pooled = float(np.mean(pooled)) if pooled else float("nan")
            coverage = float(np.mean(per_record_cov)) if per_record_cov else float("nan")
        summary.append({
            "condition": condition,
            "n_records": len({r["record_id"] for r in rows}),
            "n_windows": n,
            "publish_rate_gated": len(gated) / n,
            "publish_rate_weighted_only": sum(1 for r in rows if r["published_weighted_only"]) / n,
            "median_sqi_composite": _median([r["sqi_composite"] for r in rows]),
            "median_sqi_snr": _median([r["sqi_snr"] for r in rows]),
            "median_snr_db": _median([r["snr_db"] for r in rows]),
            "median_sqi_roi_agreement": _median([r["sqi_roi_agreement"] for r in rows]),
            "median_sqi_landmark_stability": _median([r["sqi_landmark_stability"] for r in rows]),
            "median_sqi_temporal_consistency": _median([r["sqi_temporal_consistency"] for r in rows]),
            "flicker_flag_rate": sum(1 for r in rows if r["flicker_flag"]) / n,
            # «Отказ» = штраф вообще применён: harmonic_plausibility < 1.0
            # означает, что на f/2 найден значимый горб (см. quality.py).
            "harmonic_plausibility_fail_rate": sum(
                1 for r in rows
                if r["harmonic_plausibility"] is not None
                and not np.isnan(r["harmonic_plausibility"])
                and r["harmonic_plausibility"] < 1.0
            ) / n,
            "share_published_near_injected": near,
            "mae_vs_reference": mae,
            "mae_pooled_windows": mae_pooled,
            "coverage": coverage,
        })
        for metric in PERCENTILE_METRICS:
            arr = np.array(
                [r[metric] for r in rows if r[metric] is not None and not np.isnan(r[metric])], dtype=float
            )
            # method="nearest" — берётся фактическое значение выборки, без
            # интерполяции между соседями. Интерполяция по умолчанию даёт
            # nan на выборках с -inf (a + (b-a)*f при a=b=-inf), из-за чего
            # условие, где пик в полосе не найден НИ РАЗУ, показывало бы
            # пустые квартили вместо -inf.
            percentiles.append({
                "condition": condition, "metric": metric,
                "p25": float(np.percentile(arr, 25, method="nearest")) if arr.size else float("nan"),
                "median": float(np.percentile(arr, 50, method="nearest")) if arr.size else float("nan"),
                "p75": float(np.percentile(arr, 75, method="nearest")) if arr.size else float("nan"),
                "n": int(arr.size),
            })
    return summary, percentiles


BOOL_FIELDS = ("flicker_flag", "hard_snr_gate_pass", "published_gated", "published_weighted_only")
STR_FIELDS = ("record_id", "condition", "method", "freq_method")


def load_windows_csv(path: Path) -> list[dict]:
    """Прочитать выгруженный windows.csv обратно в типизированные строки.

    Нужно для --summarize-only: сводки пересобираются из уже посчитанных
    окон, без повторного прогона видео. Пустая ячейка -> None (окно не
    дошло до оценки качества), "-inf"/"inf" -> соответствующая
    бесконечность (пик в полосе не найден — это измерение, а не пропуск).
    """
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for key, value in raw.items():
                if key in STR_FIELDS:
                    row[key] = value
                elif key in BOOL_FIELDS:
                    row[key] = value == "1"
                elif key == "window_idx":
                    row[key] = int(value)
                elif value == "":
                    row[key] = None
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def _src_digest() -> str:
    """Содержимое src/ одним хешем — дополнение к хешу коммита: при
    незакоммиченных изменениях коммит не описывает код полностью.

    Исправление: в прогонах заданий 1 и 4 здесь стояло утверждение, что
    рабочая копия не под git, и git_commit записывался как null. Это было
    неверно — репозиторий под git (проверялась родительская папка). Хеш
    src в тех run_meta.json корректен."""
    h = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        h.update(path.relative_to(REPO_ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _git(args: list[str]) -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    except OSError:
        return None
    return out.stdout.strip() or None


def write_run_meta(path: Path, cfg: PipelineConfig, records, conditions, seed_note: str,
                   preset: str = DEFAULT_PRESET) -> None:
    import mediapipe
    import scipy

    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        "src_sha256": _src_digest(),
        "versions": {
            "python": platform.python_version(),
            "mediapipe": mediapipe.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "preset": preset,
        "seed": seed_note,
        "h264_crf_for_C2": H264_CRF,
        "records": [r for r, _ in records],
        "conditions": list(conditions),
        "pipeline_config": json.loads(json.dumps(asdict(cfg), default=str)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="~/Downloads")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--records", default="", help="список record_id через запятую (по умолчанию все найденные)")
    ap.add_argument("--task0", action="store_true", help="приоритетная проверка детектора мерцания")
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESET_OVERRIDES),
                    help="пресет конфигурации; legacy_welch8 — конфигурация статьи 2 (results/degenerate_v2)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="пересобрать summary.csv/summary_percentiles.csv/run_meta.json из готового windows.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if args.summarize_only and args.task0:
        all_rows = load_windows_csv(out_dir / "task0_windows.csv")
        task0_rows = []
        for condition in TASK0_CONDITIONS:
            for record_id in TASK0_RECORDS:
                rs = [r for r in all_rows if r["condition"] == condition and r["record_id"] == record_id]
                if not rs:
                    continue
                task0_rows.append({
                    "record_id": record_id,
                    "condition": condition,
                    "n_windows": len(rs),
                    "flicker_flag_rate": sum(1 for r in rs if r["flicker_flag"]) / len(rs),
                    "median_flicker_score": _median_finite([r["flicker_score"] for r in rs]),
                })
        write_csv(
            out_dir / "flicker_check.csv",
            ["record_id", "condition", "n_windows", "flicker_flag_rate", "median_flicker_score"],
            task0_rows,
        )
        print(f"flicker_check.csv пересобран из task0_windows.csv ({len(all_rows)} строк)")
        return

    if args.summarize_only:
        all_rows = load_windows_csv(out_dir / "windows.csv")
        summary, percentiles = summarize(all_rows)
        write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)
        write_csv(out_dir / "summary_percentiles.csv", ["condition", "metric", "p25", "median", "p75", "n"], percentiles)
        cfg = make_pipeline_config(args.preset)
        cfg.face.model_asset_path = str(ev.MODEL_PATH)
        present = sorted({r["record_id"] for r in all_rows}, key=_natural_key)
        write_run_meta(
            out_dir / "run_meta.json", cfg, [(r, None) for r in present],
            [c for c in CONDITIONS if any(r["condition"] == c for r in all_rows)],
            "детерминированный: crc32(record_id|condition), см. rppg.degrade.condition_seed",
            preset=args.preset,
        )
        print(f"сводки пересобраны из {out_dir / 'windows.csv'} ({len(all_rows)} строк)")
        return

    records = discover_records(data_dir)
    if not records:
        print(f"ОШИБКА: в {data_dir} не найдено ни одной записи UBFC", file=sys.stderr)
        sys.exit(1)

    conditions = TASK0_CONDITIONS if args.task0 else tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    wanted = TASK0_RECORDS if args.task0 else tuple(r.strip() for r in args.records.split(",") if r.strip())
    if wanted:
        by_id = dict(records)
        missing = [r for r in wanted if r not in by_id]
        if missing:
            print(f"ОШИБКА: записи не найдены в {data_dir}: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
        records = [(r, by_id[r]) for r in wanted]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        print(f"ОШИБКА: неизвестные условия: {', '.join(unknown)}", file=sys.stderr)
        sys.exit(1)

    print(f"Записей: {len(records)}, условий: {len(conditions)} -> прогонов: {len(records) * len(conditions)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for condition in conditions:
        print(f"--- условие {condition} ---", flush=True)
        for record_id, subject_dir in records:
            all_rows.extend(run_condition(record_id, subject_dir, condition, preset=args.preset))
        # Промежуточная выгрузка после каждого условия: полный прогон идёт
        # часами, и падение на последнем условии не должно обнулять всё
        # посчитанное. Файл перезаписывается целиком и в конце заменяется
        # итоговым windows.csv.
        if not args.task0:
            # В режиме задачи 0 промежуточная выгрузка не нужна: её полный
            # аналог сохраняется как task0_windows.csv.
            write_csv(out_dir / "windows_partial.csv", WINDOW_FIELDS, all_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.task0:
        task0_rows = []
        for condition in conditions:
            for record_id, _ in records:
                rows = [r for r in all_rows if r["condition"] == condition and r["record_id"] == record_id]
                if not rows:
                    continue
                task0_rows.append({
                    "record_id": record_id,
                    "condition": condition,
                    "n_windows": len(rows),
                    "flicker_flag_rate": sum(1 for r in rows if r["flicker_flag"]) / len(rows),
                    "median_flicker_score": _median_finite([r["flicker_score"] for r in rows]),
                })
        # Пооконные строки задачи 0 сохраняются отдельным файлом: по ним
        # проверяется соответствие G0 опубликованным числам ещё ДО того,
        # как запускать полный прогон (критерий приёмки 1).
        write_csv(out_dir / "task0_windows.csv", WINDOW_FIELDS, all_rows)
        path = out_dir / "flicker_check.csv"
        write_csv(path, ["record_id", "condition", "n_windows", "flicker_flag_rate", "median_flicker_score"], task0_rows)
        print(f"\n{'запись':>10s}{'условие':>10s}{'окон':>7s}{'доля флага':>13s}{'медиана flicker_score, дБ':>28s}")
        for row in task0_rows:
            print(f"{row['record_id']:>10s}{row['condition']:>10s}{row['n_windows']:>7d}"
                  f"{row['flicker_flag_rate']:>13.3f}{row['median_flicker_score']:>28.2f}")
        print(f"\nвыгружено: {path}")
        return

    write_csv(out_dir / "windows.csv", WINDOW_FIELDS, all_rows)
    summary, percentiles = summarize(all_rows)
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)
    write_csv(out_dir / "summary_percentiles.csv", ["condition", "metric", "p25", "median", "p75", "n"], percentiles)
    cfg = make_pipeline_config(args.preset)
    cfg.face.model_asset_path = str(ev.MODEL_PATH)
    write_run_meta(
        out_dir / "run_meta.json", cfg, records, conditions,
        "детерминированный: crc32(record_id|condition), см. rppg.degrade.condition_seed",
        preset=args.preset,
    )
    print(f"\nвыгружено в {out_dir}: windows.csv ({len(all_rows)} строк), summary.csv, "
          f"summary_percentiles.csv, run_meta.json")


if __name__ == "__main__":
    main()
