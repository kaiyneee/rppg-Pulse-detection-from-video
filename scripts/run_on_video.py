"""
Офлайн-обработка видеофайла: извлекает временной ряд BPM/HRV/SQI и
сохраняет в CSV — удобно для последующей подачи в систему анализа ПТСР
или для построения графиков (см. benchmark/evaluate.py для сравнения
с референсным сигналом при наличии ground truth).

Конфиг эксперимента сохраняется РЯДОМ с результатом (--out result.csv ->
result.config.yaml) — п.42 требований: "каждая строчка таблицы в статье
должна быть привязана к конкретному конфигу", а не к неявному состоянию
кода/CLI-флагов на момент запуска.

Задача 48 (двухпроходный режим): --two-pass включён по умолчанию —
единственное расхождение между "я пульс 125" (истина требует коррекции
первого окна) и "я обычное" (не требует) — это пустая история в самом
начале записи, и по ОДНОМУ окну без контекста они математически
неразличимы (доказанный предел причинного правила, см. задачи 46/47). Для
уже ЗАПИСАННОГО видео этого предела нет — файл можно пройти дважды: первый
проход как обычно (причинно, вперёд по времени, копит историю
подтверждённых окон), второй — без пересчёта сигнала и БЕЗ какого-либо
усреднения по времени — пересматривает те окна, которым в первом проходе
не хватило истории для решения, используя итоговую историю целиком.
Задача 49, п.3: второй проход сначала пробует пересмотреть САМ ВЫБОР
кандидата {f, 2f, f/2} против медианы итоговой истории (см.
rppg.hrv.features.reconsider_candidate_against_history) — не только
"отличается ли ровно в 2 раза", как раньше; если это не даёт уверенного
ответа, откатывается к прежнему грубому флипу apply_history_ratio_check.
В scripts/run_webcam.py (реальное время) такого флага нет и не может
быть — там причинность обязательна.

Использование:
    python scripts/run_on_video.py --video path/to/video.mp4 --out result.csv
    python scripts/run_on_video.py --video path/to/video.mp4 --config my_experiment.yaml --out result.csv
    python scripts/run_on_video.py --video path/to/video.mp4 --out result.csv --no-two-pass
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.config import DEFAULT_PRESET, PRESET_OVERRIDES, make_pipeline_config
from rppg.config_io import load_config, save_config
from rppg.pipeline import RPPGPipeline, apply_second_pass
from rppg.structured_log import read_jsonl_records


def _patch_jsonl_second_pass(log_path: str, corrections: dict[int, float]) -> None:
    """Задача 48: JSONL уже записан ПОТОКОВО (по одному окну за раз) во
    время первого прохода — раньше второго прохода просто не существовало,
    переписывать в момент записи было нечего. Здесь файл читается целиком
    и правится ТОЧЕЧНО: только bpm и pass_that_decided у окон из
    corrections (timestamp_ms -> новый bpm), остальные поля (включая
    bpm_aggregate — задача 48 явно не пересчитывает агрегат задним числом,
    только само решение о коррекции) остаются такими, какими их видел
    первый проход."""
    headers, records = read_jsonl_records(log_path)
    for record in records:
        ts = record.get("timestamp_ms")
        if ts in corrections:
            record["bpm"] = corrections[ts]
            record["pass_that_decided"] = 2
    with open(log_path, "w", encoding="utf-8") as f:
        for obj in headers:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн rPPG обработка видео")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Базовый YAML/JSON-конфиг (см. rppg.config_io) — --method/--freq, если заданы, переопределяют его.",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESET_OVERRIDES),
                        help="Пресет конфигурации, если не задан --config; legacy_welch8 — конфигурация статей")
    parser.add_argument("--method", type=str, default=None, choices=[m.value for m in ExtractionMethod])
    parser.add_argument("--freq", type=str, default=None, choices=[m.value for m in FrequencyMethod])
    parser.add_argument("--out", type=str, default="rppg_result.csv")
    parser.add_argument(
        "--log", type=str, default=None,
        help="Путь к JSONL-логу по окнам (timestamp/BPM по ROI/компоненты SQI/warnings, см. п.43)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Дозаписать в существующий --log вместо перезаписи (задача 26; по умолчанию файл "
             "перезаписывается — прошлый баг молча склеивал записи от разных прогонов/конфигов).",
    )
    parser.add_argument(
        "--two-pass", action=argparse.BooleanOptionalAction, default=True,
        help="Задача 48: второй проход пересматривает (БЕЗ пересчёта сигнала и без усреднения по "
             "времени) окна, которым в первом (причинном) проходе не хватило истории для решения о "
             "коррекции, используя итоговую историю целиком. Включён по умолчанию для офлайн-обработки; "
             "--no-two-pass — прежнее строго причинное поведение.",
    )
    args = parser.parse_args()

    # Явный --config задаёт всё сам (файл без поля preset достраивается от
    # legacy_welch8, см. config_io); иначе — конфиг от пресета.
    config = load_config(args.config) if args.config else make_pipeline_config(args.preset)
    config.face.model_asset_path = args.model
    if args.method is not None:
        config.method = ExtractionMethod(args.method)
    if args.freq is not None:
        config.frequency_method = FrequencyMethod(args.freq)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps

    rows = []
    frame_idx = 0
    n_reconsidered = 0
    n_changed = 0
    n_changed_by_candidate = 0
    n_changed_by_ratio = 0
    jsonl_corrections: dict[int, float] = {}

    # Задача 48: --two-pass читает history_was_insufficient/cross_check
    # ЧЕРЕЗ debug=True (см. pipeline.DebugInfo) — это НЕ часть публичного
    # контракта PTSDPulseFeatures, только диагностическая надстройка,
    # существующая именно для таких офлайн-скриптов.
    with RPPGPipeline(config, log_path=args.log, log_append=args.append, debug=args.two_pass) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                rows.append(result)
            frame_idx += 1

        if args.two_pass:
            # Вся логика второго прохода — в rppg.pipeline.apply_second_pass
            # (единственная реализация, общая с scripts/eval_ubfc_subject.py
            # --two-pass; см. её docstring про оба правила коррекции).
            stats = apply_second_pass(
                rows, pipeline._trusted_bpm_history, pipeline.cfg.frequency.history_min_windows,
            )
            n_reconsidered = stats.n_reconsidered
            n_changed = stats.n_changed
            n_changed_by_candidate = stats.n_changed_by_candidate
            n_changed_by_ratio = stats.n_changed_by_ratio
            jsonl_corrections = stats.corrections

    cap.release()

    if args.two_pass and args.log and jsonl_corrections:
        _patch_jsonl_second_pass(args.log, jsonl_corrections)

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms", "bpm", "status",
            "sqi_score", "sqi_level", "sqi_spectral_snr_db",
            # Задача 25 (пересмотрено 2026-08-18): контракт с моделью ПТСР —
            # РОВНО ДВЕ колонки, не пять. bpm_aggregate — всегда реально
            # измеренное число (или NaN); bpm_aggregate_quality — одно
            # число [0,1], 0.0 означает "использовать нельзя". Устаревание
            # агрегат берёт на себя через ПЛАВНО убывающее quality (см.
            # RPPGPipeline._compute_aggregate) — отдельные last_valid_bpm/
            # last_valid_bpm_age_s/publishable для этого больше не нужны и
            # убраны из CSV (они дублировали более старый, ступенчатый
            # механизм устаревания — задача 8). window_status/n_windows/
            # spread_bpm, стоявшие за bpm_aggregate, остаются ТОЛЬКО в
            # JSONL-логе (см. structured_log.WindowLogRecord) — это
            # диагностика для отладки, не часть контракта потребителя.
            "bpm_aggregate", "bpm_aggregate_quality",
            "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "pnn20_pct",
            "lf_hf_ratio", "method_used", "frequency_method_used", "warnings",
        ])
        for r in rows:
            hrv = r.hrv
            writer.writerow([
                r.timestamp_ms, r.bpm, r.status,
                f"{r.sqi_score:.3f}", r.sqi_level, r.confidence_breakdown.get("spectral_snr_db", ""),
                r.bpm_aggregate, f"{r.bpm_aggregate_quality:.3f}",
                hrv.mean_hr_bpm if hrv else "", hrv.sdnn_ms if hrv else "",
                hrv.rmssd_ms if hrv else "", hrv.pnn50_pct if hrv else "",
                hrv.pnn20_pct if hrv else "", hrv.lf_hf_ratio if hrv else "",
                r.method_used, r.frequency_method_used, " ; ".join(r.warnings),
            ])

    config_out_path = out_path.with_suffix("").with_suffix(".config.yaml")
    save_config(config, config_out_path)

    n_publishable = sum(1 for r in rows if r.publishable)
    print(f"Обработано окон: {len(rows)}, из них publishable (SQI ok): {n_publishable}")
    if args.two_pass:
        print(f"Второй проход: пересмотрено окон (недостаточная история в первом проходе) — "
              f"{n_reconsidered}, из них изменено значение bpm — {n_changed} "
              f"(пересмотром кандидата — {n_changed_by_candidate}, флипом x2 — {n_changed_by_ratio})")
    print(f"Результат сохранён в {out_path}")
    print(f"Конфиг эксперимента сохранён в {config_out_path}")


if __name__ == "__main__":
    main()
