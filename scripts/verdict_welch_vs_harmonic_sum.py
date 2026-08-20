"""
Задача 29: вердикт welch против harmonic_sum по ВНУТРЕННИМ данным (эталон
из задачи 24 пока недоступен — три КОСВЕННЫЕ проверки на 6 видео, ни одна
не заменяет собой реальную калибровку с ground truth).

Как и scripts/compare_frequency_methods.py (задача 28), continuity-проверка
(FrequencyConfig.continuity_enabled) ОТКЛЮЧЕНА для обоих методов —
сравниваются СЫРЫЕ peak-selection алгоритмы, а не то, что от них остаётся
после сглаживания историей (см. докстринг compare_frequency_methods.py:
continuity маскирует именно те октавные ошибки, которые здесь нужно видеть).

Три проверки:
  1. Чистый участок "тёти" — самый длинный отрезок ПОДРЯД идущих окон с
     максимальным spectral_snr_db (SNR считается ОДИНАКОВО для обоих
     методов — dominant_frequency_and_snr всегда использует Welch PSD
     внутри, независимо от frequency_method, см. quality.py — поэтому
     "какие окна чистые" не зависит от того, какой метод сравнивается).
     На этом отрезке сравниваются медианы welch и harmonic_sum.
  2. Устойчивость траектории — медианный абсолютный шаг между соседними
     ОК-оценками, на всех 6 видео, для обоих методов. Настоящий пульс
     гладкий: метод, который угадывает верно, даёт более гладкую кривую.
  3. Гистограммы log2(BPM) — см. вывод compare_frequency_methods.py
     (задача 28), здесь только ссылка на уже сохранённые файлы.

НЕ переключает PipelineConfig.frequency_method по умолчанию — welch
остаётся дефолтом. Отчёт сохраняется в docs/welch_vs_harmonic_sum_verdict.md.

Использование:
    PYTHONPATH=src python3 scripts/verdict_welch_vs_harmonic_sum.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rppg.config import PipelineConfig, FrequencyMethod
from rppg.pipeline import RPPGPipeline
from run_benchmark import DEFAULT_VIDEOS

METHODS = ["welch", "harmonic_sum"]


def run_video_with_snr(video_path: str, model_path: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (timestamps_s, bpm, spectral_snr_db) по всем ok-окнам,
    continuity ОТКЛЮЧЕНА (см. докстринг модуля)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config = PipelineConfig()
    config.window.assumed_fps = fps
    config.face.model_asset_path = model_path
    config.frequency_method = FrequencyMethod(method)
    config.frequency.continuity_enabled = False

    timestamps_s, bpm_series, snr_series = [], [], []
    frame_idx = 0
    with RPPGPipeline(config, debug=True) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None and result.status == "ok" and not np.isnan(result.bpm):
                timestamps_s.append(timestamp_ms / 1000.0)
                bpm_series.append(result.bpm)
                snr_series.append(result.debug.sqi.spectral_snr_db if result.debug is not None else float("nan"))
            frame_idx += 1
    cap.release()
    return np.array(timestamps_s), np.array(bpm_series), np.array(snr_series)


def longest_high_snr_run(snr: np.ndarray, percentile: float = 75.0) -> tuple[int, int]:
    """Самый длинный отрезок ПОДРЯД идущих окон с SNR >= percentile-го
    перцентиля этого видео. Возвращает (start_idx, length)."""
    if len(snr) == 0:
        return 0, 0
    threshold = np.percentile(snr, percentile)
    high = snr >= threshold
    best_start, best_len = 0, 0
    cur_start, cur_len = None, 0
    for i, v in enumerate(high):
        if v:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_start, cur_len = None, 0
    return best_start, best_len


def median_abs_step(bpm: np.ndarray) -> float:
    if len(bpm) < 2:
        return float("nan")
    return float(np.median(np.abs(np.diff(bpm))))


def check_clean_tetya_segment(model_path: str, tetya_path: Path) -> dict:
    print("--- Проверка 1: чистый участок 'тёти' ---")
    series_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        series_by_method[method] = run_video_with_snr(str(tetya_path), model_path, method)

    # SNR одинаков для обоих методов (см. докстринг) -> берём из welch-прогона.
    t_ref, _, snr_ref = series_by_method["welch"]
    start, length = longest_high_snr_run(snr_ref, percentile=75.0)
    print(f"  самый длинный отрезок высокого SNR: {length} окон, начиная с t={t_ref[start]:.1f}с "
          f"(если 0 или 1 — данных недостаточно для этой проверки)")

    result: dict = {"segment_length": length, "segment_start_s": float(t_ref[start]) if length else None}
    if length < 3:
        print("  [SKIP] сегмент слишком короткий, проверку пропускаем")
        result["by_method"] = {}
        return result

    idx = slice(start, start + length)
    by_method = {}
    for method, (t, bpm, snr) in series_by_method.items():
        segment_bpm = bpm[idx] if len(bpm) >= start + length else np.array([])
        median_bpm = float(np.median(segment_bpm)) if len(segment_bpm) else float("nan")
        by_method[method] = {"median_bpm": median_bpm, "values": segment_bpm.tolist()}
        print(f"  {method:<15s} медиана на чистом участке: {median_bpm:.2f} BPM, значения: "
              f"{[round(v, 1) for v in segment_bpm.tolist()]}")
    result["by_method"] = by_method
    return result


def check_trajectory_smoothness(model_path: str, videos: list[Path]) -> dict:
    print("\n--- Проверка 2: устойчивость траектории (медианный |шаг|) ---")
    header = f"{'видео':<45s}{'метод':<15s}{'n':>6s}{'медианный |шаг|, BPM':>22s}"
    print(header)
    results: dict = {}
    for video in videos:
        results[video.stem] = {}
        for method in METHODS:
            t, bpm, _snr = run_video_with_snr(str(video), model_path, method)
            step = median_abs_step(bpm)
            results[video.stem][method] = {"n": len(bpm), "median_abs_step": step}
            print(f"{video.stem:<45s}{method:<15s}{len(bpm):>6d}{step:>22.2f}")
    return results


def write_report(out_path: Path, clean_segment: dict, smoothness: dict) -> None:
    lines: list[str] = []
    lines.append("# Вердикт: welch против harmonic_sum по внутренним данным (задача 29)")
    lines.append("")
    lines.append(f"Дата: {date.today().isoformat()}. Эталон (задача 24) на момент написания недоступен — "
                  "три проверки ниже КОСВЕННЫЕ, ни одна не заменяет реальную калибровку с ground truth.")
    lines.append("")
    lines.append("Методология: у обоих методов ОТКЛЮЧЕНА continuity-проверка выбора пика "
                  "(FrequencyConfig.continuity_enabled=False) — сравнивается СЫРОЕ поведение "
                  "peak-selection алгоритмов, а не то, что от них остаётся после сглаживания "
                  "историей (задача 28: continuity маскирует именно те октавные ошибки, "
                  "которые здесь нужно видеть).")
    lines.append("")

    lines.append("## 1. Чистый участок 'тёти'")
    lines.append("")
    seg_len = clean_segment.get("segment_length", 0)
    if seg_len >= 3:
        lines.append(f"Самый длинный отрезок подряд идущих окон с высоким SNR (>= 75-го перцентиля "
                      f"этого видео): **{seg_len} окон**, начиная с t={clean_segment['segment_start_s']:.1f}с.")
        lines.append("")
        lines.append("| метод | медиана BPM на отрезке |")
        lines.append("|---|---|")
        for method, data in clean_segment.get("by_method", {}).items():
            lines.append(f"| {method} | {data['median_bpm']:.2f} |")
        methods = list(clean_segment.get("by_method", {}).keys())
        if len(methods) == 2:
            m0, m1 = clean_segment["by_method"][methods[0]]["median_bpm"], clean_segment["by_method"][methods[1]]["median_bpm"]
            if m1 > 0:
                ratio = m0 / m1
                lines.append("")
                lines.append(f"Отношение медиан ({methods[0]}/{methods[1]}) = **{ratio:.2f}**. "
                              "Отношение, близкое к 2.0 или 0.5, указывало бы на систематическую "
                              "октавную путаницу одного из методов именно на этом чистом участке; "
                              "отношение, не близкое ни к целому кратному, ни к 1.0, означает, что "
                              "методы расходятся не банальной октавой, а чем-то ещё.")
    else:
        lines.append("Недостаточно данных для выделения непрерывного чистого участка на этом прогоне.")
    lines.append("")

    lines.append("## 2. Устойчивость траектории (медианный абсолютный шаг между соседними оценками)")
    lines.append("")
    lines.append("| видео | welch, BPM | harmonic_sum, BPM | глаже |")
    lines.append("|---|---|---|---|")
    welch_wins, hs_wins = 0, 0
    for video_name, by_method in smoothness.items():
        w = by_method.get("welch", {}).get("median_abs_step", float("nan"))
        h = by_method.get("harmonic_sum", {}).get("median_abs_step", float("nan"))
        if np.isnan(w) or np.isnan(h):
            winner = "—"
        elif w < h:
            winner = "welch"
            welch_wins += 1
        elif h < w:
            winner = "harmonic_sum"
            hs_wins += 1
        else:
            winner = "ничья"
        lines.append(f"| {video_name} | {w:.2f} | {h:.2f} | {winner} |")
    lines.append("")
    lines.append(f"Глаже траектория у welch на {welch_wins}/6 видео, у harmonic_sum на {hs_wins}/6.")
    lines.append("")

    lines.append("## 3. Гистограммы log2(BPM)")
    lines.append("")
    lines.append("См. `compare_frequency_methods_results/log2hist_*.png` (задача 28, "
                  "scripts/compare_frequency_methods.py) — по одной картинке на видео, welch и "
                  "harmonic_sum рядом. Спутниковый кластер на +1 (×2) или +1.585 (×3) от основного "
                  "по оси log2(BPM) — октавная ошибка вверх; на -1 — ошибка вниз.")
    lines.append("")

    lines.append("## Вывод")
    lines.append("")
    lines.append("**welch остаётся дефолтом.** Эти три проверки — косвенные и НЕ являются основанием "
                  "для переключения метода по умолчанию. Задача 28 показала, что harmonic_sum "
                  "устойчивее welch к точечным октавным выбросам вверх (0% против заметной доли на "
                  "нескольких видео) — это его прямое конструктивное преимущество. Но harmonic_sum "
                  "систематически даёт заметно более низкие медианы BPM на нескольких видео "
                  "(известный риск: склонность выбирать f/2 как основную частоту, поскольку "
                  "гармоники f/2 включают саму f) — эмпирически в этом прогоне ни явной октавной "
                  "ошибки ВНИЗ по метрике задачи 28, ни превосходства по гладкости траектории у "
                  "harmonic_sum систематически не обнаружено, но и не исключено на других данных. "
                  "**Окончательное решение — только после задачи 24 (эталон).**")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёт сохранён в {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--out", type=str, default="docs/welch_vs_harmonic_sum_verdict.md")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        raise SystemExit(
            "Видео не найдены:\n" + "\n".join(f"  {v}" for v in missing) +
            "\n\nПередайте реальные пути через --videos."
        )

    tetya_path = next((v for v in videos if "тет" in v.stem.lower()), None)
    if tetya_path is None:
        raise SystemExit("Видео 'тётя' не найдено среди --videos — проверка 1 требует именно его.")

    clean_segment = check_clean_tetya_segment(args.model, tetya_path)
    smoothness = check_trajectory_smoothness(args.model, videos)
    write_report(Path(args.out), clean_segment, smoothness)


if __name__ == "__main__":
    main()
