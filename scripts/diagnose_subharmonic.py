"""
Задача 42, часть 2: диагностическая матрица 4 конфигурации x 2 видео,
различающая механизм A (субгармоническая ошибка выбора пика) от механизма B
(залипание continuity на ошибочной первой оценке) — см. докстринг-таблицу
интерпретации в задаче 42. НЕ вносит правок в код — только существующие
поля конфига.

    Ячейка                    filt.low_hz   continuity_enabled   max_physiological_change_bpm_per_sec
    A_base                    0.7           True                 10.0
    B_narrow                  1.3           True                 10.0
    C_no_continuity           0.7           False                1000.0
    D_narrow_no_continuity    1.3           False                1000.0

1.3 Гц = 78 BPM: ниже такого пульса сразу после приседаний физиологически
быть не может, а субгармоника 62 BPM оказывается ЗА границей полосы поиска.
1000.0 BPM/с у continuity_enabled=False фактически ничего не ограничивает
(параметр действует, только если continuity_enabled=True и есть история) —
здесь он для явной симметрии со строкой A/B, а выключение continuity само по
себе снимает и мягкое, и (в реализации задачи 37) жёсткое ограничение сразу.

"покой" (я обычное.mov) — КОНТРОЛЬ: истинный пульс покоя ~62 BPM, поэтому
при узкой полосе (B/D, низ 78 BPM) SNR там ОБЯЗАН обвалиться — без этого
результат на приседаниях (A/B) неинтерпретируем (см. часть 3 задачи 42).

Использование:
    PYTHONPATH=src python3 scripts/diagnose_subharmonic.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.config import PipelineConfig
from rppg.pipeline import RPPGPipeline

VIDEOS = {
    "приседания": "/Users/ylhjn/Desktop/проекты/статья пульс по видо/я 20 приседаний.mov",
    "покой": "/Users/ylhjn/Desktop/проекты/статья пульс по видо/я обычное.mov",
}

CELLS = {
    "A_base": dict(low_hz=0.7, continuity_enabled=True, max_change=10.0),
    "B_narrow": dict(low_hz=1.3, continuity_enabled=True, max_change=10.0),
    "C_no_continuity": dict(low_hz=0.7, continuity_enabled=False, max_change=1000.0),
    "D_narrow_no_continuity": dict(low_hz=1.3, continuity_enabled=False, max_change=1000.0),
}

MODEL_PATH = str(REPO_ROOT / "models" / "face_landmarker.task")
OUT_DIR = REPO_ROOT / "diagnose_subharmonic_results"


def run_cell(video_path: str, cell_params: dict, out_stem: str) -> dict:
    cfg = PipelineConfig()
    cfg.filt.low_hz = cell_params["low_hz"]
    cfg.frequency.continuity_enabled = cell_params["continuity_enabled"]
    cfg.frequency.max_physiological_change_bpm_per_sec = cell_params["max_change"]
    cfg.face.model_asset_path = MODEL_PATH

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.window.assumed_fps
    cfg.window.assumed_fps = fps

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{out_stem}.csv"
    jsonl_path = OUT_DIR / f"{out_stem}.jsonl"

    rows = []
    frame_idx = 0
    with RPPGPipeline(cfg, log_path=str(jsonl_path)) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                rows.append(result)
            frame_idx += 1
    cap.release()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "bpm", "status", "publishable", "sqi_score", "spectral_snr_db"])
        for r in rows:
            writer.writerow([
                r.timestamp_ms, r.bpm, r.status, r.publishable, f"{r.sqi_score:.3f}",
                r.confidence_breakdown.get("spectral_snr_db", ""),
            ])

    bpm_values = np.array([r.bpm for r in rows if not np.isnan(r.bpm)])
    snr_values = np.array([r.confidence_breakdown.get("spectral_snr_db", float("nan")) for r in rows], dtype=float)
    snr_values = snr_values[np.isfinite(snr_values)]
    n_pub = sum(1 for r in rows if r.publishable)

    return {
        "n_windows": len(rows),
        "n_publishable": n_pub,
        "coverage": (n_pub / len(rows)) if rows else float("nan"),
        "median_bpm": float(np.median(bpm_values)) if len(bpm_values) else float("nan"),
        "median_snr_db": float(np.median(snr_values)) if len(snr_values) else float("nan"),
    }


def main() -> None:
    missing = [v for v in VIDEOS.values() if not Path(v).exists()]
    if missing:
        print("Видео недоступны локально, диагностика невозможна:")
        for m in missing:
            print(f"  - {m}")
        return

    print(f"{'ячейка':<24s}{'видео':<14s}{'median BPM':>12s}{'median SNR дБ':>15s}{'покрытие':>10s}")
    summary_rows = []
    for cell_name, params in CELLS.items():
        for video_name, video_path in VIDEOS.items():
            out_stem = f"{cell_name}__{video_name}"
            summary = run_cell(video_path, params, out_stem)
            print(f"{cell_name:<24s}{video_name:<14s}{summary['median_bpm']:>12.1f}"
                  f"{summary['median_snr_db']:>15.2f}{summary['coverage'] * 100:>9.1f}%")
            summary_rows.append((cell_name, video_name, summary))

    summary_csv = OUT_DIR / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "video", "n_windows", "n_publishable", "coverage", "median_bpm", "median_snr_db"])
        for cell_name, video_name, s in summary_rows:
            writer.writerow([
                cell_name, video_name, s["n_windows"], s["n_publishable"],
                f"{s['coverage']:.3f}", f"{s['median_bpm']:.2f}", f"{s['median_snr_db']:.2f}",
            ])
    print(f"\nСводная таблица сохранена в {summary_csv}")


if __name__ == "__main__":
    main()
