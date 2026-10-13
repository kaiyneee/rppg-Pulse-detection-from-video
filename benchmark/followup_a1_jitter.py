"""
Задача B: дрожание лицевых ориентиров на НЕИЗМЕННОМ входе (условие A1).

Кадры A1 строятся тем же кодом, которым собирался прогон задания 1
(rppg.degrade.prepare_record + FrameDegrader) — новая логика заморозки не
вводится. Детектор ориентиров прогоняется ОТДЕЛЬНО от пайплайна: ни SQI,
ни оценка частоты здесь не участвуют, меряется только геометрия.

Смещение точки за кадр — евклидово расстояние (в пикселях) от МЕДИАННОЙ
по всем кадрам позиции этой же точки. Медиана, а не позиция на первом
кадре: первый кадр ничем не выделен, и привязка к нему завысила бы
смещения, если именно он оказался выбросом.

    PYTHONPATH=src venv/bin/python3 benchmark/followup_a1_jitter.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from rppg.config import PipelineConfig  # noqa: E402
from rppg.degrade import FrameDegrader, prepare_record  # noqa: E402
from rppg.face.landmarker import FaceLandmarkerWrapper  # noqa: E402
from rppg.face.roi import ROI_LANDMARK_INDICES, landmarks_to_pixels  # noqa: E402

import degenerate as dg  # noqa: E402
import eval_ubfc_subject as ev  # noqa: E402

N_FRAMES = 300
REGIONS = ("forehead", "left_cheek", "right_cheek")


def published_records(followup_dir: Path) -> set[str]:
    path = followup_dir / "a1_published_breakdown.csv"
    with path.open(encoding="utf-8") as fh:
        return {r["record_id"] for r in csv.DictReader(fh) if int(r["n_windows_published"]) > 0}


def run_record(record_id: str, subject_dir: Path, cfg: PipelineConfig):
    video = str(ev.find_video(subject_dir))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.window.assumed_fps
    cap.release()

    prep = prepare_record(video, "A1", float(fps))
    degrader = FrameDegrader("A1", record_id, float(fps), prep)
    frame = degrader(None, 0)

    landmarker = FaceLandmarkerWrapper(
        model_asset_path=cfg.face.model_asset_path, delegate=cfg.face.delegate,
    )
    per_frame_px: list[np.ndarray | None] = []
    try:
        for idx in range(N_FRAMES):
            res = landmarker.detect(frame, int(idx / fps * 1000))
            if res is None or res.landmarks_norm is None:
                per_frame_px.append(None)
            else:
                per_frame_px.append(landmarks_to_pixels(res.landmarks_norm, frame.shape))
    finally:
        landmarker.close()

    detected = [p for p in per_frame_px if p is not None]
    medians = {}
    for region in REGIONS:
        idx = ROI_LANDMARK_INDICES[region]
        if detected:
            stack = np.stack([p[idx][:, :2] for p in detected])  # (n_frames, n_pts, 2)
            medians[region] = np.median(stack, axis=0)
        else:
            medians[region] = None

    rows = []
    for frame_idx, px in enumerate(per_frame_px):
        for region in REGIONS:
            row = {"record_id": record_id, "roi_region": region, "frame_idx": frame_idx,
                   "mean_abs_px_shift_from_median": "", "max_abs_px_shift_from_median": ""}
            if px is not None and medians[region] is not None:
                pts = px[ROI_LANDMARK_INDICES[region]][:, :2]
                dist = np.linalg.norm(pts - medians[region], axis=1)
                row["mean_abs_px_shift_from_median"] = f"{float(dist.mean()):.6g}"
                row["max_abs_px_shift_from_median"] = f"{float(dist.max()):.6g}"
            rows.append(row)
    n_lost = sum(1 for p in per_frame_px if p is None)
    return rows, n_lost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="~/Downloads")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate" / "followup"))
    ap.add_argument("--records", default="", help="подмножество record_id через запятую (по умолчанию все)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    cfg.face.model_asset_path = str(ev.MODEL_PATH)

    records = dg.discover_records(Path(args.data_dir).expanduser())
    if args.records:
        wanted = [r.strip() for r in args.records.split(",") if r.strip()]
        by_id = dict(records)
        records = [(r, by_id[r]) for r in wanted]

    pub = published_records(out_dir)
    all_rows, summary = [], []
    for record_id, subject_dir in records:
        rows, n_lost = run_record(record_id, subject_dir, cfg)
        all_rows.extend(rows)
        for region in REGIONS:
            vals = np.array([
                float(r["mean_abs_px_shift_from_median"]) for r in rows
                if r["roi_region"] == region and r["mean_abs_px_shift_from_median"] != ""
            ])
            summary.append({
                "record_id": record_id,
                "roi_region": region,
                "median_shift_px": f"{np.median(vals):.6g}" if vals.size else "",
                "p95_shift_px": f"{np.percentile(vals, 95):.6g}" if vals.size else "",
                "max_shift_px": f"{vals.max():.6g}" if vals.size else "",
                "matches_published": "1" if record_id in pub else "0",
            })
        med = summary[-3]["median_shift_px"]
        print(f"  {record_id:16s} кадров без лица {n_lost:3d}, медиана смещения (лоб) {med} px", flush=True)

    def write(path: Path, fields, data):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write(out_dir / "a1_landmark_jitter.csv", list(all_rows[0].keys()), all_rows)
    write(out_dir / "a1_jitter_summary.csv", list(summary[0].keys()), summary)
    print(f"\nвыгружено: {len(all_rows)} строк в a1_landmark_jitter.csv, {len(summary)} в a1_jitter_summary.csv")


if __name__ == "__main__":
    main()
