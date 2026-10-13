"""
Кэш покадровых средних по МЕЛКИМ участкам кожи внутри трёх штатных ROI.

Зачем. Главный выигрыш точности дало усреднение нормированных спектров
трёх ROI: шум разных участков лица независим, пульс общий. Естественный
следующий вопрос — даёт ли выигрыш более мелкое разбиение: лоб на 3 полосы
по горизонтали, каждая щека на сетку 2x2 — всего 11 участков.

Участки строятся из ТОЙ ЖЕ маски, что штатный ROI (полигон по landmark-
точкам ∩ кэшированная маска кожи пайплайна), и делятся по квантилям
координат пикселей — так у участков одинаковое число пикселей. Среднее
считается той же roi_mean_rgb (обрезка выбросов, порог валидности).
Пайплайн при этом НЕ меняется: он прогоняется штатно, участки считаются
обёрткой вокруг extract_rois из её же аргументов.

    PYTHONPATH=src venv/bin/python3 benchmark/patch_cache.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import rppg.pipeline as pipeline_mod  # noqa: E402
from rppg.face.roi import build_roi_mask, landmarks_to_pixels, roi_mean_rgb  # noqa: E402
from rppg.pipeline import RPPGPipeline  # noqa: E402

import eval_ubfc_subject as ev  # noqa: E402
import trace_cache as tc  # noqa: E402

OUT_DIR = REPO_ROOT / "cache" / "patches"
PATCH_SPLITS = {"forehead": (1, 3), "left_cheek": (2, 2), "right_cheek": (2, 2)}  # (строк, столбцов)


def patch_names() -> list[str]:
    return [f"{roi}_{i}{j}" for roi, (r, c) in PATCH_SPLITS.items() for i in range(r) for j in range(c)]


def split_mask(mask: np.ndarray, rows: int, cols: int) -> list[np.ndarray | None]:
    ys, xs = np.nonzero(mask)
    if len(ys) < rows * cols * 4:
        return [None] * (rows * cols)
    y_edges = np.quantile(ys, np.linspace(0, 1, rows + 1))
    x_edges = np.quantile(xs, np.linspace(0, 1, cols + 1))
    out = []
    for i in range(rows):
        y_sel = (ys >= y_edges[i]) & ((ys < y_edges[i + 1]) if i < rows - 1 else (ys <= y_edges[i + 1]))
        for j in range(cols):
            x_sel = (xs >= x_edges[j]) & ((xs < x_edges[j + 1]) if j < cols - 1 else (xs <= x_edges[j + 1]))
            sel = y_sel & x_sel
            m = np.zeros(mask.shape, dtype=np.uint8)
            m[ys[sel], xs[sel]] = 1
            out.append(m)
    return out


def build_record(record_id: str, subject_dir: Path) -> dict:
    video_path = ev.find_video(subject_dir)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg = tc.make_config(fps)
    names = patch_names()
    rgb = np.zeros((0, len(names), 3))
    rows_rgb, rows_valid, detected = [], [], []
    holder: dict = {}
    original_extract = pipeline_mod.extract_rois

    def recording_extract(frame, landmarks_norm, **kw):
        res = original_extract(frame, landmarks_norm, **kw)
        px = landmarks_to_pixels(landmarks_norm, frame.shape)
        skin = kw.get("skin_mask")
        vals, oks = [], []
        for roi, (r, c) in PATCH_SPLITS.items():
            base = build_roi_mask(px, roi, frame.shape, kw.get("shrink_factor", 0.9))
            if skin is not None:
                base = base & skin
            for pm in split_mask(base, r, c):
                if pm is None:
                    vals.append(np.zeros(3)); oks.append(False)
                    continue
                m_rgb, ok, _n, _cause = roi_mean_rgb(frame, pm, kw.get("min_valid_fraction", 0.6),
                                                     kw.get("trim_fraction", 0.10))
                vals.append(m_rgb if (ok and m_rgb is not None) else np.zeros(3))
                oks.append(bool(ok and m_rgb is not None))
        holder["rgb"], holder["valid"] = np.array(vals), np.array(oks)
        return res

    pipeline_mod.extract_rois = recording_extract
    try:
        with RPPGPipeline(cfg, debug=False) as pipe:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                holder.clear()
                pipe.process_frame(frame, int(idx / fps * 1000))
                if "rgb" in holder:
                    rows_rgb.append(holder["rgb"]); rows_valid.append(holder["valid"]); detected.append(True)
                else:
                    rows_rgb.append(np.zeros((len(names), 3))); rows_valid.append(np.zeros(len(names), bool))
                    detected.append(False)
                idx += 1
    finally:
        pipeline_mod.extract_rois = original_extract
        cap.release()
    return {
        "record_id": record_id, "fps": float(fps), "names": names,
        "timestamps_ms": [int(i / fps * 1000) for i in range(len(rows_rgb))],
        "rgb": np.array(rows_rgb), "valid": np.array(rows_valid), "detected": np.array(detected),
    }


def main() -> None:
    import degenerate as dg
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for record_id, subject_dir in dg.discover_records(Path("~/Downloads").expanduser()):
        path = OUT_DIR / f"{record_id}.pkl"
        if path.exists():
            continue
        t0 = time.time()
        data = build_record(record_id, subject_dir)
        path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        share_valid = float(data["valid"].mean())
        print(f"  {record_id:16s} кадров {len(data['rgb']):5d}, участков {len(data['names'])}, "
              f"валидных {share_valid:.1%}, {time.time() - t0:5.1f}с", flush=True)
    print("PATCHES_DONE", flush=True)


if __name__ == "__main__":
    main()
