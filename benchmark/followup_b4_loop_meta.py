"""
Задание 3: метаданные петли условия B4 и эталонный пульс зацикленного
фрагмента.

Ничего не пересчитывает: видео только СКАНИРУЕТСЯ (cap.grab без
декодирования в массив) ради числа кадров и временны́х меток контейнера,
сам B4 не собирается и пайплайн не запускается.

Детектор пиков пайплайна (hrv/features.py) здесь НЕ используется — по
условию задания: у него известное систематическое завышение числа пиков.
Эталон фрагмента считается только scipy.signal.find_peaks с параметрами
из задания.

    PYTHONPATH=src venv/bin/python3 benchmark/followup_b4_loop_meta.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from rppg.config import PipelineConfig  # noqa: E402
from rppg.degrade import LOOP_SECONDS  # noqa: E402

import degenerate as dg  # noqa: E402
import eval_ubfc_subject as ev  # noqa: E402

MAX_HR_BPM = 180.0                      # distance в find_peaks: 60/180 = 0.33 с
MIN_PEAK_DISTANCE_S = 60.0 / MAX_HR_BPM
PROMINENCE_FACTOR = 0.3


def scan_video(video_path: str) -> tuple[int, np.ndarray]:
    """Число кадров и временны́е метки КОНТЕЙНЕРА, по одной на кадр.

    POS_MSEC читается ПОСЛЕ успешного grab: проверено на реальном файле,
    что именно так значение относится к только что захваченному кадру
    (перед первым grab позиция ещё нулевая и кадру не соответствует).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео: {video_path}")
    ts = []
    while cap.grab():
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return len(ts), np.array(ts, dtype=float) / 1000.0


def supplied_timestamps_ms(n_frames: int, fps_container: float) -> np.ndarray:
    """Метки, которые офлайн-прогон РЕАЛЬНО подаёт в пайплайн.

    Копия выражения из benchmark/degenerate.py (строка 273):
        pipeline.process_frame(degraded, int(frame_idx / fps * 1000))
    Целочисленное усечение здесь существенно: шаг меток гуляет между
    соседними целыми миллисекундами, и именно из их медианы пайплайн
    выводит частоту дискретизации для Уэлча.
    """
    return np.array([int(i / fps_container * 1000) for i in range(n_frames)], dtype=float)


def fs_from_timestamps(ts_ms: np.ndarray) -> float:
    """Ровно то, что делает RPPGPipeline._estimate_fps (строки 776-784)."""
    dt = np.diff(ts_ms) / 1000.0
    dt = dt[dt > 1e-3]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def loop_reference_hr(t_ref: np.ndarray, wave: np.ndarray, t0: float, t1: float):
    """Пульс контактной волны на интервале петли.

    Приведение к равномерной сетке — той же процедурой, что в основной
    работе (scripts/eval_ubfc_subject.py::reference_bpm_from_wave):
    сохраняются границы интервала и число отсчётов, интерполяция линейная.
    Нужна потому, что во втором подмножестве UBFC шаг временно́й оси
    прибора гуляет от 0.010 до 0.223 с.
    """
    mask = (t_ref >= t0) & (t_ref <= t1)
    t_win, x_raw = t_ref[mask], np.asarray(wave, dtype=float)[mask]
    if len(t_win) < 4:
        return None
    span = float(t_win[-1] - t_win[0])
    if span <= 0:
        return None
    fs = float(len(t_win) - 1) / span
    t_uniform = np.linspace(t_win[0], t_win[-1], len(t_win))
    x = np.interp(t_uniform, t_win, x_raw)

    distance = max(int(round(MIN_PEAK_DISTANCE_S * fs)), 1)
    prominence = PROMINENCE_FACTOR * float(np.std(x))
    peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    peak_times = t_uniform[peaks]
    if len(peak_times) < 2:
        return {"fs_wave": fs, "peaks": peak_times, "mean_ibi": None, "bpm": None}
    ibis = np.diff(peak_times)
    mean_ibi = float(np.mean(ibis))
    return {"fs_wave": fs, "peaks": peak_times, "mean_ibi": mean_ibi, "bpm": 60.0 / mean_ibi}


def g0_window0_reference(windows_csv: Path) -> dict[str, str]:
    out = {}
    with windows_csv.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] == "G0" and r["window_idx"] == "0":
                out[r["record_id"]] = r["bpm_reference"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="~/Downloads")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate" / "followup"))
    ap.add_argument("--windows", default=str(REPO_ROOT / "results" / "degenerate" / "windows.csv"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    g0_ref = g0_window0_reference(Path(args.windows).expanduser())
    cfg = PipelineConfig()
    window_frames_hint = cfg.window.window_seconds

    meta_rows, peak_rows = [], []
    for record_id, subject_dir in dg.discover_records(Path(args.data_dir).expanduser()):
        video = str(ev.find_video(subject_dir))
        gt_path = ev.find_ground_truth(subject_dir)
        subset = 1 if gt_path.name == "gtdump.xmp" else 2
        t_ref, wave, _hr_device = ev.load_reference_waveform(gt_path)

        cap = cv2.VideoCapture(video)
        fps_container = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        n_frames, t_video = scan_video(video)
        fps_effective = (n_frames - 1) / (t_video[-1] - t_video[0])

        # Правило из rppg/degrade.py (строка 167): n_loop = int(round(LOOP_SECONDS * fps))
        loop_n_frames = int(round(LOOP_SECONDS * fps_container))

        ts_supplied = supplied_timestamps_ms(n_frames, fps_container)
        loop_period_s = float(ts_supplied[loop_n_frames] - ts_supplied[0]) / 1000.0
        fs_used = fs_from_timestamps(ts_supplied)
        # Сверка: пайплайн считает fs по окну (~10 с), а не по всей записи.
        n_win = int(round(window_frames_hint * fps_container))
        fs_first_window = fs_from_timestamps(ts_supplied[:n_win])

        # Интервал петли — по меткам ВИДЕО (реальное время записи, к которому
        # привязана контактная волна), от первого кадра петли до последнего
        # включительно.
        t0, t1 = float(t_video[0]), float(t_video[loop_n_frames - 1])
        ref = loop_reference_hr(t_ref, wave, t0, t1)

        meta_rows.append({
            "record_id": record_id,
            "subset": subset,
            "fps_container": f"{fps_container:.6f}",
            "fps_effective": f"{fps_effective:.6f}",
            "n_frames_total": n_frames,
            "loop_n_frames": loop_n_frames,
            "loop_period_s": f"{loop_period_s:.6f}",
            "fs_used_hz": f"{fs_used:.6f}",
            "time_base_used": "timestamps",
            "ref_n_peaks": len(ref["peaks"]) if ref else 0,
            "ref_mean_ibi_s": f"{ref['mean_ibi']:.6f}" if ref and ref["mean_ibi"] else "",
            "ref_hr_loop_bpm": f"{ref['bpm']:.6f}" if ref and ref["bpm"] else "",
            "ref_hr_g0_window0_bpm": g0_ref.get(record_id, ""),
        })
        if ref:
            for k, t_peak in enumerate(ref["peaks"]):
                peak_rows.append({"record_id": record_id, "peak_idx": k, "t_s": f"{float(t_peak):.6f}"})
        flag = "" if fs_first_window == fs_used else f"  (fs первого окна {fs_first_window:.4f})"
        print(f"  {record_id:16s} subset {subset}  fps {fps_container:8.4f}  кадров {n_frames:5d}  "
              f"петля {loop_n_frames:3d} кадров / {loop_period_s:.4f}с  fs {fs_used:7.4f}  "
              f"пиков {len(ref['peaks']) if ref else 0}{flag}", flush=True)

    def write(path: Path, rows):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(out_dir / "b4_loop_meta.csv", meta_rows)
    write(out_dir / "b4_ref_peaks.csv", peak_rows)
    print(f"\nвыгружено: {len(meta_rows)} строк в b4_loop_meta.csv, {len(peak_rows)} в b4_ref_peaks.csv")


if __name__ == "__main__":
    main()
