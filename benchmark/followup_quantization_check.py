"""
Задача D: искажает ли квантование уже ДОСТАВЛЕННЫЕ кадры условия B1f.

Кадры деградаций на диск не писались (они строились на лету), поэтому
«доставленные кадры» воспроизводятся тем же кодом с тем же seed:
FrameDegrader детерминирован (seed = crc32(record_id|condition), см.
rppg.degrade.condition_seed). Это не допущение: скрипт ПРОВЕРЯЕТ побитовое
совпадение — эталонный float-кадр, проведённый через то же самое клипование
и приведение к uint8, обязан совпасть с кадром, который выдаёт сам
FrameDegrader. Если хоть один кадр разошёлся, скрипт падает.

Два ряда считаются по ОДНОЙ И ТОЙ ЖЕ маске (фиксированная маска лица из
prep), поэтому единственная разница между ними — квантование:
  signal_float_expected        — среднее по маске от float-кадра, без uint8;
  signal_from_delivered_frames — среднее по маске от кадра, как его получил
                                  пайплайн (uint8).

    PYTHONPATH=src venv/bin/python3 benchmark/followup_quantization_check.py --record 10
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

from rppg.degrade import (  # noqa: E402
    FLICKER_SPECS, FrameDegrader, condition_seed, prepare_record,
)
import degenerate as dg  # noqa: E402
import eval_ubfc_subject as ev  # noqa: E402

CONDITION = "B1f"


def fourier_at(series: np.ndarray, fps: float, f_hz: float) -> tuple[float, float]:
    """Амплитуда (в долях от постоянной составляющей) и фаза (рад) на f_hz."""
    t = np.arange(len(series)) / fps
    x = series - series.mean()
    c = 2.0 * np.mean(x * np.cos(2 * np.pi * f_hz * t))
    s = 2.0 * np.mean(x * np.sin(2 * np.pi * f_hz * t))
    return float(np.hypot(c, s) / series.mean()), float(np.arctan2(s, c))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default="10")
    ap.add_argument("--data-dir", default="~/Downloads")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate" / "followup"))
    ap.add_argument("--start-frame", type=int, default=300, help="начало окна (кадры холодного старта пропускаются)")
    ap.add_argument("--n-frames", type=int, default=300, help="длина окна в кадрах (10 с при ~30 fps)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = dict(dg.discover_records(Path(args.data_dir).expanduser()))
    video = str(ev.find_video(by_id[args.record]))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    injected_hz = FLICKER_SPECS[CONDITION][0]
    amp = FLICKER_SPECS[CONDITION][1]
    prep = prepare_record(video, CONDITION, float(fps))
    mask = prep.face_mask[:, :, 0] > 0.5   # та же зона для обоих рядов
    soft = prep.face_mask

    delivered = FrameDegrader(CONDITION, args.record, float(fps), prep, injected_hz=injected_hz)
    # Копия арифметики FrameDegrader БЕЗ квантования. Порядок операций и
    # порядок обращений к генератору повторяются один в один, иначе шум
    # разойдётся и сравнение потеряет смысл.
    rng_float = np.random.default_rng(condition_seed(args.record, CONDITION))
    base_median = prep.median_frame.astype(np.float32)

    rows = []
    for idx in range(args.start_frame + args.n_frames):
        noise = rng_float.normal(0.0, prep.noise_sigma, base_median.shape).astype(np.float32)
        wave = amp * np.sin(2.0 * np.pi * injected_hz * (idx / float(fps)))
        float_frame = (base_median + noise) * (1.0 + wave * soft)
        uint8_frame = delivered(None, idx)
        if idx < args.start_frame:
            continue
        requantized = np.clip(float_frame, 0.0, 255.0).astype(np.uint8)
        if not np.array_equal(requantized, uint8_frame):
            raise SystemExit(
                f"кадр {idx}: float-эталон после квантования НЕ совпал с кадром деградации — "
                "копия арифметики разошлась с rppg.degrade, сравнение недействительно"
            )
        rows.append({
            "record_id": args.record,
            "condition": CONDITION,
            "frame_idx": idx,
            "t_s": f"{idx / fps:.6f}",
            "signal_float_expected": f"{float(float_frame[:, :, 1][mask].mean()):.10g}",
            "signal_from_delivered_frames": f"{float(uint8_frame[:, :, 1][mask].mean()):.10g}",
        })

    path = out_dir / "quantization_check.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    exp = np.array([float(r["signal_float_expected"]) for r in rows])
    got = np.array([float(r["signal_from_delivered_frames"]) for r in rows])
    a_exp, ph_exp = fourier_at(exp, fps, injected_hz)
    a_got, ph_got = fourier_at(got, fps, injected_hz)
    print(f"запись {args.record}, условие {CONDITION}, наложено {injected_hz * 60:.1f} BPM "
          f"({injected_hz} Гц), амплитуда {amp * 100:.1f}% по маске лица")
    print(f"кадров в окне: {len(rows)} (с {args.start_frame} по {args.start_frame + args.n_frames - 1})")
    print(f"побитовая сверка float->uint8 с кадрами деградации: совпало на всех {len(rows)} кадрах")
    print(f"\n{'ряд':>32s}{'среднее':>12s}{'амплитуда на f':>17s}{'фаза, рад':>12s}")
    print(f"{'signal_float_expected':>32s}{exp.mean():>12.4f}{a_exp * 100:>16.4f}%{ph_exp:>12.4f}")
    print(f"{'signal_from_delivered_frames':>32s}{got.mean():>12.4f}{a_got * 100:>16.4f}%{ph_got:>12.4f}")
    print(f"\nотношение амплитуд (доставленное/эталон): {a_got / a_exp:.4f}")
    print(f"сдвиг фазы: {ph_got - ph_exp:+.5f} рад")
    print(f"СКО поканальной разности рядов: {np.std(got - exp):.5f} ур. яркости")


if __name__ == "__main__":
    main()
