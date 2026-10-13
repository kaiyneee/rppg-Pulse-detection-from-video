"""
Задача A: разбор 125 опубликованных окон условия A1 по записям.

Считается ПО УЖЕ СОБРАННОМУ results/degenerate/windows.csv — прогон не
повторяется. Строки выгружаются по всем 50 записям, включая те, где не
опубликовано ничего: без нулевых строк не видно, размазано явление по
выборке или сидит в паре записей.

    PYTHONPATH=src venv/bin/python3 benchmark/followup_a1_breakdown.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REST_BAND = (50.0, 100.0)      # типичный покой
PLAUSIBLE_BAND = (100.0, 180.0)


def _natural_key(record_id: str):
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", record_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default=str(REPO_ROOT / "results" / "degenerate" / "windows.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate" / "followup"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(Path(args.windows).expanduser().open(encoding="utf-8"))
            if r["condition"] == "A1"]

    by_record: dict[str, list[dict]] = {}
    for r in rows:
        by_record.setdefault(r["record_id"], []).append(r)

    breakdown, published_rows = [], []
    for rid in sorted(by_record, key=_natural_key):
        rs = by_record[rid]
        pub = [r for r in rs if r["published_gated"] == "1"]
        bpms = np.array([float(r["bpm_estimate"]) for r in pub if r["bpm_estimate"] not in ("", "nan")])
        breakdown.append({
            "record_id": rid,
            "n_windows_total": len(rs),
            "n_windows_published": len(pub),
            "publish_rate": f"{len(pub) / len(rs):.6g}",
            "bpm_min": f"{bpms.min():.6g}" if bpms.size else "",
            "bpm_median": f"{np.median(bpms):.6g}" if bpms.size else "",
            "bpm_max": f"{bpms.max():.6g}" if bpms.size else "",
            "n_bpm_in_50_100": int(((bpms >= REST_BAND[0]) & (bpms < REST_BAND[1])).sum()) if bpms.size else 0,
            "n_bpm_in_100_180": int(((bpms >= PLAUSIBLE_BAND[0]) & (bpms < PLAUSIBLE_BAND[1])).sum()) if bpms.size else 0,
            "n_bpm_outside_50_180": int(((bpms < REST_BAND[0]) | (bpms >= PLAUSIBLE_BAND[1])).sum()) if bpms.size else 0,
        })
        for r in sorted(pub, key=lambda x: int(x["window_idx"])):
            published_rows.append({
                "record_id": rid,
                "window_idx": r["window_idx"],
                "t_start_s": r["t_start_s"],
                "bpm_estimate": r["bpm_estimate"],
                "sqi_composite": r["sqi_composite"],
                "sqi_snr": r["sqi_snr"],
                "snr_db": r["snr_db"],
            })

    def write(path: Path, fields, data):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write(out_dir / "a1_published_breakdown.csv", list(breakdown[0].keys()), breakdown)
    write(out_dir / "a1_published_windows.csv", list(published_rows[0].keys()), published_rows)

    total_pub = sum(b["n_windows_published"] for b in breakdown)
    nonzero = [b for b in breakdown if b["n_windows_published"] > 0]
    all_bpm = np.array([float(r["bpm_estimate"]) for r in published_rows])
    in_rest = int(((all_bpm >= REST_BAND[0]) & (all_bpm < REST_BAND[1])).sum())
    print(f"записей: {len(breakdown)}, опубликовано окон суммарно: {total_pub}")
    print(f"записей с ненулевой публикацией: {len(nonzero)} из {len(breakdown)}")
    top = sorted(nonzero, key=lambda b: -b["n_windows_published"])[:6]
    print("  больше всего: " + ", ".join(f"{b['record_id']} ({b['n_windows_published']})" for b in top))
    share_top2 = sum(b["n_windows_published"] for b in top[:2]) / max(total_pub, 1)
    print(f"  доля двух самых «урожайных» записей: {share_top2:.1%}")
    print(f"BPM опубликованных окон: {all_bpm.min():.1f}..{all_bpm.max():.1f}, медиана {np.median(all_bpm):.1f}")
    print(f"  в диапазоне покоя 50-100: {in_rest} из {len(all_bpm)} ({in_rest / len(all_bpm):.1%})")
    print(f"  вне 50-180: {int(((all_bpm < 50) | (all_bpm >= 180)).sum())}")


if __name__ == "__main__":
    main()
