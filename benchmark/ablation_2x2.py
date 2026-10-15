"""
Абляция 2x2 против основного эталона (results/reference/ubfc_ref_full10s.csv):
спектр (Уэлч 8 с / периодограмма полного окна) x объединение ROI (один ROI
с наибольшим SNR / среднее нормированных спектров трёх ROI).

Каждый вариант — сквозное воспроизведение пайплайна из кэша сигналов;
окна и флаги публикации — собственные у каждого варианта.
MAE — по опубликованным окнам внутри записи, затем среднее по записям.
Грубый промах — опубликованное окно с |оценка - эталон| > 5 уд/мин;
доля считается от всех опубликованных окон части (пул по окнам).

    PYTHONPATH=src venv/bin/python3 benchmark/ablation_2x2.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import experiments as ex  # noqa: E402
import reference_full10s as R  # noqa: E402
import trace_cache as tc  # noqa: E402
from rppg.config import FrequencyMethod  # noqa: E402

OUTLIER_BPM = 5.0
VARIANTS = [
    ("старая (Уэлч 8 с, один ROI)", "welch", "best_snr"),
    ("только полное окно", "fft", "best_snr"),
    ("только усреднение ROI", "welch", "spectral_mean"),
    ("обе (новое умолчание)", "fft", "spectral_mean"),
]


def run_variant(records, freq: str, comb: str) -> dict[str, dict]:
    ref = R.load_reference()
    out = {}
    for rid in records:
        trace = tc.load_trace(rid)
        cfg = tc.make_config(trace.fps, "legacy_welch8")
        cfg.frequency_method = FrequencyMethod(freq)
        cfg.roi.spectral_combination = comb
        rows, _ = tc.replay(trace, cfg)
        errs, n_pub = [], 0
        for r in rows:
            if not r.publishable:
                continue
            n_pub += 1
            rv = ref.get((rid, r.timestamp_ms))
            if rv is not None and np.isfinite(r.bpm):
                errs.append(r.bpm - rv)
        e = np.array(errs)
        out[rid] = {"mae": float(np.mean(np.abs(e))) if e.size else float("nan"),
                    "n_scored": int(e.size), "n_out": int((np.abs(e) > OUTLIER_BPM).sum()),
                    "coverage": n_pub / max(len(rows), 1)}
    return out


def aggregate(res: dict, ids: list[str]) -> dict:
    mae = np.array([res[r]["mae"] for r in ids if np.isfinite(res[r]["mae"])])
    n = sum(res[r]["n_scored"] for r in ids)
    return {"mae": float(mae.mean()), "mae_sd": float(mae.std(ddof=1)),
            "outlier": sum(res[r]["n_out"] for r in ids) / max(n, 1),
            "coverage": float(np.mean([res[r]["coverage"] for r in ids])), "n_windows": n}


def main() -> None:
    dev, test = ex.split()
    table = []
    print(f"{'вариант':32s}{'часть':>7s}{'MAE':>8s}{'± SD':>7s}{'промахи>5':>11s}{'окон':>7s}{'покрытие':>10s}")
    for label, freq, comb in VARIANTS:
        res = run_variant(dev + test, freq, comb)
        for part, ids in (("dev", dev), ("test", test)):
            a = aggregate(res, ids)
            table.append({"variant": label, "frequency_method": freq, "spectral_combination": comb,
                          "part": part, "n_records": len(ids), "mae": f"{a['mae']:.6f}",
                          "mae_sd": f"{a['mae_sd']:.4f}", "outlier_rate": f"{a['outlier']:.6f}",
                          "n_windows_scored": a["n_windows"], "coverage": f"{a['coverage']:.6f}"})
            print(f"{label:32s}{part:>7s}{a['mae']:>8.3f}{a['mae_sd']:>7.2f}{a['outlier']:>10.2%}"
                  f"{a['n_windows']:>7d}{a['coverage']:>9.1%}", flush=True)
    out = REPO_ROOT / "results" / "accuracy" / "ablation_2x2.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"\nвыгружено: {out}")


if __name__ == "__main__":
    main()
