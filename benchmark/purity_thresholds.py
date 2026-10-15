"""
Порог жёсткой проверки чистоты спектра (QualityConfig.min_spectral_purity):
доля опубликованных окон, MAE и доля грубых промахов для набора АБСОЛЮТНЫХ
порогов, на разработочной и тестовой частях UBFC.

Для каждого порога пайплайн (умолчания, пресет fft_spectral_mean)
воспроизводится заново из кэша с включённой проверкой: решение о
публикации влияет на доверенную историю пульса и через неё на следующие
окна, поэтому фильтровать готовые окна по порогу было бы неточно.

Мерило — основной эталон (results/reference/ubfc_ref_full10s.csv).
Покрытие — доля опубликованных окон от всех выданных (пул по окнам части).
MAE — по опубликованным окнам внутри записи, затем среднее по записям,
у которых опубликовано хотя бы одно окно. Грубый промах — опубликованное
окно с |оценка - эталон| > 5 уд/мин; доля — от опубликованных окон части.

    PYTHONPATH=src venv/bin/python3 benchmark/purity_thresholds.py
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

THRESHOLDS = [None, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40]
OUTLIER_BPM = 5.0


def run(records, threshold):
    ref = R.load_reference()
    out = {}
    for rid in records:
        trace = tc.load_trace(rid)
        cfg = tc.make_config(trace.fps)
        cfg.quality.min_spectral_purity = threshold
        rows, _ = tc.replay(trace, cfg)
        errs = []
        n_pub = 0
        for r in rows:
            if not r.publishable:
                continue
            n_pub += 1
            rv = ref.get((rid, r.timestamp_ms))
            if rv is not None and np.isfinite(r.bpm):
                errs.append(r.bpm - rv)
        out[rid] = {"n": len(rows), "n_pub": n_pub, "errs": np.array(errs)}
    return out


def summarize(res, ids):
    n_all = sum(res[r]["n"] for r in ids)
    n_pub = sum(res[r]["n_pub"] for r in ids)
    maes = [float(np.mean(np.abs(res[r]["errs"]))) for r in ids if res[r]["errs"].size]
    all_e = np.concatenate([res[r]["errs"] for r in ids if res[r]["errs"].size]) if maes else np.array([])
    return {
        "coverage": n_pub / max(n_all, 1),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "outlier": float(np.mean(np.abs(all_e) > OUTLIER_BPM)) if all_e.size else float("nan"),
        "records_with_output": len(maes),
        "n_published": n_pub,
    }


def main() -> None:
    dev, test = ex.split()
    table = []
    print(f"{'порог':>7s} | {'dev: покрытие':>14s}{'MAE':>7s}{'промахи':>9s}{'записей':>9s} | "
          f"{'test: покрытие':>15s}{'MAE':>7s}{'промахи':>9s}{'записей':>9s}")
    for thr in THRESHOLDS:
        res = run(dev + test, thr)
        d, t = summarize(res, dev), summarize(res, test)
        label = "выкл." if thr is None else f"{thr:.2f}"
        print(f"{label:>7s} | {d['coverage']:>13.1%}{d['mae']:>7.3f}{d['outlier']:>8.2%}{d['records_with_output']:>6d}/20 | "
              f"{t['coverage']:>14.1%}{t['mae']:>7.3f}{t['outlier']:>8.2%}{t['records_with_output']:>6d}/30", flush=True)
        for part, a in (("dev", d), ("test", t)):
            table.append({"threshold": "" if thr is None else thr, "part": part,
                          "coverage": f"{a['coverage']:.6f}", "mae": f"{a['mae']:.6f}",
                          "outlier_rate": f"{a['outlier']:.6f}", "n_published": a["n_published"],
                          "records_with_output": a["records_with_output"]})
    out = REPO_ROOT / "results" / "accuracy" / "purity_thresholds.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"\nвыгружено: {out}")


if __name__ == "__main__":
    main()
