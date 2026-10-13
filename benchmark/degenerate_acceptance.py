"""
Проверка пяти критериев приёмки прогона по вырожденным входам
(см. раздел 8 задания). Ничего не считает заново — только сверяет уже
выгруженные results/degenerate/*.csv с опубликованными числами основной
работы и с внутренними требованиями согласованности.

    PYTHONPATH=src venv/bin/python3 benchmark/degenerate_acceptance.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLISHED_N_WINDOWS = 2923
PUBLISHED_COVERAGE = 0.989
PUBLISHED_MAE = 1.67
TOLERANCE = 0.01  # 1% по MAE и покрытию; по числу окон — точное совпадение

WINDOW_FIELDS_REQUIRED = [
    "record_id", "condition", "window_idx", "t_start_s",
    "sqi_composite", "sqi_snr", "snr_db", "sqi_roi_agreement",
    "sqi_landmark_stability", "sqi_temporal_consistency",
    "harmonic_plausibility", "flicker_flag", "flicker_score",
    "hard_snr_gate_pass", "published_gated", "published_weighted_only",
    "bpm_estimate", "bpm_reference", "bpm_injected", "method", "freq_method",
]


def _f(x):
    if x in ("", "nan"):
        return float("nan")
    return float(x)


def check(out_dir: Path) -> bool:
    rows = list(csv.DictReader((out_dir / "windows.csv").open(encoding="utf-8")))
    ok_all = True

    def verdict(n: int, title: str, passed: bool, detail: str) -> None:
        nonlocal ok_all
        ok_all &= passed
        print(f"{n}. {title}: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'}")
        print(f"   {detail}")

    # --- 1. Регрессия на контроле G0 ---
    g0 = [r for r in rows if r["condition"] == "G0"]
    per_record = defaultdict(list)
    for r in g0:
        per_record[r["record_id"]].append(r)
    maes, covs = [], []
    for rs in per_record.values():
        gated = [r for r in rs if r["published_gated"] == "1"]
        errs = [
            abs(_f(r["bpm_estimate"]) - _f(r["bpm_reference"])) for r in gated
            if r["bpm_reference"] not in ("", "nan")
        ]
        if errs:
            maes.append(float(np.mean(errs)))
        covs.append(len(gated) / len(rs))
    mae, cov = float(np.mean(maes)), float(np.mean(covs))
    passed = (
        len(g0) == PUBLISHED_N_WINDOWS
        and abs(mae - PUBLISHED_MAE) / PUBLISHED_MAE <= TOLERANCE
        and abs(cov - PUBLISHED_COVERAGE) / PUBLISHED_COVERAGE <= TOLERANCE
    )
    verdict(1, "регрессия на контроле G0", passed,
            f"окон {len(g0)} (нужно {PUBLISHED_N_WINDOWS}), MAE {mae:.3f} (нужно {PUBLISHED_MAE}), "
            f"покрытие {cov:.4f} (нужно {PUBLISHED_COVERAGE})")

    # --- 2. A1: publish_rate_gated == 0 ---
    a1 = [r for r in rows if r["condition"] == "A1"]
    a1_pub = [r for r in a1 if r["published_gated"] == "1"]
    a1_detail = f"опубликовано {len(a1_pub)} окон из {len(a1)}"
    if a1_pub:
        where = sorted({f"{r['record_id']}#{r['window_idx']}" for r in a1_pub})
        a1_detail += "; окна: " + ", ".join(where[:10]) + (" ..." if len(where) > 10 else "")
    verdict(2, "A1: доля публикаций равна нулю", len(a1_pub) == 0, a1_detail)

    # --- 3. Число окон на запись одинаково между условиями ---
    counts = defaultdict(dict)
    for r in rows:
        counts[r["record_id"]][r["condition"]] = counts[r["record_id"]].get(r["condition"], 0) + 1
    mismatched = {rec: c for rec, c in counts.items() if len(set(c.values())) > 1}
    verdict(3, "число окон на запись совпадает между условиями", not mismatched,
            "все записи согласованы" if not mismatched
            else f"расходятся {len(mismatched)}: " + "; ".join(f"{k}:{v}" for k, v in list(mismatched.items())[:5]))

    # --- 4. Колонки на месте и не пусты целиком ---
    header = list(rows[0].keys()) if rows else []
    missing = [c for c in WINDOW_FIELDS_REQUIRED if c not in header]
    empty = [c for c in WINDOW_FIELDS_REQUIRED if c in header and all(r[c] == "" for r in rows)]
    verdict(4, "все колонки присутствуют и заполнены", not missing and not empty,
            f"отсутствуют: {missing or 'нет'}; полностью пустые: {empty or 'нет'}")

    # --- 5. Воспроизводимость (проверяется отдельным повторным прогоном) ---
    repro = out_dir / "reproducibility_check.txt"
    if repro.exists():
        text = repro.read_text(encoding="utf-8").strip()
        verdict(5, "повторный прогон даёт те же байты", text.startswith("СОВПАДАЕТ"), text.splitlines()[0])
    else:
        verdict(5, "повторный прогон даёт те же байты", False,
                "не выполнялся: нет results/degenerate/reproducibility_check.txt")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate"))
    args = ap.parse_args()
    ok = check(Path(args.out_dir).expanduser())
    print("\nИТОГ:", "все критерии пройдены" if ok else "ЕСТЬ НЕПРОЙДЕННЫЕ КРИТЕРИИ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
