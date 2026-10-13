"""
Четыре анализа по пооконным логам прогона на UBFC (задача 53, повтор на
двухпроходных логах с эталоном из контактной волны):

  1. Кривая "покрытие — MAE" по порогу sqi_score.
  2. Корреляция каждой компоненты индекса качества с |ошибкой|.
  3. Бланд-Альтман: смещение, SD разностей, 95% пределы согласия.
  4. MAE по 10-секундным интервалам от начала записи и по диапазонам
     эталонной частоты.

Плюс сводка по качеству эталонной волны (ref_wave_snr_db) — она заменила
прежний фильтр правдоподобия значений ЧСС.

Окна пулятся по всем записям: вопрос всюду — "предсказывает ли величина
ошибку ВНУТРИ записи", а не сравнение испытуемых между собой (для второго
нужно усреднение по испытуемому, см. report_ubfc_split.py).

Использование:
    PYTHONPATH=src python3 scripts/analyze_ubfc_windows.py [--dir ubfc_windows_twopass]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

SQI_COMPONENTS = [
    "sqi_score", "spectral_snr_db", "harmonic_score",
    "cross_roi_agreement", "landmark_stability", "temporal_consistency",
]


def to_float(x):
    if x in (None, "", "None"):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def load(dir_path: Path) -> list[dict]:
    rows = []
    for f in sorted(dir_path.glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["subject"] = f.stem
                rows.append(row)
    return rows


def coverage_mae_curve(sqi: np.ndarray, err: np.ndarray) -> None:
    print("\n=== 1. Кривая «порог sqi_score -> покрытие, MAE» ===")
    n_total = len(sqi)
    print(f"{'порог':>8s}{'покрытие':>11s}{'окон':>7s}{'MAE':>8s}")
    thr_list, mae_list = [], []
    for thr in np.round(np.arange(0.0, 1.01, 0.05), 2):
        m = sqi >= thr
        n = int(m.sum())
        if n == 0:
            print(f"{thr:>8.2f}{0.0:>10.1f}%{n:>7d}{'—':>8s}")
            continue
        mae = float(np.mean(np.abs(err[m])))
        print(f"{thr:>8.2f}{n / n_total * 100:>10.1f}%{n:>7d}{mae:>8.2f}")
        if n >= 5:
            thr_list.append(thr); mae_list.append(mae)
    if len(thr_list) >= 3:
        r = float(np.corrcoef(thr_list, mae_list)[0, 1])
        drop = mae_list[0] - mae_list[-1]
        verdict = ("ИНФОРМАТИВЕН" if r < -0.5 and drop > 0.5
                   else "СЛАБО ИНФОРМАТИВЕН" if r < -0.2 else "НЕ ИНФОРМАТИВЕН")
        print(f"  корреляция порога с MAE: {r:.3f}, падение MAE {drop:+.2f} -> SQI {verdict}")


def component_correlations(rows: list[dict], err: np.ndarray) -> None:
    print("\n=== 2. Корреляция компонент индекса качества с |ошибкой| ===")
    print(f"{'компонента':<22s}{'r':>8s}{'n':>7s}")
    for comp in SQI_COMPONENTS:
        v = np.array([to_float(r.get(comp)) for r in rows])
        m = ~(np.isnan(v) | np.isnan(err))
        n = int(m.sum())
        if n < 3 or np.std(v[m]) < 1e-9:
            print(f"{comp:<22s}{'—':>8s}{n:>7d}   (константа/нет данных)")
            continue
        print(f"{comp:<22s}{float(np.corrcoef(v[m], np.abs(err[m]))[0, 1]):>8.3f}{n:>7d}")


def bland_altman(pred: np.ndarray, ref: np.ndarray) -> None:
    print("\n=== 3. Бланд-Альтман (предсказание минус эталон) ===")
    d = pred - ref
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    print(f"  смещение: {bias:+.2f} BPM")
    print(f"  SD разностей: {sd:.2f} BPM")
    print(f"  95% пределы согласия: [{lo:+.2f}, {hi:+.2f}] BPM")
    print(f"  доля окон вне пределов: {float(np.mean((d < lo) | (d > hi))) * 100:.1f}% (ожидается ~5%)")


def mae_by_time_and_hr(rows: list[dict], err: np.ndarray, ref: np.ndarray) -> None:
    print("\n=== 4a. MAE по 10-секундным интервалам от начала записи ===")
    t = np.array([to_float(r["second"]) for r in rows])
    print(f"{'интервал':>14s}{'окон':>7s}{'MAE':>8s}")
    for start in range(0, int(np.nanmax(t)) + 1, 10):
        m = (t >= start) & (t < start + 10)
        if m.sum() == 0:
            continue
        print(f"{f'{start}-{start + 10} с':>14s}{int(m.sum()):>7d}{float(np.mean(np.abs(err[m]))):>8.2f}")

    print("\n=== 4б. MAE по диапазонам эталонной частоты ===")
    edges = [40, 60, 70, 80, 90, 100, 110, 120, 200]
    print(f"{'диапазон':>14s}{'окон':>7s}{'MAE':>8s}")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ref >= a) & (ref < b)
        if m.sum() == 0:
            continue
        print(f"{f'{a}-{b} BPM':>14s}{int(m.sum()):>7d}{float(np.mean(np.abs(err[m]))):>8.2f}")


def reference_wave_quality(rows: list[dict]) -> None:
    """Задача 53, п.3: вместо прежнего фильтра правдоподобия значений ЧСС —
    доля окон, где НЕПРИГОДНА САМА ВОЛНА (низкий SNR). Это число и должно
    приводиться в статье вместо формулировки про фильтр."""
    snr = np.array([to_float(r.get("ref_wave_snr_db")) for r in rows])
    ok = ~np.isnan(snr)
    if not ok.any():
        return
    print("\n=== Качество эталонной волны (замена прежнего фильтра правдоподобия) ===")
    print(f"  медиана SNR волны: {float(np.median(snr[ok])):.2f} dB, "
          f"перцентили 5/25/75/95: "
          + ", ".join(f"{float(np.percentile(snr[ok], p)):.2f}" for p in (5, 25, 75, 95)))
    for thr in (-3.0, 0.0, 2.0, 3.0):
        print(f"  окон с SNR волны < {thr:+.1f} dB: {int((snr[ok] < thr).sum())} "
              f"({float(np.mean(snr[ok] < thr)) * 100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Анализы по пооконным логам UBFC")
    ap.add_argument("--dir", default="ubfc_windows_twopass")
    args = ap.parse_args()

    dir_path = REPO_ROOT / args.dir
    if not dir_path.exists():
        print(f"ОШИБКА: {dir_path} не найден", file=sys.stderr)
        sys.exit(1)

    rows = load(dir_path)
    ref_key = "bpm_ref_wave" if rows and "bpm_ref_wave" in rows[0] else "bpm_ref"
    rows = [r for r in rows
            if to_float(r.get("bpm_pred")) == to_float(r.get("bpm_pred"))
            and to_float(r.get(ref_key)) == to_float(r.get(ref_key))]
    if not rows:
        print("ОШИБКА: нет окон с парой (предсказание, эталон)", file=sys.stderr)
        sys.exit(1)

    pred = np.array([to_float(r["bpm_pred"]) for r in rows])
    ref = np.array([to_float(r[ref_key]) for r in rows])
    err = pred - ref
    sqi = np.array([to_float(r["sqi_score"]) for r in rows])
    subjects = sorted({r["subject"] for r in rows})
    print(f"Загружено {len(rows)} окон по {len(subjects)} записям из {dir_path.name} "
          f"(эталон: {ref_key})")

    coverage_mae_curve(sqi, err)
    component_correlations(rows, err)
    bland_altman(pred, ref)
    mae_by_time_and_hr(rows, err, ref)
    reference_wave_quality(rows)


if __name__ == "__main__":
    main()
