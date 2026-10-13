"""
Задача 51, пункт 1 (диагностика — НИЧЕГО не меняет в коде): проверяет,
предсказывает ли SQI ошибку НА УРОВНЕ ОКНА (не испытуемого — это то, для
чего SQI вообще существует: гейтинг конкретных измерений, а не среднее по
записи).

Использует уже накопленные scripts/eval_ubfc_subject.py --подобные
пооконные логи в ubfc_windows/*.csv (столбцы sqi_score и отдельные
компоненты SQI + error — см. WINDOW_FIELDS в eval_ubfc_subject.py).

Печатает:
  1. Корреляцию Пирсона каждой компоненты SQI с |error| (пул по ВСЕМ
     окнам ВСЕХ испытуемых — здесь это корректно: проверяется связь
     компоненты с ошибкой внутри записи, а не сравнение испытуемых между
     собой, которое требовало бы усреднения по испытуемому).
  2. Кривую "порог по sqi_score -> покрытие, MAE" — если SQI информативен,
     кривая должна монотонно (или почти) убывать по MAE при росте порога
     ценой покрытия. Сохраняет график в sqi_error_diagnosis.png.
  3. Явный вывод: информативен SQI на уровне окна или нет.

Использование:
    PYTHONPATH=src python3 scripts/diagnose_sqi_error_correlation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = REPO_ROOT / "ubfc_windows"

SQI_COMPONENTS = [
    "sqi_score", "spectral_snr_db", "harmonic_score",
    "cross_roi_agreement", "landmark_stability", "temporal_consistency",
]


def load_all_windows() -> list[dict]:
    rows = []
    for csv_path in sorted(WINDOWS_DIR.glob("*.csv")):
        subject = csv_path.stem
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("error") in (None, "", "None"):
                    continue
                row["subject"] = subject
                rows.append(row)
    return rows


def to_float(x):
    if x in (None, "", "None"):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def main() -> None:
    if not WINDOWS_DIR.exists():
        print(f"ОШИБКА: {WINDOWS_DIR} не найден — сначала прогоните scripts/eval_ubfc_subject.py "
              "хотя бы на нескольких испытуемых.", file=sys.stderr)
        sys.exit(1)

    rows = load_all_windows()
    subjects = sorted(set(r["subject"] for r in rows))
    print(f"Загружено {len(rows)} окон с валидной парой (предсказание, эталон) "
          f"по {len(subjects)} испытуемым: {', '.join(subjects)}")
    if len(rows) < 10:
        print("ОШИБКА: слишком мало окон для содержательной диагностики.", file=sys.stderr)
        sys.exit(1)

    abs_error = np.array([abs(to_float(r["error"])) for r in rows])

    print("\n=== Корреляция каждой компоненты SQI с |ошибка| (пул по всем окнам) ===")
    print(f"{'компонента':<22s}{'r':>8s}{'n':>6s}")
    for comp in SQI_COMPONENTS:
        values = np.array([to_float(r.get(comp)) for r in rows])
        mask = ~(np.isnan(values) | np.isnan(abs_error))
        n = int(mask.sum())
        if n < 3 or np.std(values[mask]) < 1e-9:
            print(f"{comp:<22s}{'—':>8s}{n:>6d}   (нет данных/константа)")
            continue
        r = float(np.corrcoef(values[mask], abs_error[mask])[0, 1])
        print(f"{comp:<22s}{r:>8.3f}{n:>6d}")

    print("\n=== Порог по sqi_score -> покрытие, MAE (по всем окнам, все испытуемые вместе) ===")
    sqi_scores = np.array([to_float(r["sqi_score"]) for r in rows])
    errors_signed = np.array([to_float(r["error"]) for r in rows])
    n_total = len(rows)

    thresholds = np.round(np.arange(0.0, 1.01, 0.05), 2)
    curve_thr, curve_cov, curve_mae, curve_n = [], [], [], []
    print(f"{'порог':>8s}{'покрытие':>10s}{'окон':>7s}{'MAE':>8s}")
    for thr in thresholds:
        mask = sqi_scores >= thr
        n = int(mask.sum())
        coverage = n / n_total
        mae = float(np.mean(np.abs(errors_signed[mask]))) if n > 0 else None
        curve_thr.append(thr)
        curve_cov.append(coverage)
        curve_mae.append(mae)
        curve_n.append(n)
        mae_str = f"{mae:.2f}" if mae is not None else "—"
        print(f"{thr:>8.2f}{coverage * 100:>9.1f}%{n:>7d}{mae_str:>8s}")

    # Явный вывод: смотрим, монотонно (с допуском на шум) ли убывает MAE
    # по мере роста порога, используя только точки с >=5 окнами (иначе
    # MAE на 2-3 последних окнах — чистый шум, не сигнал).
    valid = [(t, c, m, n) for t, c, m, n in zip(curve_thr, curve_cov, curve_mae, curve_n) if m is not None and n >= 5]
    verdict = "НЕДОСТАТОЧНО ДАННЫХ"
    if len(valid) >= 3:
        maes = np.array([v[2] for v in valid])
        # Ранговая корреляция порога с MAE — должна быть заметно отрицательной
        # (выше порог -> ниже MAE), если SQI информативен.
        thr_arr = np.array([v[0] for v in valid])
        rank_corr = float(np.corrcoef(thr_arr, maes)[0, 1])
        mae_drop = maes[0] - maes[-1]
        print(f"\nКорреляция порога и MAE по кривой: {rank_corr:.3f} "
              f"(MAE на пороге {valid[0][0]:.2f} = {maes[0]:.2f}, на пороге {valid[-1][0]:.2f} = {maes[-1]:.2f})")
        if rank_corr < -0.5 and mae_drop > 0.5:
            verdict = "SQI ИНФОРМАТИВЕН: MAE заметно и последовательно снижается с ростом порога"
        elif rank_corr < -0.2:
            verdict = "SQI СЛАБО ИНФОРМАТИВЕН: направление верное, но эффект небольшой/шумный"
        else:
            verdict = "SQI НЕ ИНФОРМАТИВЕН НА УРОВНЕ ОКНА: кривая плоская или не монотонная"
    print(f"\n=== ВЫВОД: {verdict} ===")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(curve_thr, curve_mae, "o-", color="tab:red", label="MAE")
        ax1.set_xlabel("порог sqi_score")
        ax1.set_ylabel("MAE (BPM)", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.plot(curve_thr, [c * 100 for c in curve_cov], "s--", color="tab:blue", label="покрытие")
        ax2.set_ylabel("покрытие (%)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        plt.title(f"Задача 51: покрытие/MAE по порогу sqi_score ({len(subjects)} испытуемых, {n_total} окон)\n{verdict}")
        fig.tight_layout()
        out_path = REPO_ROOT / "sqi_error_diagnosis.png"
        fig.savefig(out_path, dpi=120)
        print(f"\nГрафик сохранён в {out_path}")
    except ImportError:
        print("\n(matplotlib недоступен — график не сохранён, только текстовые таблицы выше)")


if __name__ == "__main__":
    main()
