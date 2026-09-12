"""
Итоговый отчёт по ubfc_results.csv с разделением на НАСТРОЕЧНУЮ и ОТЛОЖЕННУЮ
выборки — проверка на переобучение.

Настроечные (на них велась разработка задач 42-52): 5, 6, 7, 8, 10, 11.
Отложенные: всё остальное — 12, after-exercise и все subject*.
Разница средних между группами и есть мера переобучения: если отложенная
выборка заметно хуже настроечной, значит правки подгонялись под шесть записей,
а не улучшали метод.

Использование:
    PYTHONPATH=src python3 scripts/report_ubfc_split.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "ubfc_results.csv"

# Шесть записей, на которых велась разработка (задачи 49-52) — их результат
# НЕ является честной оценкой обобщающей способности, только отложенная выборка.
TUNING_SUBJECTS = {"5", "6", "7", "8", "10", "11"}

# Записи формата DATASET_1 (gtdump.xmp). Важно для честного вывода о
# переобучении: ВСЕ шесть настроечных записей — DATASET_1, а отложенная
# выборка почти целиком DATASET_2. У этих форматов РАЗНОЕ качество эталонной
# волны (62.5 Гц равномерная против ~30 Гц неравномерной, см.
# reference_bpm_from_wave), и наивное сравнение "настроечные против
# отложенных" смешивает эффект подгонки с эффектом формата. Поэтому
# сравнение ведётся ещё и ВНУТРИ формата — там, где выборки сопоставимы.
DATASET_1_SUBJECTS = {"5", "6", "7", "8", "10", "11", "12", "after-exercise"}


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42):
    """95% ДИ среднего ресемплингом ИСПЫТУЕМЫХ (не окон) — окна внутри одной
    записи не независимы, ресемплинг окон занизил бы интервал."""
    values = values[~np.isnan(values)]
    if len(values) < 4:
        return None, None
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scored_windows(subject: str) -> tuple[int, int]:
    """(сколько окон реально сверено с эталоном, сколько всего выдано).

    Эти числа расходятся, когда часть эталона забракована как недостоверная
    (отказ пульсоксиметра — см. REFERENCE_MIN_PLAUSIBLE_BPM в
    eval_ubfc_subject.py): у subject20 битые 51% отсчётов эталона, и MAE там
    считается по сильно меньшему числу окон, чем выдал пайплайн. Без этой
    колонки читатель решил бы, что метрика опирается на все окна записи.
    """
    path = REPO_ROOT / "ubfc_windows" / f"{subject}.csv"
    if not path.exists():
        return -1, -1
    total = scored = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if row.get("bpm_ref") not in (None, "", "None"):
                scored += 1
    return scored, total


def summarize(rows: list[dict], label: str) -> dict | None:
    if not rows:
        print(f"\n=== {label}: нет данных ===")
        return None
    mae = np.array([to_float(r.get("mae_pub")) for r in rows])
    rmse = np.array([to_float(r.get("rmse_pub")) for r in rows])
    cov = np.array([to_float(r.get("coverage")) for r in rows])
    lo, hi = bootstrap_ci(mae)

    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"{'запись':<18s}{'окон':>6s}{'сверено':>9s}{'покрытие':>10s}{'MAE':>8s}{'RMSE':>8s}{'эталон HR':>14s}")
    for r in sorted(rows, key=lambda x: to_float(x.get("mae_pub"))):
        scored, total = scored_windows(r["subject"])
        mark = " !" if 0 <= scored < 0.8 * max(total, 1) else ""
        print(
            f"{r['subject']:<18s}{r.get('n_windows',''):>6s}{scored:>8d}{mark:<1s}"
            f"{to_float(r.get('coverage'))*100:>9.1f}%"
            f"{to_float(r.get('mae_pub')):>8.2f}{to_float(r.get('rmse_pub')):>8.2f}"
            f"{to_float(r.get('gt_min_bpm')):>7.0f}-{to_float(r.get('gt_max_bpm')):<6.0f}"
        )
    print("  ! — сверено менее 80% окон: часть эталона забракована как недостоверная "
          "(отказ пульсоксиметра), MAE опирается на меньшую выборку")
    print(f"  MAE  по испытуемым: {np.nanmean(mae):.2f} ± {np.nanstd(mae, ddof=1):.2f}")
    print(f"  RMSE по испытуемым: {np.nanmean(rmse):.2f} ± {np.nanstd(rmse, ddof=1):.2f}")
    print(f"  Среднее покрытие:   {np.nanmean(cov)*100:.1f}%")
    if lo is not None:
        print(f"  Бутстрэп 95% ДИ среднего MAE: [{lo:.2f}, {hi:.2f}]")
    return {"mae": mae, "rmse": rmse, "cov": cov, "ci": (lo, hi)}


def compare_passes() -> None:
    """Сравнение однопроходной и двухпроходной методик по-испытуемому.

    Два файла — две РАЗНЫЕ методики измерения, а не уточнение одной: смысл
    сравнения в том, чтобы видеть, где именно второй проход помогает
    (холодностартовые октавные ошибки) и не портит ли он что-то ещё.
    """
    single, double = REPO_ROOT / "ubfc_results.csv", REPO_ROOT / "ubfc_results_twopass.csv"
    if not (single.exists() and double.exists()):
        print("Для сравнения нужны оба файла: ubfc_results.csv и ubfc_results_twopass.csv", file=sys.stderr)
        sys.exit(1)
    a = {r["subject"]: r for r in csv.DictReader(open(single, newline="", encoding="utf-8"))}
    b = {r["subject"]: r for r in csv.DictReader(open(double, newline="", encoding="utf-8"))}
    common = sorted(set(a) & set(b))

    deltas = []
    print(f"{'запись':<16s}{'MAE 1 проход':>14s}{'MAE 2 прохода':>15s}{'разница':>10s}")
    for s_ in common:
        m1, m2 = to_float(a[s_].get("mae_pub")), to_float(b[s_].get("mae_pub"))
        d = m2 - m1
        deltas.append(d)
        mark = "  <<<" if abs(d) > 1.0 else ""
        print(f"{s_:<16s}{m1:>14.2f}{m2:>15.2f}{d:>+10.2f}{mark}")

    deltas = np.array(deltas)
    m1 = np.nanmean([to_float(a[s_].get("mae_pub")) for s_ in common])
    m2 = np.nanmean([to_float(b[s_].get("mae_pub")) for s_ in common])
    print(f"\n  Среднее MAE: {m1:.2f} (1 проход) -> {m2:.2f} (2 прохода), разница {m2-m1:+.2f} BPM")
    print(f"  Улучшилось у {(deltas < -0.01).sum()} записей, ухудшилось у {(deltas > 0.01).sum()}, "
          f"без изменений у {(np.abs(deltas) <= 0.01).sum()}")
    if (deltas > 0.01).any():
        worst = [(common[i], deltas[i]) for i in np.argsort(-deltas)[:5] if deltas[i] > 0.01]
        print("  Где второй проход УХУДШИЛ: " + ", ".join(f"{s_} ({d:+.2f})" for s_, d in worst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Отчёт по ubfc_results.csv с разбиением выборки")
    parser.add_argument("--results", type=str, default=None,
                        help="Путь к файлу результатов (по умолчанию ubfc_results.csv)")
    parser.add_argument("--compare", action="store_true",
                        help="Сравнить ubfc_results.csv и ubfc_results_twopass.csv по-испытуемому")
    args = parser.parse_args()

    if args.compare:
        compare_passes()
        return

    global RESULTS_CSV
    if args.results:
        RESULTS_CSV = Path(args.results)

    if not RESULTS_CSV.exists():
        print(f"ОШИБКА: {RESULTS_CSV} не найден — сначала прогоните eval_ubfc_subject.py", file=sys.stderr)
        sys.exit(1)

    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tuning = [r for r in rows if r["subject"] in TUNING_SUBJECTS]
    heldout = [r for r in rows if r["subject"] not in TUNING_SUBJECTS]

    all_stats = summarize(rows, "ВСЕ ЗАПИСИ")
    t_stats = summarize(tuning, "НАСТРОЕЧНАЯ выборка (на ней велась разработка)")
    h_stats = summarize(heldout, "ОТЛОЖЕННАЯ выборка (не использовалась при разработке)")

    if t_stats is not None and h_stats is not None:
        t_mae, h_mae = np.nanmean(t_stats["mae"]), np.nanmean(h_stats["mae"])
        gap = h_mae - t_mae
        print("\n=== ПРОВЕРКА НА ПЕРЕОБУЧЕНИЕ ===")
        d1_tune = [r for r in tuning if r["subject"] in DATASET_1_SUBJECTS]
        d1_held = [r for r in heldout if r["subject"] in DATASET_1_SUBJECTS]
        d2_held = [r for r in heldout if r["subject"] not in DATASET_1_SUBJECTS]
        print(f"  MAE настроечная: {t_mae:.2f}")
        print(f"  MAE отложенная:  {h_mae:.2f}")
        print(f"  Разрыв:          {gap:+.2f} BPM")
        # Пересечение бутстрэп-интервалов — грубый, но честный признак того,
        # что разрыв не отличим от шума при таком объёме выборки.
        t_lo, t_hi = t_stats["ci"]
        h_lo, h_hi = h_stats["ci"]
        if None not in (t_lo, t_hi, h_lo, h_hi):
            overlap = not (t_hi < h_lo or h_hi < t_lo)
            print(f"  ДИ настроечной [{t_lo:.2f}, {t_hi:.2f}] и отложенной [{h_lo:.2f}, {h_hi:.2f}]: "
                  f"{'ПЕРЕСЕКАЮТСЯ' if overlap else 'НЕ пересекаются'}")
            if overlap:
                print("  ВЫВОД (без учёта формата): явных признаков переобучения нет")
            else:
                print("  ВЫВОД (без учёта формата): разрыв статистически заметен")

        # Контроль на смешение с форматом: сравниваем сопоставимое —
        # настроечные и отложенные записи ВНУТРИ DATASET_1.
        if d1_tune and d1_held:
            a = np.nanmean([to_float(r.get("mae_pub")) for r in d1_tune])
            b = np.nanmean([to_float(r.get("mae_pub")) for r in d1_held])
            c = (np.nanmean([to_float(r.get("mae_pub")) for r in d2_held])
                 if d2_held else float("nan"))
            print(f"\n  Контроль на формат эталона:")
            print(f"    DATASET_1, настроечные (n={len(d1_tune)}): MAE {a:.2f}")
            print(f"    DATASET_1, отложенные  (n={len(d1_held)}): MAE {b:.2f}")
            if d2_held:
                print(f"    DATASET_2, отложенные  (n={len(d2_held)}): MAE {c:.2f}")
            if b <= a + 0.3:
                print("    ВЫВОД: внутри одного формата отложенные записи НЕ хуже настроечных —")
                print("    разрыв выше объясняется РАЗНЫМ КАЧЕСТВОМ ЭТАЛОНА у двух форматов")
                print("    (62.5 Гц равномерная волна против ~30 Гц неравномерной), а не подгонкой.")
            else:
                print("    ВЫВОД: отложенные хуже настроечных и ВНУТРИ одного формата —")
                print("    это уже нельзя списать на формат, признак переобучения остаётся.")


if __name__ == "__main__":
    main()
