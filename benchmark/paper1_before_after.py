"""
Таблица «до/после» по числам первой статьи (задание 4, раздел 5).

Числа берутся ИЗ СУЩЕСТВУЮЩИХ скриптов статьи: scripts/report_ubfc_split.py
и scripts/analyze_ubfc_windows.py прогоняются на обоих наборах артефактов
(results/fs_fix/before и .../after), их вывод сохраняется целиком рядом
(*_report.txt, *_analysis.txt) и разбирается. Колонка `source` в выгрузке
говорит, каким скриптом получено число.

Отдельно помечены как `новый расчёт` те величины, которых существующие
скрипты не считают: ошибки кратности, MAE при приборном эталоне и
Бланд-Альтман по опубликованным окнам (задание требует его отдельно от
варианта «по всем окнам», который печатает analyze_ubfc_windows.py).

    PYTHONPATH=src venv/bin/python3 benchmark/paper1_before_after.py
"""

from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FS_FIX = REPO_ROOT / "results" / "fs_fix"
PY_BIN = str(REPO_ROOT / "venv" / "bin" / "python3")

OCTAVE_HIGH = (1.7, 2.3)
OCTAVE_LOW = (0.43, 0.59)


def run_script(script: str, args: list[str]) -> str:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    proc = subprocess.run([PY_BIN, str(REPO_ROOT / "scripts" / script)] + args,
                          capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} упал: {proc.stderr[-500:]}")
    return proc.stdout


def parse_report(text: str) -> dict[str, float]:
    """Числа из report_ubfc_split.py, ключ — «раздел / метрика»."""
    out: dict[str, float] = {}
    section = ""
    for line in text.splitlines():
        m = re.match(r"=== (.+?) ===", line.strip())
        if m:
            section = m.group(1).split(" (")[0].strip()
            continue
        for pattern, name in (
            (r"MAE\s+по испытуемым:\s*([\d.]+) ± ([\d.]+)", "MAE по записям"),
            (r"RMSE по испытуемым:\s*([\d.]+) ± ([\d.]+)", "RMSE по записям"),
        ):
            m = re.search(pattern, line)
            if m:
                out[f"{section} / {name}"] = float(m.group(1))
                out[f"{section} / {name}, СКО"] = float(m.group(2))
        m = re.search(r"Среднее покрытие:\s*([\d.]+)%", line)
        if m:
            out[f"{section} / среднее покрытие, %"] = float(m.group(1))
        m = re.search(r"Бутстрэп 95% ДИ среднего MAE: \[([\d.]+), ([\d.]+)\]", line)
        if m:
            out[f"{section} / ДИ нижняя"] = float(m.group(1))
            out[f"{section} / ДИ верхняя"] = float(m.group(2))
        m = re.search(r"MAE (настроечная|отложенная):\s*([\d.]+)", line)
        if m:
            out[f"переобучение / MAE {m.group(1)}"] = float(m.group(2))
        m = re.search(r"Разрыв:\s*\+?(-?[\d.]+)", line)
        if m:
            out["переобучение / разрыв"] = float(m.group(1))
        m = re.search(r"(DATASET_\d), (настроечные|отложенные)\s+\(n=(\d+)\): MAE ([\d.]+)", line)
        if m:
            out[f"контроль формата / {m.group(1)} {m.group(2)} (n={m.group(3)})"] = float(m.group(4))
    return out


def parse_analysis(text: str) -> dict[str, float]:
    """Числа из analyze_ubfc_windows.py."""
    out: dict[str, float] = {}
    block = ""
    for line in text.splitlines():
        if "1. Кривая" in line:
            block = "sqi"
        elif "2. Корреляция компонент" in line:
            block = "corr"
        elif "3. Бланд-Альтман" in line:
            block = "ba"
        elif "4a. MAE по 10-секундным" in line:
            block = "time"
        elif "4б. MAE по диапазонам" in line:
            block = "band"
        elif "Качество эталонной волны" in line:
            block = "wave"

        if block == "sqi":
            m = re.match(r"\s*([\d.]+)\s+([\d.]+)%\s+(\d+)\s+([\d.]+)\s*$", line)
            if m and m.group(1) in ("0.00", "0.50", "0.70", "0.80", "0.90"):
                thr = m.group(1)
                out[f"кривая SQI / порог {thr}: покрытие, %"] = float(m.group(2))
                out[f"кривая SQI / порог {thr}: окон"] = float(m.group(3))
                out[f"кривая SQI / порог {thr}: MAE"] = float(m.group(4))
        if block == "corr":
            m = re.match(r"\s*(\w+)\s+(-?[\d.]+)\s+(\d+)\s*$", line)
            if m:
                out[f"корреляции компонент / {m.group(1)}"] = float(m.group(2))
        if block == "time":
            m = re.match(r"\s*(\d+-\d+) с\s+(\d+)\s+([\d.]+)\s*$", line)
            if m:
                out[f"MAE по времени / {m.group(1)} с"] = float(m.group(3))
        if block == "band":
            m = re.match(r"\s*(\d+-\d+) BPM\s+(\d+)\s+([\d.]+)\s*$", line)
            if m:
                out[f"MAE по частоте / {m.group(1)} BPM"] = float(m.group(3))

        m = re.search(r"корреляция порога с MAE: (-?[\d.]+)", line)
        if m:
            out["кривая SQI / корреляция порога с MAE"] = float(m.group(1))
        m = re.search(r"смещение: (-?[\d.]+) BPM", line)
        if m:
            out["Бланд-Альтман, все окна / смещение"] = float(m.group(1))
        m = re.search(r"SD разностей: ([\d.]+) BPM", line)
        if m:
            out["Бланд-Альтман, все окна / SD разностей"] = float(m.group(1))
        m = re.search(r"95% пределы согласия: \[(-?[\d.]+), \+?(-?[\d.]+)\] BPM", line)
        if m:
            out["Бланд-Альтман, все окна / предел нижний"] = float(m.group(1))
            out["Бланд-Альтман, все окна / предел верхний"] = float(m.group(2))
        m = re.search(r"доля окон вне пределов: ([\d.]+)%", line)
        if m:
            out["Бланд-Альтман, все окна / вне пределов, %"] = float(m.group(1))
        m = re.search(r"Загружено (\d+) окон по (\d+) записям", line)
        if m:
            out["общее / окон"] = float(m.group(1))
            out["общее / записей"] = float(m.group(2))
    return out


def _rows(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["subject"] = os.path.basename(f)[:-4]
                out.append(r)
    return out


def _num(value):
    if value in (None, "", "nan", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extra_metrics(root: Path, single_ref_col: str) -> dict[str, float]:
    """Величины, которых существующие скрипты статьи не считают."""
    out: dict[str, float] = {}

    for label, pattern, ref_col in (
        ("двухпроходный, эталон волна", str(root / "ubfc_windows_twopass" / "*.csv"), "bpm_ref_wave"),
        ("однопроходный, эталон прибор", str(root / "ubfc_windows" / "*.csv"), single_ref_col),
    ):
        tot = oc = 0
        for r in _rows(pattern):
            pred, ref = _num(r.get("bpm_pred")), _num(r.get(ref_col))
            if pred is None or ref is None or ref <= 0:
                continue
            tot += 1
            k = pred / ref
            if OCTAVE_HIGH[0] < k < OCTAVE_HIGH[1] or OCTAVE_LOW[0] < k < OCTAVE_LOW[1]:
                oc += 1
        out[f"ошибки кратности / {label}: окон"] = float(tot)
        out[f"ошибки кратности / {label}: ошибок"] = float(oc)

    # MAE при приборном эталоне — по записям, как и основная метрика
    per_record = []
    for f in sorted(glob.glob(str(root / "ubfc_windows_twopass" / "*.csv"))):
        errs = []
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("publishable") not in ("True", "1"):
                    continue
                pred, ref = _num(r.get("bpm_pred")), _num(r.get("bpm_ref_device"))
                if pred is not None and ref is not None:
                    errs.append(abs(pred - ref))
        if errs:
            per_record.append(float(np.mean(errs)))
    # ВАЖНО: это НЕ число статьи (3.56). В статье приборный эталон проходил
    # через фильтр правдоподобия (см. REFERENCE_MIN_PLAUSIBLE_BPM в
    # eval_ubfc_subject.py), который отбрасывал окна с отказавшим
    # пульсоксиметром; здесь берётся колонка bpm_ref_device как есть, без
    # фильтра, поэтому величина систематически выше. Число статьи
    # добавляется отдельной строкой ниже и только для «до»: прогон с
    # приборным эталоном после правки не повторялся.
    out["сравнение эталонов / MAE при приборном эталоне, без фильтра правдоподобия"] = (
        float(np.mean(per_record)) if per_record else float("nan")
    )

    # Бланд-Альтман ТОЛЬКО по опубликованным окнам
    diffs = []
    for r in _rows(str(root / "ubfc_windows_twopass" / "*.csv")):
        if r.get("publishable") not in ("True", "1"):
            continue
        pred, ref = _num(r.get("bpm_pred")), _num(r.get("bpm_ref_wave"))
        if pred is not None and ref is not None:
            diffs.append(pred - ref)
    d = np.array(diffs)
    out["Бланд-Альтман, опубликованные окна / окон"] = float(d.size)
    out["Бланд-Альтман, опубликованные окна / смещение"] = float(np.mean(d))
    out["Бланд-Альтман, опубликованные окна / SD разностей"] = float(np.std(d, ddof=1))
    out["Бланд-Альтман, опубликованные окна / предел нижний"] = float(np.mean(d) - 1.96 * np.std(d, ddof=1))
    out["Бланд-Альтман, опубликованные окна / предел верхний"] = float(np.mean(d) + 1.96 * np.std(d, ddof=1))

    # Пять записей с прежним систематическим занижением
    for rid in ("subject3", "subject4", "subject14", "subject27", "subject37"):
        path = root / "ubfc_windows_twopass" / f"{rid}.csv"
        if not path.exists():
            continue
        ratios, errs = [], []
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                pred, ref = _num(r.get("bpm_pred")), _num(r.get("bpm_ref_wave"))
                if pred is None or ref is None or ref <= 0:
                    continue
                ratios.append(pred / ref)
                if r.get("publishable") in ("True", "1"):
                    errs.append(abs(pred - ref))
        out[f"пять записей занижения / {rid}: медиана отношения"] = float(np.median(ratios))
        out[f"пять записей занижения / {rid}: MAE"] = float(np.mean(errs)) if errs else float("nan")
    return out


def paper_device_mae(root: Path) -> float | None:
    """MAE при приборном эталоне ровно так, как в статье — из готового
    ubfc_results_twopass_device.csv. Файл существует только для «до»:
    прогон с приборным эталоном после правки не повторялся."""
    path = root / "ubfc_results_twopass_device.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        vals = [float(r["mae_pub"]) for r in csv.DictReader(fh) if r["mae_pub"] not in ("", "None")]
    return float(np.mean(vals)) if vals else None


def collect(root: Path, single_ref_col: str, tag: str) -> dict[str, float]:
    report = run_script("report_ubfc_split.py", ["--results", str(root / "ubfc_results_twopass.csv")])
    analysis = run_script("analyze_ubfc_windows.py", ["--dir", str(root / "ubfc_windows_twopass")])
    (FS_FIX / f"{tag}_report.txt").write_text(report, encoding="utf-8")
    (FS_FIX / f"{tag}_analysis.txt").write_text(analysis, encoding="utf-8")
    values = {}
    values.update({k: v for k, v in parse_report(report).items()})
    values.update({k: v for k, v in parse_analysis(analysis).items()})
    values.update(extra_metrics(root, single_ref_col))
    paper_device = paper_device_mae(root)
    if paper_device is not None:
        values["сравнение эталонов / MAE при приборном эталоне (прогон статьи)"] = paper_device
    return values


SOURCE_BY_PREFIX = [
    ("ошибки кратности", "новый расчёт"),
    ("сравнение эталонов", "новый расчёт"),
    ("Бланд-Альтман, опубликованные окна", "новый расчёт"),
    ("пять записей занижения", "новый расчёт"),
    ("кривая SQI", "analyze_ubfc_windows.py"),
    ("корреляции компонент", "analyze_ubfc_windows.py"),
    ("Бланд-Альтман, все окна", "analyze_ubfc_windows.py"),
    ("MAE по времени", "analyze_ubfc_windows.py"),
    ("MAE по частоте", "analyze_ubfc_windows.py"),
    ("общее", "analyze_ubfc_windows.py"),
]


def source_of(key: str) -> str:
    for prefix, src in SOURCE_BY_PREFIX:
        if key.startswith(prefix):
            return src
    return "report_ubfc_split.py"


def main() -> None:
    FS_FIX.mkdir(parents=True, exist_ok=True)
    before = collect(FS_FIX / "before", "bpm_ref", "before")
    after = collect(FS_FIX / "after", "bpm_ref_device", "after")

    keys = sorted(set(before) | set(after))
    rows = []
    for key in keys:
        section, _, metric = key.partition(" / ")
        b, a = before.get(key), after.get(key)
        delta = "" if (b is None or a is None) else f"{a - b:+.6g}"
        rows.append({
            "section": section,
            "table": source_of(key),
            "metric": metric,
            "before": "" if b is None else f"{b:.6g}",
            "after": "" if a is None else f"{a:.6g}",
            "delta": delta,
        })
    path = FS_FIX / "paper1_before_after.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "table", "metric", "before", "after", "delta"])
        w.writeheader()
        w.writerows(rows)
    print(f"выгружено {len(rows)} строк в {path}")
    missing = [r["section"] + " / " + r["metric"] for r in rows if r["before"] == "" or r["after"] == ""]
    if missing:
        print("без пары до/после:", len(missing))
        for m in missing[:10]:
            print("  ", m)


if __name__ == "__main__":
    main()
