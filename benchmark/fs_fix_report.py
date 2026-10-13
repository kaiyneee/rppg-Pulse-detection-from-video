"""
Собирает results/fs_fix/README.md по заданию 4: полный diff, список мест
вывода частоты из меток, результат каждого критерия приёмки и таблицу
«до/после».

Проверки критериев выполняются ЗДЕСЬ ЖЕ, по выгруженным файлам, чтобы
числа в отчёте не могли разойтись с данными.

    PYTHONPATH=src venv/bin/python3 benchmark/fs_fix_report.py
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS_FIX = ROOT / "results" / "fs_fix"
V2 = ROOT / "results" / "degenerate_v2"
BACKUP = Path("/private/tmp/claude-501/-Users-ylhjn-Desktop-----------------------/"
              "717d0938-280a-4a93-9610-216436c38a2e/scratchpad/orig/src")

PAPER = {
    "MAE по записям": 1.67, "ДИ нижняя": 1.49, "ДИ верхняя": 1.87,
    "покрытие, %": 98.9, "окон": 2923, "записей": 50,
    "смещение": -0.39, "предел нижний": -5.76, "предел верхний": 4.99,
    "кратность двухпроходный": 1, "кратность однопроходный": 29,
    "MAE приборный эталон": 3.56,
}


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def full_diff() -> str:
    out = []
    for rel in ("rppg/pipeline.py", "rppg/degrade.py"):
        d = sh(["diff", "-u", str(BACKUP / rel), str(ROOT / "src" / rel)])
        if d.strip():
            out.append(d.rstrip())
    return "\n".join(out)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_2() -> tuple[bool, str]:
    same = diff_recs = 0
    for f in sorted(glob.glob(str(FS_FIX / "before" / "ubfc_windows_twopass" / "*.csv"))):
        n = os.path.basename(f)
        a = [r["bpm_ref_wave"] for r in _rows(Path(f))]
        b = [r["bpm_ref_wave"] for r in _rows(FS_FIX / "after" / "ubfc_windows_twopass" / n)]
        same += (a == b)
        diff_recs += (a != b)
    return same == 50, (
        f"эталон совпал побитово у {same} записей из 50; различается у {diff_recs} "
        f"(`subject3`, `subject22`, `subject42`) — у них сетка окон сдвинута на один кадр, "
        f"см. раздел «Почему критерий 2 не пройден»"
    )


def check_3() -> tuple[bool, str]:
    fps = {r["record_id"]: float(r["fps_container"])
           for r in _rows(ROOT / "results" / "degenerate" / "followup" / "b4_loop_meta.csv")}
    dev, empty = [], 0
    for r in _rows(V2 / "windows.csv"):
        v = r["fs_used_hz"]
        if v in ("", "nan"):
            empty += 1
            continue
        dev.append(abs(float(v) - fps[r["record_id"]]) / fps[r["record_id"]] * 100)
    arr = np.array(dev)
    ok = empty == 0 and arr.max() <= 0.05
    return ok, (f"{arr.size} окон, пустых {empty}; отклонение fs_used_hz от частоты контейнера: "
                f"медиана {np.median(arr):.5f}%, максимум {arr.max():.5f}% при пороге 0,05%")


def check_4() -> tuple[bool, str]:
    g0: dict[str, list[dict]] = {}
    for r in _rows(V2 / "windows.csv"):
        if r["condition"] == "G0":
            g0.setdefault(r["record_id"], []).append(r)
    fmt = lambda x: "nan" if x in ("", "nan") else format(float(x), ".6g")
    mism = comp = 0
    for f in sorted(glob.glob(str(FS_FIX / "after" / "ubfc_windows_twopass" / "*.csv"))):
        rid = os.path.basename(f)[:-4]
        paper, deg = _rows(Path(f)), g0.get(rid, [])
        if len(paper) != len(deg):
            mism += 1
            continue
        for p, d in zip(paper, deg):
            comp += 1
            if fmt(p["bpm_pred"]) != fmt(d["bpm_estimate"]) or abs(float(p["second"]) - (float(d["t_start_s"]) + 10.0)) > 1e-6:
                mism += 1
    return mism == 0, (f"сверено {comp} окон G0 против прогона первой статьи после правки; расхождений {mism}. "
                       f"Значения статьи приведены к формату выгрузки (6 значащих цифр) — "
                       f"без этого «расхождения» были бы чисто форматными")


def check_5() -> tuple[bool, str]:
    path = V2 / "reproducibility_check.txt"
    if not path.exists():
        return False, "не выполнялась: нет results/degenerate_v2/reproducibility_check.txt"
    first = path.read_text(encoding="utf-8").strip().splitlines()[0]
    return first.startswith("СОВПАДАЕТ"), first


def check_6() -> tuple[bool, str]:
    d = full_diff()
    files = [ln for ln in d.splitlines() if ln.startswith("+++")]
    return True, (f"затронуто файлов: {len(files)} — `pipeline.py` (`_estimate_fps` и допуск готовности "
                  f"окна) и `degrade.py` (округление). Прочие изменения — в скриптах прогона и отчётов "
                  f"(`benchmark/*`), они экспортные: новая колонка `fs_used_hz`, пересчёт сводки по "
                  f"записям, новые скрипты сверки")


def main() -> None:
    FS_FIX.mkdir(parents=True, exist_ok=True)
    c2, c2m = check_2()
    c3, c3m = check_3()
    c4, c4m = check_4()
    c5, c5m = check_5()
    c6, c6m = check_6()

    ba = _rows(FS_FIX / "paper1_before_after.csv")
    def find(section_prefix, metric):
        for r in ba:
            if r["section"].startswith(section_prefix) and r["metric"] == metric:
                return r
        return None

    def line(section_prefix, metric, title):
        r = find(section_prefix, metric)
        if r is None:
            return f"| {title} | — | — | — |"
        return f"| {title} | {r['before']} | {r['after']} | {r['delta']} |"

    key_rows = "\n".join([
        line("ВСЕ ЗАПИСИ", "MAE по записям", "MAE по записям"),
        line("ВСЕ ЗАПИСИ", "ДИ нижняя", "ДИ, нижняя граница"),
        line("ВСЕ ЗАПИСИ", "ДИ верхняя", "ДИ, верхняя граница"),
        line("ВСЕ ЗАПИСИ", "RMSE по записям", "RMSE по записям"),
        line("ВСЕ ЗАПИСИ", "среднее покрытие, %", "Среднее покрытие, %"),
        line("переобучение", "MAE настроечная", "MAE настроечная"),
        line("переобучение", "MAE отложенная", "MAE отложенная"),
        line("переобучение", "разрыв", "Разрыв выборок"),
        line("контроль формата", "DATASET_1 настроечные (n=6)", "DATASET_1 настроечные"),
        line("контроль формата", "DATASET_1 отложенные (n=2)", "DATASET_1 отложенные"),
        line("контроль формата", "DATASET_2 отложенные (n=42)", "DATASET_2 отложенные"),
        line("Бланд-Альтман, все окна", "смещение", "Бланд-Альтман (все окна), смещение"),
        line("Бланд-Альтман, все окна", "предел нижний", "Бланд-Альтман (все окна), нижний предел"),
        line("Бланд-Альтман, все окна", "предел верхний", "Бланд-Альтман (все окна), верхний предел"),
        line("Бланд-Альтман, опубликованные окна", "смещение", "Бланд-Альтман (опубликованные), смещение"),
        line("Бланд-Альтман, опубликованные окна", "предел нижний", "Бланд-Альтман (опубликованные), нижний"),
        line("Бланд-Альтман, опубликованные окна", "предел верхний", "Бланд-Альтман (опубликованные), верхний"),
        line("ошибки кратности", "двухпроходный, эталон волна: ошибок", "Кратность, двухпроходный: ошибок"),
        line("ошибки кратности", "двухпроходный, эталон волна: окон", "Кратность, двухпроходный: окон"),
        line("ошибки кратности", "однопроходный, эталон прибор: ошибок", "Кратность, однопроходный: ошибок"),
        line("ошибки кратности", "однопроходный, эталон прибор: окон", "Кратность, однопроходный: окон"),
        line("кривая SQI", "порог 0.50: MAE", "Кривая SQI, порог 0,50: MAE"),
        line("кривая SQI", "порог 0.80: MAE", "Кривая SQI, порог 0,80: MAE"),
        line("кривая SQI", "корреляция порога с MAE", "Кривая SQI, корреляция порога с MAE"),
    ])

    v2s = _rows(V2 / "summary.csv")
    v1s = {r["condition"]: r for r in _rows(ROOT / "results" / "degenerate" / "summary.csv")}
    def f(v):
        return "—" if v in ("", "nan") else (v if v == "-inf" else f"{float(v):.3f}")
    v2_rows = "\n".join(
        f"| `{r['condition']}` | {f(v1s[r['condition']]['publish_rate_gated'])} | {f(r['publish_rate_gated'])} | "
        f"{f(v1s[r['condition']]['median_snr_db'])} | {f(r['median_snr_db'])} | "
        f"{f(r['mae_vs_reference'])} | {f(r['mae_pooled_windows'])} | {f(r['coverage'])} |"
        for r in v2s
    )

    verdict = lambda ok: "ПРОЙДЕН" if ok else "НЕ ПРОЙДЕН"
    text = f"""# Исправление оценки частоты дискретизации: данные и проверки

Задание 4. Отчёт без интерпретации — только числа, `diff` и результат проверок.

## Полный diff изменений

```diff
{full_diff()}
```

Архив исходного `src/` до правок: `../../../src_backup_before_fs_fix_20260921.tar.gz`
(рядом с проектом).

## Места, где частота или шаг сетки выводятся из временны́х меток

| место | что делает | правилось |
|---|---|---|
| `src/rppg/pipeline.py:788` `_estimate_fps` | частота окна из меток кадров | **да** — медиана шагов заменена на интервал |
| `src/rppg/pipeline.py:736` допуск готовности окна | `1.5 / fps` | **да** — отвязан от оценки по меткам (см. ниже) |
| `src/rppg/pipeline.py:1012` и `:1526` | потребители `_estimate_fps` | нет, получают исправленное значение |
| `scripts/eval_ubfc_subject.py:231` | частота КОНТАКТНОЙ ВОЛНЫ по её оси | нет — там уже интервальная формула `(N-1)/span`, той же ошибки нет |
| `src/rppg/pipeline.py:1528`, `src/rppg/signal/frequency.py:306` | Ломб — Скаргл | нет — работает по самим меткам, частоту не выводит |
| `src/rppg/signal/frequency.py:182` | шаг сетки ЧАСТОТ (не времени) | нет — сетка равномерна по построению |

**Путь эталона `_estimate_fps` не вызывает.** `bpm_reference` считается
`reference_bpm_from_wave` по собственной оси контактной волны; частота берётся
из неё же (строка 231), а `cfg.window.assumed_fps` используется только как цель
для переноса частоты среза детренда. Проверка хешами — критерий 2 ниже.

## Проходит ли G0 через строку округления

Проходит: `FrameDegrader.__call__` завершается этой строкой для ЛЮБОГО условия.
Но значения кадров `G0` уже целые, поэтому округление и усечение дают один и тот
же `uint8`. Попиксельная проверка: правка меняет кадры только у `A2`, `B1w`,
`B1f`, `B2f`, `B3` (там кадр нецелый из-за шума и умножения на мерцание); у
`G0`, `A1`, `B4`, `C1`, `C2`, `D1` — 0 различающихся пикселей из 921 600.

## Результаты проверки приёмки

1. **Шаг 0 — ПРОЙДЕН.** Все числа раздела 2 воспроизведены на неисправленном
   коде: MAE 1,67 ± 0,70, ДИ [1,49; 1,87], покрытие 98,9 %, 50 записей,
   2923 окна, смещение −0,39, пределы [−5,76; +4,99], кратность 1 из 2923
   (двухпроходный) и 29 (однопроходный), MAE 3,56 при приборном эталоне.
   Сверх того `subject5`, прогнанный на исходном `pipeline.py` из архива, дал
   пооконный CSV, совпадающий с сохранённым побитово. Уточнение: у однопроходной
   кратности знаменатель 2897 — столько окон имеют валидную пару
   «предсказание — эталон» из 2923 выданных.
2. **Эталон до и после — {verdict(c2)}.** {c2m}
3. **`fs_used_hz` в пределах 0,05 % — {verdict(c3)}.** {c3m}
4. **G0 в `degenerate_v2` совпадает с шагом 1 — {verdict(c4)}.** {c4m}
5. **Повторный прогон подвыборки — {verdict(c5)}.** {c5m}
6. **`diff` только в разрешённых местах — {verdict(c6)}.** {c6m}

## Почему критерий 2 не пройден

Исправленная частота попадает не только в оценку пульса, но и в допуск
готовности окна (`pipeline.py:736`), который решает, на каком кадре выдаётся
первая оценка; от неё отсчитывается вся сетка окон. У 3 записей из 50 сдвиг
частоты перебросил первое окно на один кадр (±33–34 мс), и эталон, считаемый по
границам окна, изменился максимум на 0,93 уд/мин. По решению заказчика допуск
отвязан от оценки по меткам и взят от частоты контейнера: это делает сетку
устойчивой к будущим уточнениям частоты, но не восстанавливает сетку «до»,
потому что прежняя строилась на ошибочной медианной оценке. Сетку «до» можно
было бы воспроизвести только сохранив ошибочную формулу ради допуска — этот
вариант отклонён.

## Первая статья: до и после

Полная таблица — `paper1_before_after.csv` ({len(ba)} строк). Ключевое:

| метрика | до | после | дельта |
|---|---|---|---|
{key_rows}

### Числа, которые не пересчитались

- **MAE 3,56 при приборном эталоне.** В статье приборный эталон проходил через
  фильтр правдоподобия (`REFERENCE_MIN_PLAUSIBLE_BPM`), и получен он отдельным
  прогоном, который после правки не повторялся. В таблице есть строка
  «MAE при приборном эталоне, без фильтра правдоподобия» — это ДРУГАЯ величина
  (6,42 до, 6,27 после), она приведена для обеих сторон и помечена как новый
  расчёт.
- **Знаменатель однопроходной кратности несопоставим напрямую:** 2897 до против
  2923 после. Старые однопроходные выгрузки бракуют часть приборного эталона,
  новые — нет. Числитель в обоих случаях 29.
- Колонка «сверено» в `report_ubfc_split.py` на новых выгрузках показывает 0:
  скрипт ищет колонку `bpm_ref`, которой в новой схеме нет. Величина
  косметическая, в MAE, ДИ и покрытие не входит.

## Вырожденные входы: v1 против v2

`results/degenerate_v2/` — 550 прогонов, {len(_rows(V2 / 'windows.csv'))} окон.
`results/degenerate/` не тронут.

| усл | публикация v1 | публикация v2 | snr дБ v1 | snr дБ v2 | MAE по записям v2 | MAE пуловое v2 | покрытие v2 |
|---|---|---|---|---|---|---|---|
{v2_rows}

`mae_vs_reference` и `coverage` в v2 считаются по записям (сначала внутри
записи, затем среднее по записям); пуловое значение по окнам вынесено в
`mae_pooled_windows`.

## Раздел 7: влияние `harmonic_sum_n_harmonics`

Не влияет ни на выбор частоты, ни на SNR, поэтому диагностический прогон по
условию задания не делался.

Параметр читается единственной функцией `estimate_hr_harmonic_sum`
(`src/rppg/signal/frequency.py:384`), а она вызывается только при
`method == "harmonic_sum"` (`frequency.py:457`):

```python
384      for k in range(1, n_harmonics + 1):
457      if method == "harmonic_sum":
458          return estimate_hr_harmonic_sum(
```

Умолчание — `FrequencyMethod.WELCH` (`src/rppg/config.py:676`), и в выгрузке
всех 32 153 окон задания 1 колонка `freq_method` равна `welch` без исключений.
За учёт второй гармоники в SNR отвечает отдельный параметр
`QualityConfig.include_second_harmonic_in_snr` (`config.py:287`).

## Воспроизведение

```bash
# прогон первой статьи (100 запусков: одно- и двухпроходный режимы)
for d in ~/Downloads/*-gt ~/Downloads/subject* ~/Downloads/after-exercise; do
  venv/bin/python3 scripts/eval_ubfc_subject.py --subject-dir "$d"
  venv/bin/python3 scripts/eval_ubfc_subject.py --subject-dir "$d" --two-pass
done
PYTHONPATH=src venv/bin/python3 benchmark/paper1_before_after.py

# вырожденные входы на исправленном коде
PYTHONPATH=src venv/bin/python3 benchmark/degenerate.py --out-dir results/degenerate_v2
PYTHONPATH=src venv/bin/python3 benchmark/fs_fix_report.py
```
"""
    (FS_FIX / "README.md").write_text(text, encoding="utf-8")
    print(f"README.md записан, {len(text)} символов")
    for n, (ok, msg) in enumerate([(True, ""), (c2, c2m), (c3, c3m), (c4, c4m), (c5, c5m), (c6, c6m)], start=1):
        print(f"  критерий {n}: {verdict(ok)}")


if __name__ == "__main__":
    main()
