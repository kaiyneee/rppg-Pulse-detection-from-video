"""
Собирает results/degenerate/followup/README_b4.md из выгрузок задания 3.
Числа берутся из CSV, цитаты кода — из файлов по номерам строк.

    PYTHONPATH=src venv/bin/python3 benchmark/followup_b4_report.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "degenerate" / "followup"


def quote(path: Path, ranges, gap="       ..."):
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = []
    for n, rng in enumerate(ranges):
        if n:
            parts.append(gap)
        parts += [f"{i:3d}  {lines[i-1]}" for i in rng]
    return "\n".join(parts)


def c(value, spec=".4g") -> str:
    return format(value, spec).replace(".", ",")


rows = list((OUT / "b4_loop_meta.csv").open(encoding="utf-8") and csv.DictReader((OUT / "b4_loop_meta.csv").open(encoding="utf-8")))
peaks = list(csv.DictReader((OUT / "b4_ref_peaks.csv").open(encoding="utf-8")))

lp = np.array([float(r["loop_period_s"]) for r in rows])
exact_3s = [r["record_id"] for r in rows if abs(float(r["loop_period_s"]) - 3.0) < 1e-9]
fs_counts = Counter(r["fs_used_hz"] for r in rows)
lf_counts = Counter(int(r["loop_n_frames"]) for r in rows)
npk = np.array([int(r["ref_n_peaks"]) for r in rows])
out_of_range = [(r["record_id"], int(r["ref_n_peaks"])) for r in rows if not 2 <= int(r["ref_n_peaks"]) <= 8]
diffs = sorted(
    ((r["record_id"], float(r["ref_hr_loop_bpm"]), float(r["ref_hr_g0_window0_bpm"]),
      abs(float(r["ref_hr_loop_bpm"]) - float(r["ref_hr_g0_window0_bpm"]))) for r in rows),
    key=lambda x: -x[3],
)
big = [d for d in diffs if d[3] > 15.0]
mismatch_rule = [r["record_id"] for r in rows
                 if int(r["loop_n_frames"]) != int(round(3.0 * float(r["fps_container"])))]
fps_eff_delta = max(abs(float(r["fps_container"]) - float(r["fps_effective"])) for r in rows)

fs_table = "\n".join(
    f"| {k.replace('.', ',')} | {v} | {c(1000.0 / float(k), '.0f')} |"
    for k, v in sorted(fs_counts.items(), key=lambda kv: float(kv[0]))
)
lf_table = ", ".join(f"{k} кадров у {v} записей" for k, v in sorted(lf_counts.items()))
big_table = "\n".join(
    f"| `{rid}` | {c(loop, '.1f')} | {c(g0, '.1f')} | {c(d, '.1f')} |" for rid, loop, g0, d in big
) or "| — | — | — | — |"

text = f"""# Метаданные петли условия B4

Продолжение `README.md`. Ни 550 прогонов, ни сам B4 не пересчитывались:
видео только сканируются ради числа кадров и меток контейнера, а эталон
фрагмента считается по контактной волне. Детектор пиков пайплайна не
использовался — только `scipy.signal.find_peaks` с параметрами из задания
(`distance` = 0,33 с, то есть 180 уд/мин; `prominence` = 0,3·СКО).

## Цитаты кода

Построение петли, `src/rppg/degrade.py`:

```python
{quote(ROOT / "src" / "rppg" / "degrade.py", [range(164, 187), range(246, 249)])}
```

Временна́я ось офлайн-прогона, `benchmark/degenerate.py`:

```python
{quote(ROOT / "benchmark" / "degenerate.py", [range(231, 239), range(267, 277)])}
```

Вывод частоты дискретизации из меток, `src/rppg/pipeline.py`:

```python
{quote(ROOT / "src" / "rppg" / "pipeline.py", [range(776, 785)])}
```

## Ответы на три вопроса

**1. Число кадров в петле.** `n_loop = int(round(LOOP_SECONDS * fps))`,
строка 167 `degrade.py`, где `LOOP_SECONDS = 3.0`, а `fps` — частота
КОНТЕЙНЕРА (`cap.get(cv2.CAP_PROP_FPS)`, строка 234 `degenerate.py`).
Округление арифметическое (`round`), не усечение. Отбор идёт по индексу
кадра (`idx < n_loop`, строка 174), а не по временны́м меткам. Правило
выполняется для всех 50 записей{'' if not mismatch_rule else ': кроме ' + ', '.join(mismatch_rule)}.
Фактические значения: {lf_table}.

**2. Временны́е метки на выходе B4 — синтезированные по частоте кадров,
сквозные по всей записи.** Кадр петли повторяется по содержимому
(`frames[idx % len(frames)]`, строка 248 `degrade.py`), но метка берётся не
из петли и не из контейнера: прогон подаёт
`int(frame_idx / fps * 1000)` (строка 273 `degenerate.py`), где `frame_idx`
монотонно растёт по всей записи. Метки исходной записи по порядку при этом
численно совпали бы с синтезированными: расхождение
`fps_container` и `fps_effective` по меткам контейнера равно
{c(fps_eff_delta, '.6f')} на всех 50 записях. Существенно другое —
целочисленное усечение до миллисекунд, см. пункт 3.

**3. Уэлч получает частоту, выведенную ИЗ МЕТОК, а не `assumed_fps = 30.0`.**
`RPPGPipeline._estimate_fps` (строки 776–784) возвращает
`1 / median(diff(timestamps))`, и к `assumed_fps` обращается только если
меток меньше двух или все шаги нулевые — на этих данных такого не бывает.
Поскольку метки усечены до целых миллисекунд, медиана шага тоже целое число
миллисекунд, и `fs_used_hz` принимает лишь несколько дискретных значений:

| fs_used_hz | записей | медиана шага, мс |
|---|---|---|
{fs_table}

Колонка `time_base_used` в выгрузке равна `timestamps` у всех 50 записей.

## Период петли

`loop_period_s` (разность поданных меток между выходными кадрами
`loop_n_frames` и 0) лежит в пределах {c(lp.min(), '.4f')}…{c(lp.max(), '.4f')} с
при медиане {c(np.median(lp), '.4f')} с. Ровно 3,000 с он составляет у
{len(exact_3s)} записи из {len(rows)} ({', '.join('`' + r + '`' for r in exact_3s)}),
у остальных {len(rows) - len(exact_3s)} отличается.

## Списки по критериям приёмки

**Критерий 3, `ref_n_peaks` вне диапазона 2…8:**
{'нет таких записей; фактический диапазон ' + str(npk.min()) + '…' + str(npk.max()) + ' пиков.' if not out_of_range else ', '.join(f'`{r}` ({n})' for r, n in out_of_range)}

**Критерий 4, `|ref_hr_loop_bpm − ref_hr_g0_window0_bpm| > 15`:**
записей с таким расхождением — {len(big)}.

| запись | пульс петли, уд/мин | G0 окно 0, уд/мин | разница |
|---|---|---|---|
{big_table}

## Выгрузки

- `b4_loop_meta.csv` — {len(rows)} строк, по одной на запись, пустых значений нет.
- `b4_ref_peaks.csv` — {len(peaks)} строк: сырые пики контактной волны на
  интервале петли, по которым посчитан `ref_hr_loop_bpm`.

Интервал для эталона взят по меткам ВИДЕО (реальное время записи, к которому
привязана контактная волна) — от первого кадра петли до кадра
`loop_n_frames − 1` включительно. Приведение к равномерной сетке повторяет
процедуру основной работы (`scripts/eval_ubfc_subject.py::reference_bpm_from_wave`):
сохраняются границы интервала и число отсчётов, интерполяция линейная.

## Воспроизведение

```bash
PYTHONPATH=src venv/bin/python3 benchmark/followup_b4_loop_meta.py
PYTHONPATH=src venv/bin/python3 benchmark/followup_b4_report.py
```
"""
(OUT / "README_b4.md").write_text(text, encoding="utf-8")
print(f"README_b4.md записан, {len(text)} символов")
