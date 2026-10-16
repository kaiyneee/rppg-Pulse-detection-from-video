"""Собирает results/prv/checks/README.md из выгрузок benchmark/prv_checks.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "prv" / "checks"


def rd(n):
    return list(csv.DictReader((OUT / f"{n}.csv").open(encoding="utf-8")))


def q(a, s=".1f"):
    a = np.array([float(x) for x in a if x not in ("", None) and np.isfinite(float(x))])
    if not len(a):
        return "—"
    fm = lambda v: format(v, s).replace(".", ",")
    return f"{fm(np.median(a))} [{fm(np.percentile(a, 25))}…{fm(np.percentile(a, 75))}]"


pol, off, met = rd("polarity"), rd("offsets"), rd("shifted_metrics")
pol_rows = "\n".join(
    f"| {s} | {q([r['skew_video_v2_signal'] for r in pol if r['subset'] == s], '.2f')} | "
    f"{q([r['skew_reference_raw'] for r in pol if r['subset'] == s], '.2f')} | "
    f"{q([r['skew_reference_bandpassed'] for r in pol if r['subset'] == s], '.2f')} | "
    f"{sum(r['xcorr_sign'] == '-' for r in pol if r['subset'] == s)} из {sum(r['subset'] == s for r in pol)} | "
    f"{q([abs(float(r['xcorr_max_abs_r'])) for r in pol if r['subset'] == s], '.2f')} | "
    f"{q([r['xcorr_lag_s'] for r in pol if r['subset'] == s], '.3f')} |" for s in ("1", "2"))
off_rows = "\n".join(
    f"| {v} | {'вершины' if t == 'peaks' else 'начала'} | {q([r['median_offset_ms'] for r in off if r['variant'] == v and r['reference'] == t and r['subset'] == '1'], '.0f')} | "
    f"{q([r['median_offset_ms'] for r in off if r['variant'] == v and r['reference'] == t and r['subset'] == '2'], '.0f')} |"
    for v in ("V3", "V0") for t in ("peaks", "onsets"))
met_rows = []
for v in ("V3", "V0"):
    for t in ("peaks", "onsets"):
        for cor in ("0", "1"):
            for tol in ("150", "50"):
                cells = []
                for s in ("1", "2"):
                    rr = [r for r in met if r["variant"] == v and r["reference"] == t and r["own_offset_subtracted"] == cor
                          and r["tolerance_ms"] == tol and r["subset"] == s]
                    cells.append(f"{q([r['f1'] for r in rr], '.3f')} | {q([r['ibi_mae_ms'] for r in rr])} | "
                                 f"{q([r['ibi_bias_ms'] for r in rr])} | {q([r['n_ibi_pairs'] for r in rr], '.0f')}")
                met_rows.append(f"| {v} | {'вершины' if t == 'peaks' else 'начала'} | {'да' if cor == '1' else 'нет'} | ±{tol} | " + " | ".join(cells) + " |")

text = f"""# Проверки PRV по сохранённым ударам

Удары — из `results/prv` (эталон: вершины MSPTD; видео: V3 и V0); сигналы для проверки полярности
пересчитаны из кэша. Начала волны эталона — впадины MSPTD с тем же параболическим уточнением.
**Вычитание сдвига своей записи — поправка, оценённая по тем же данным; это диагностика, а не
метрика метода.** Медиана [межквартильный размах] по записям; подмножество 1 — 8 записей
(62,5 Гц), подмножество 2 — 42 (эталон с неравномерной осью).

## Полярность

Асимметрия (skew) непрерывного сигнала видео V2 (область V3) и контактной волны; взаимная
корреляция видео и эталона (та же полоса 0,7–4,0 Гц) в пределах ±0,5 с — знак и лаг в максимуме |r|.

| подмножество | skew видео | skew эталона (сырой) | skew эталона (после фильтра) | знак корреляции «−» | |r| | лаг, с |
|---|---|---|---|---|---|---|
{pol_rows}

## Сдвиг ударов видео относительно эталона

Медиана по ударам записи сдвига до ближайшего эталонного удара; ниже — распределение по записям, мс.
V0 — удары, составляющие ряд `_ibi_log` онлайн-цепочки.

| вариант | эталон | подмножество 1 | подмножество 2 |
|---|---|---|---|
{off_rows}

## Метрики со сдвигом своей записи и без

В ячейках по подмножествам: F1 | ошибка интервала, мс | смещение интервала, мс | пар интервалов.

| вариант | эталон | сдвиг вычтен | допуск | F1 (1) | ошибка (1) | смещение (1) | пар (1) | F1 (2) | ошибка (2) | смещение (2) | пар (2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(met_rows)}

```bash
PYTHONPATH=src venv/bin/python3 benchmark/prv_checks.py
PYTHONPATH=src venv/bin/python3 benchmark/prv_checks_report.py
```
"""
(OUT / "README.md").write_text(text, encoding="utf-8")
print("README.md записан")
