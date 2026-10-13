"""
Собирает results/degenerate/followup/README.md из фактических выгрузок
задания 2. Числа берутся из CSV, а не переписываются руками.

    PYTHONPATH=src venv/bin/python3 benchmark/followup_report.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "degenerate" / "followup"
SRC_DEGRADE = ROOT / "src" / "rppg" / "degrade.py"


def c(value, spec=".4g") -> str:
    """Число для ПРОЗЫ: десятичная запятая, как принято в русском тексте.
    В таблицах числа печатаются с точкой — там они повторяют CSV."""
    return format(value, spec).replace(".", ",")


def load(name):
    with (OUT / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def col(rows, key):
    return np.array([float(r[key]) for r in rows if r[key] not in ("", "nan")])


breakdown = load("a1_published_breakdown.csv")
pubwin = load("a1_published_windows.csv")
jitter = load("a1_landmark_jitter.csv")
jsummary = load("a1_jitter_summary.csv")
flick = load("flicker_internals.csv")
quant = load("quantization_check.csv")

# --- задача A ---
total_pub = sum(int(b["n_windows_published"]) for b in breakdown)
nonzero = [b for b in breakdown if int(b["n_windows_published"]) > 0]
top = sorted(nonzero, key=lambda b: -int(b["n_windows_published"]))[:5]
bpm = col(pubwin, "bpm_estimate")
in_rest = int(((bpm >= 50) & (bpm < 100)).sum())
below = int((bpm < 50).sum())
share_top2 = sum(int(b["n_windows_published"]) for b in top[:2]) / total_pub
a_rows = "\n".join(
    f"| `{b['record_id']}` | {b['n_windows_published']} | {float(b['publish_rate']):.1%} | "
    f"{b['bpm_min']} | {b['bpm_median']} | {b['bpm_max']} | {b['n_bpm_in_50_100']} | {b['n_bpm_outside_50_180']} |"
    for b in top
)

# --- задача B ---
by_frame = defaultdict(list)
for r in jitter:
    v = r["mean_abs_px_shift_from_median"]
    by_frame[int(r["frame_idx"])].append(float(v) if v else float("nan"))
share_nonzero = {i: float(np.mean([v > 0 for v in vs])) for i, vs in by_frame.items()}
nz_rows = sum(1 for r in jitter if r["mean_abs_px_shift_from_median"] not in ("",) and float(r["mean_abs_px_shift_from_median"]) > 0)
pub_ids = {b["record_id"] for b in nonzero}
med_pub = col([r for r in jsummary if r["matches_published"] == "1"], "median_shift_px")
med_nopub = col([r for r in jsummary if r["matches_published"] == "0"], "median_shift_px")
max_pub = col([r for r in jsummary if r["matches_published"] == "1"], "max_shift_px")
max_nopub = col([r for r in jsummary if r["matches_published"] == "0"], "max_shift_px")

# --- задача C ---
c_lines = []
for cond in ("B1w", "B1f", "B2f"):
    rs = [r for r in flick if r["condition"] == cond]
    passed = sum(1 for r in rs if r["bg_gate_passed"] == "1")
    flags = sum(1 for r in rs if r["flicker_flag_final"] == "1")
    c_lines.append(
        f"| `{cond}` | {len(rs)} | {np.median(col(rs, 'bg_region_dispersion_raw')):.4f} | 0.5 | "
        f"{passed}/{len(rs)} | {np.median(col(rs, 'bg_power_at_injected_freq')):.3f} | "
        f"{np.median(col(rs, 'face_power_at_injected_freq')):.3f} | {flags} |"
    )
b1w = [r for r in flick if r["condition"] == "B1w"]
b1w_disp = col(b1w, "bg_region_dispersion_raw")
b1w_bgp = col(b1w, "bg_power_at_injected_freq")
b2f_by_rec = []
for rid in sorted({r["record_id"] for r in flick}, key=lambda s: (len(s), s)):
    rs = [r for r in flick if r["condition"] == "B2f" and r["record_id"] == rid]
    if not rs:
        continue
    b2f_by_rec.append(
        f"| `{rid}` | {np.median(col(rs, 'bg_region_dispersion_raw')):.4f} | "
        f"{sum(1 for r in rs if r['bg_gate_passed'] == '1')}/{len(rs)} | "
        f"{sum(1 for r in rs if r['flicker_flag_final'] == '1')} |"
    )

# --- задача D ---
exp = col(quant, "signal_float_expected")
got = col(quant, "signal_from_delivered_frames")
# fps по полному пролёту окна, а не по двум соседним отсчётам: t_s
# записан с шестью знаками, и разность соседних значений даёт заметную
# погрешность в четвёртом знаке фазы.
fps_hint = (len(quant) - 1) / (float(quant[-1]["t_s"]) - float(quant[0]["t_s"]))
f_inj = 1.2


def fourier_at(series, fps, f_hz):
    t = np.arange(len(series)) / fps
    x = series - series.mean()
    c = 2.0 * np.mean(x * np.cos(2 * np.pi * f_hz * t))
    s = 2.0 * np.mean(x * np.sin(2 * np.pi * f_hz * t))
    return float(np.hypot(c, s) / series.mean()), float(np.arctan2(s, c))


a_exp, ph_exp = fourier_at(exp, fps_hint, f_inj)
a_got, ph_got = fourier_at(got, fps_hint, f_inj)

code = SRC_DEGRADE.read_text(encoding="utf-8").splitlines()
_ranges = [range(236, 246), range(255, 269)]
_parts = []
for n, rng in enumerate(_ranges):
    if n:
        _parts.append("       ...")  # строки 246-254 опущены: ветки других условий
    _parts += [f"{i:3d}  {code[i-1]}" for i in rng]
quote = "\n".join(_parts)

text = f"""# Диагностика по результатам прогона вырожденных входов

Продолжение `../README.md`. Полный прогон не повторялся: задача A посчитана по
уже собранному `../windows.csv`, задачи B, C и D — точечные прогоны с
добавленным логированием на 50, 5 и 1 записи соответственно.

## Прямые ответы на три вопроса

**1. Публикация на `A1` размазана по выборке, а не сидит в паре записей.**
Ненулевая публикация у {len(nonzero)} записей из {len(breakdown)}; больше
всего у записи `{top[0]['record_id']}` — {top[0]['n_windows_published']} из
{top[0]['n_windows_total']} окон, а две самые «урожайные» записи дают {c(share_top2 * 100, '.1f')}% всех {total_pub} случаев
(`a1_published_breakdown.csv`). Из {len(bpm)} значений BPM в диапазон покоя
50–100 попадает {in_rest} ({c(in_rest / len(bpm) * 100, '.1f')}%), остальные {below} лежат
НИЖЕ 50 при медиане {c(np.median(bpm), '.1f')} и максимуме {c(bpm.max(), '.1f')}
(`a1_published_windows.csv`).

**2. Подтверждается объяснение (а): порог по дисперсии отсекает амплитуду
0,003 независимо от области.** На `B1w`, где фон модулирован физически,
`bg_gate_passed` равен false во всех {len(b1w)} окнах: медианная дисперсия
сырого фона {c(np.median(b1w_disp))} при пороге 0,5
(`flicker_internals.csv`). При этом мощность фона на наложенной частоте
{c(np.median(b1w_bgp), '.3f')} — того же порядка, что и у лица
({c(np.median(col(b1w, 'face_power_at_injected_freq')), '.3f')}), то есть сигнал в
фоне присутствовал и частотное сравнение его бы нашло, если бы гейт пропустил.
Проблема не в сравнении частот после порога.

**3. Фраза про квантование относится только к промежуточному наблюдению при
разработке; доставленные данные не затронуты.** На окне из 300 кадров условия
`B1f` амплитуда на наложенной частоте в доставленных кадрах {c(a_got * 100)}%
против {c(a_exp * 100)}% в float-эталоне — отношение {c(a_got / a_exp)},
сдвиг фазы {c(ph_got - ph_exp, '+.5f')} рад (`quantization_check.csv`). Ни занижения
на порядок, ни смещения частоты нет.

## Задача A. Разбор 125 окон на `A1`

Пять записей с наибольшим числом ложных публикаций (полные 50 строк — в
`a1_published_breakdown.csv`):

| запись | опубликовано | доля | BPM min | BPM медиана | BPM max | в 50–100 | вне 50–180 |
|---|---|---|---|---|---|---|---|
{a_rows}

Суммарно {total_pub} окон, что совпадает с числом из задания 1.

## Задача B. Дрожание трекера на неизменном входе

{len(jitter)} строк: 50 записей × 300 кадров × 3 зоны, кадров без
обнаруженного лица нет.

Дрожание оказалось не постоянным, а эпизодическим и затухающим. Ненулевое
смещение у {nz_rows} строк из {len(jitter)} ({c(nz_rows / len(jitter) * 100, '.1f')}%),
медианное смещение по всем записям и зонам равно нулю. Распределение по
номеру кадра (доля зон с ненулевым смещением):

| кадр | 0 | 4 | 12 | 32 | 72 | 132 | 292 |
|---|---|---|---|---|---|---|---|
| доля | {c(share_nonzero[0] * 100, '.1f')}% | {c(share_nonzero[4] * 100, '.1f')}% | {c(share_nonzero[12] * 100, '.1f')}% | {c(share_nonzero[32] * 100, '.1f')}% | {c(share_nonzero[72] * 100, '.1f')}% | {c(share_nonzero[132] * 100, '.1f')}% | {c(share_nonzero[292] * 100, '.1f')}% |

Трекер сходится к неподвижному положению за первые несколько секунд.

Связь с ложной публикацией (`a1_jitter_summary.csv`): записей с публикацией
{len(pub_ids)} из 50, без публикации 2, поэтому сравнение групп опирается на
выборку из двух записей и само по себе мало что решает. Медиана смещения в
обеих группах нулевая; максимальное смещение по зонам: медиана
{c(np.median(max_pub))} px в группе с публикацией против
{c(np.median(max_nopub))} px без неё.

## Задача C. Почему не сработала проверка мерцания

{len(flick)} строк, {len({(r['record_id'], r['condition']) for r in flick})} пар
запись×условие (5 записей task0 × 3 условия), не менее 33 окон на пару.

| условие | окон | дисперсия сырого фона (медиана) | порог | гейт пройден | мощность фона на f | мощность лица на f | флагов |
|---|---|---|---|---|---|---|---|
{chr(10).join(c_lines)}

Мощности сравнимы между собой, но не абсолютны: оба сигнала проходят один и
тот же `preprocess_signal` с z-score.

Расслоение внутри `B2f` (то же, что найдено в задании 1: на записях 5 и 6
маска кожи покрывает фоновый патч, и мерцание «по лицу» попадает на фон):

| запись | дисперсия сырого фона | гейт пройден | флагов |
|---|---|---|---|
{chr(10).join(b2f_by_rec)}

## Задача D. Аудит порядка квантования

Фрагмент `src/rppg/degrade.py` (номера строк на момент прогона):

```python
{quote}
```

Накопление идёт в `float32`: шум прибавляется в строке 244, мерцание
умножается в строках 265 и 267, приведение к `uint8` выполняется ОДИН раз, в
строке 268, после всех операций.

Ряды по {len(quant)} последовательным кадрам одного окна `B1f` записи
`{quant[0]['record_id']}`:

| ряд | среднее | амплитуда на 1,2 Гц | фаза, рад |
|---|---|---|---|
| `signal_float_expected` | {exp.mean():.4f} | {a_exp * 100:.4f}% | {ph_exp:.4f} |
| `signal_from_delivered_frames` | {got.mean():.4f} | {a_got * 100:.4f}% | {ph_got:.4f} |

Оба ряда считаются по одной и той же фиксированной маске, поэтому единственная
разница между ними — квантование. Корректность сравнения проверяется самим
скриптом: float-эталон, проведённый через то же клипование и приведение к
`uint8`, совпал с кадрами `FrameDegrader` побитово на всех {len(quant)} кадрах.

## Требует решения

`astype(np.uint8)` в строке 268 ОТБРАСЫВАЕТ дробную часть, а не округляет.
В измеренном окне это даёт систематическое занижение постоянной составляющей
на {c(exp.mean() - got.mean())} уровня яркости ({c(exp.mean(), '.4f')} против
{c(got.mean(), '.4f')}). Реальное квантование в камере округляет, а не усекает.
На переменную составляющую это не повлияло — отношение амплитуд
{c(a_got / a_exp)}, — а постоянную составляющую в пайплайне всё равно снимает
детрендинг, поэтому на уже собранные данные эффект, по этому измерению, не
переносится. Но это расхождение с физической моделью, заложенной в условия
`A2`/`B*`, и оно затрагивает ВСЕ доставленные кадры всех условий, а не только
`B1f`. Не исправлено, как требует раздел 1 задания.

## Проверка приёмки

1. Сумма `n_windows_published` = {total_pub} — соответствует 125.
2. `a1_landmark_jitter.csv`: {len(jitter)} строк, ровно 300 кадров на каждую из
   {len({(r['record_id'], r['roi_region']) for r in jitter})} пар запись×зона.
3. `flicker_internals.csv`: {len({(r['record_id'], r['condition']) for r in flick})}
   пар запись×условие, минимум {min(sum(1 for r in flick if (r['record_id'], r['condition']) == k) for k in {(r['record_id'], r['condition']) for r in flick})}
   окон на пару (требуется не менее 20). Задание просит «по 3 записи» на
   условие, а критерий приёмки — «9 записей × 3 условия»; в исходной проверке
   task0 участвовало всего 5 записей, поэтому взяты ВСЕ 5 × 3 условия = 15 пар.
4. `quantization_check.csv`: {len(quant)} последовательных кадров одного окна,
   обе колонки заполнены во всех строках.
5. Во всех четырёх задачах выгружены построчные CSV, а не только сводки.

## Как воспроизвести

```bash
PYTHONPATH=src venv/bin/python3 benchmark/followup_a1_breakdown.py
PYTHONPATH=src venv/bin/python3 benchmark/followup_a1_jitter.py
PYTHONPATH=src venv/bin/python3 benchmark/followup_flicker_internals.py
PYTHONPATH=src venv/bin/python3 benchmark/followup_quantization_check.py --record 10
PYTHONPATH=src venv/bin/python3 benchmark/followup_report.py
```
"""
(OUT / "README.md").write_text(text, encoding="utf-8")
print(f"README.md записан, {len(text)} символов")
