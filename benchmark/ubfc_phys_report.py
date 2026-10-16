"""
Собирает results/ubfc_phys/README.md из выгрузок benchmark/ubfc_phys.py.

    PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys_report.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ubfc_phys"
CONFIGS = ["legacy_welch8", "fft_best_snr", "welch8_spectral_mean", "fft_spectral_mean"]
TASKS = ["T1", "T2", "T3"]
FULL = {"s1", "s2", "s3", "s4", "s5", "s10", "s11"}


def rd(name, sub=""):
    return list(csv.DictReader((OUT / sub / f"{name}.csv").open(encoding="utf-8")))


def c(v, spec=".2f"):
    if v in ("", None):
        return "—"
    return format(float(v), spec).replace(".", ",")


def pct(v, spec=".1f"):
    return "—" if v in ("", None) else c(float(v) * 100, spec) + " %"


def summary_table(rows, task):
    out = ["| конфигурация | окон | надёжных | покрытие | MAE | грубые промахи | смещение | пределы согласия | медиана эталона | MAE по всем окнам |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["task"] != task:
            continue
        out.append(f"| {r['config']} | {r['n_windows']} | {r['n_reliable']} | {pct(r['coverage'])} | {c(r['mae'])} | "
                   f"{pct(r['gross_rate'])} | {c(r['ba_bias'])} | [{c(r['ba_loa_low'])}; {c(r['ba_loa_high'])}] | "
                   f"{c(r['ref_bpm_median'], '.1f')} | {c(r['mae_all_ref_windows'])} |")
    return "\n".join(out)


def main() -> None:
    meta = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    dc, lag, sync = rd("data_check"), rd("record_lag"), rd("sync_lag")
    rq, summ, summ_ns = rd("ref_quality"), rd("summary"), rd("summary", "no_shift")
    summ_id, trunc, extra = rd("summary_lag_identified"), rd("summary_truncated"), rd("summary_extra_s12")
    pairs, strata, curve, perrec = rd("pairwise"), rd("motion_strata"), rd("purity_curve"), rd("per_record")
    t1 = {n: rd(n, "shift_from_T1") for n in ("summary", "per_record", "motion_strata", "purity_curve", "record_lag")}
    diag = rd("hr_series_lag_diagnostic")
    cache_chk = json.loads((OUT / "cache_check_ubfc_rppg.json").read_text(encoding="utf-8"))
    protocol = (OUT / "protocol.md").read_text(encoding="utf-8")

    dc_rows = "\n".join(f"| {r['subject']} | {r['task']} | {r['group']} | {r['n_frames_decoded']} | {c(r['video_s'], '.1f')} | "
                        f"{c(r['bvp_s'], '.1f')} | {'да' if r['truncated'] == '1' else ''} | {r['set']} |" for r in dc)
    win_full = defaultdict(set)
    n_win = defaultdict(int)
    for r in rq:
        n_win[(r["subject"], r["task"])] += 1
    for (s, t), n in n_win.items():
        if s in FULL:
            win_full[n].add(f"{s} {t}")
    trunc_win = ", ".join(f"{s} T1: {n_win[(s, 'T1')]} окон при {c(next(r['video_s'] for r in dc if r['subject'] == s), '.1f')} с"
                          for s in ("s6", "s7", "s8", "s9"))

    rel = defaultdict(lambda: [0, 0])
    for r in rq:
        if r["subject"] in FULL:
            rel[r["task"]][0] += 1
            rel[r["task"]][1] += r["ref_reliable"] == "1"
    rel_rows = "\n".join(f"| {t} | {rel[t][0]} | {rel[t][0] - rel[t][1]} | {c((rel[t][0] - rel[t][1]) / rel[t][0] * 100, '.1f')} % |" for t in TASKS)

    sync_rows = "\n".join(f"| {r['subject']} | {c(r['lag_s'], '.3f')} | {c(r['corr'], '.3f')} | {'да' if abs(float(r['lag_s'])) > 1 else ''} |" for r in sync)
    lag_rows = "\n".join(f"| {r['subject']} | {r['task']} | {r['set']} | {c(r['lag_s'])} | {c(r['r_at_lag'], '.3f')} | "
                         f"{'да' if r['lag_identified'] == '1' else 'нет'} | {c(r['applied_shift_s'])} |" for r in lag)
    diag_rows = "\n".join(f"| {r['subject']} | {r['task']} | {c(r['best_shift_s'], '.1f')} | {c(r['mae_at_0'])} → {c(r['mae_at_best'])} | "
                          f"{c(r['r_at_0'])} → {c(r['r_at_best'])} |" for r in diag)
    n_ident = sum(1 for r in lag if r["set"] == "full" and r["lag_identified"] == "1")

    pair_rows = "\n".join(f"| {r['config_a']} − {r['config_b']} | {r['task']} | {c(r['median_mae_diff_a_minus_b'], '.3f')} | "
                          f"{r['n_a_better']} из {r['n_records']} |" for r in pairs)

    def strata_table(task, rows=None):
        out = ["| конфигурация | терциль 1 (мало движения) | терциль 2 | терциль 3 (много) |", "|---|---|---|---|"]
        for cf in CONFIGS:
            cells = []
            for q in ("1", "2", "3"):
                r = next(x for x in (rows or strata) if x["config"] == cf and x["task"] == task and x["motion_tertile"] == q)
                cells.append(f"{c(r['mae'])} / {pct(r['gross_rate'])} (n={r['n_windows']})")
            out.append(f"| {cf} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    def curve_table(task, rows=None):
        out = ["| порог | покрытие | MAE по окнам | MAE по записям (записей) | грубые промахи |", "|---|---|---|---|---|"]
        for r in rows or curve:
            if r["task"] == task:
                out.append(f"| {'выкл.' if r['threshold'] == '' else c(r['threshold'], '.2f')} | {pct(r['coverage'])} | "
                           f"{c(r['mae'])} | {c(r['mae_by_records'])} ({r['n_records_with_output']}) | {pct(r['gross_rate'])} |")
        return "\n".join(out)

    def per_subject(cf, rows=None):
        out = ["| участник | T1 | T2 | T3 |", "|---|---|---|---|"]
        for s in ["s1", "s2", "s3", "s4", "s5", "s10", "s11"]:
            cells = [c(next((r["mae"] for r in (rows or perrec) if r["config"] == cf and r["subject"] == s and r["task"] == t), "")) for t in TASKS]
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    edges = meta["motion_tertile_edges"]
    text = f"""# Внешняя проверка на UBFC-Phys

Без выводов о том, какая конфигурация лучше, — только числа и факт проверок. Протокол, включая
правило надёжности эталона, записан до любых сравнений с видео — `protocol.md`, приведён ниже
целиком.

{protocol.replace('# Протокол проверки на UBFC-Phys (записан до сравнений с видео)', '## Протокол (записан до сравнений с видео)')}

## Проверки приёмки

1. **Регрессия:** текущее умолчание на G0 UBFC-rPPG — MAE по записям
   {c(meta['regression_ubfc_rppg_g0']['mae_by_records'], '.4f')}, покрытие
   {c(meta['regression_ubfc_rppg_g0']['coverage_mean'], '.4f')}: совпадает с точкой сверки PRV (1,1970 / 0,9921).
2. **Число окон:** у всех 21 полного видео — {', '.join(f'{n} окон' for n in sorted(win_full))}. Обрезанные:
   {trunc_win} — окон столько, сколько помещается до последнего реального кадра.
3. **Правило надёжности эталона** записано в `protocol.md` раньше результатов; не менялось.
4. **Ни одного параметра, подобранного на UBFC-Phys.** Конфигурации, правило надёжности, пороги
   чистоты, определение движения заданы до прогона. Сдвиг BVP (поправка 1) — по решению заказчика
   после диагностики, его метод и порог зафиксированы до сравнения конфигураций и не используют ни
   одну из них.
5. **Воспроизведение одной командой из кэша:**
   `PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py eval` (и `eval --no-shift`).
   Кэш в формате UBFC-rPPG; на трёх записях UBFC-rPPG оценки из кэша совпали с прямым прогоном
   по видео точно: наибольшее расхождение {max(x['max_abs_bpm_diff'] for x in cache_chk):g} уд/мин на всех трёх.

## Проверка данных

Совпадает с разделом 1 задания, с одним дополнением: в папке есть **s12** — полный (три видео
по 180 с, группа ctrl), в задании не упомянут. Основные выводы — по 21 видео задания; s12 — отдельно.
Число кадров — по фактически декодированным: у полных видео 6325 (в заголовке 6325–6326).

| участник | задание | группа | кадров | видео, с | BVP, с | обрезано | набор |
|---|---|---|---|---|---|---|---|
{dc_rows}

## Синхронизация

**Предписанная проверка** (взаимная корреляция сигнала видео и BVP на T1, ±3 с):

| участник | задержка, с | корреляция | |задержка| > 1 с |
|---|---|---|---|
{sync_rows}

Эта проверка не видит сдвигов больше 3 с, а квазипериодические сигналы дают пики корреляции через
каждый удар. Диагностика по рядам пульса (`hr_series_lag_diagnostic.csv`, сдвиг ±15 с по оценке
видео — только для диагностики) показала сдвиги на секунды:

| участник | задание | лучший сдвиг, с | MAE при 0 → при лучшем | r при 0 → при лучшем |
|---|---|---|---|---|
{diag_rows}

**Сдвиг, применённый в основных таблицах** (поправка 1: мгновенный пульс MSPTD видео против BVP,
независимо от конфигураций, принимается при r ≥ 0,5). Определён у {n_ident} из 21 полного видео:

| участник | задание | набор | сдвиг, с | r | определён | применён, с |
|---|---|---|---|---|---|---|
{lag_rows}

На T2 и T3 видео отслеживает пульс слабо, корреляция почти нигде не достигает порога, поэтому там
сдвиг в основном не применён и основные таблицы для T2/T3 близки к таблицам без сдвига (`no_shift/`).

## Надёжность эталона

Окна, где две независимые оценки пульса по BVP расходятся больше чем на 5 уд/мин (21 полное видео):

| задание | окон | ненадёжных | доля |
|---|---|---|---|
{rel_rows}

## Сводка по заданиям (основная: со сдвигом, окна с надёжным эталоном)

MAE — среднее по записям участник × задание; грубый промах — |оценка − эталон| > 5 уд/мин;
смещение и пределы согласия Бланда — Альтмана — по опубликованным окнам с надёжным эталоном;
«MAE по всем окнам» — по опубликованным окнам без учёта надёжности эталона.

**T1 — покой**

{summary_table(summ, 'T1')}

**T2 — речь**

{summary_table(summ, 'T2')}

**T3 — арифметика**

{summary_table(summ, 'T3')}

### Без сдвига (`no_shift/`)

| конфигурация | T1 MAE | T2 MAE | T3 MAE |
|---|---|---|---|
""" + "\n".join(f"| {cf} | " + " | ".join(c(next(r['mae'] for r in summ_ns if r['config'] == cf and r['task'] == t)) for t in TASKS) + " |" for cf in CONFIGS) + f"""

### Только записи с определённым сдвигом

| конфигурация | задание | записей окон | MAE | грубые промахи |
|---|---|---|---|---|
""" + "\n".join(f"| {r['config']} | {r['task']} | {r['n_windows']} | {c(r['mae'])} | {pct(r['gross_rate'])} |" for r in summ_id) + f"""

### Сдвиг T1 перенесён на T2/T3 (`shift_from_T1/`)

Сдвиг, найденный на T1, применён к T2 и T3 того же участника, если у записи свой сдвиг не
определился; на T2/T3 сдвиг не подбирался. Прежние таблицы не изменены.

| запись | свой сдвиг, с | r | применён, с | источник |
|---|---|---|---|---|
""" + "\n".join(f"| {r['subject']} {r['task']} | {c(r['lag_s'])} | {c(r['r_at_lag'])} | {c(r['applied_shift_s'])} | {r['shift_source']} |"
               for r in t1['record_lag'] if r['subject'] in ("s1", "s2", "s3", "s4", "s5", "s10", "s11")) + f"""

У s2 определились оба собственных сдвига: """ + "; ".join(
    f"{r['task']} {c(r['lag_s'])} с (r {c(r['r_at_lag'])})" for r in t1['record_lag'] if r['subject'] == 's2' and r['task'] in ("T1", "T2")) + """.
У T2 применён свой сдвиг, у T3 — сдвиг T1.

| конфигурация | T1 MAE | T2 MAE | T3 MAE |
|---|---|---|---|
""" + "\n".join(f"| {cf} | " + " | ".join(c(next(r['mae'] for r in t1['summary'] if r['config'] == cf and r['task'] == t)) for t in TASKS) + " |" for cf in CONFIGS) + f"""

{summary_table(t1['summary'], 'T2')}

{summary_table(t1['summary'], 'T3')}

MAE по участникам (fft_spectral_mean):

{per_subject('fft_spectral_mean', t1['per_record'])}

Движение, T2: MAE / грубые промахи

{strata_table('T2', t1['motion_strata'])}

Движение, T3:

{strata_table('T3', t1['motion_strata'])}

Кривая порога чистоты, T2:

{curve_table('T2', t1['purity_curve'])}

Кривая порога чистоты, T3:

{curve_table('T3', t1['purity_curve'])}

### MAE по участникам, текущее умолчание (fft_spectral_mean)

{per_subject('fft_spectral_mean')}

Таблицы для остальных конфигураций — `per_record.csv`.

### Парное сравнение по записям (7 участников)

Медиана разности MAE «a − b» по записям участник × задание; отрицательная — у a меньше.

| пара | задание | медиана разности, уд/мин | у a меньше MAE |
|---|---|---|---|
{pair_rows}

## Терцили движения

Движение окна — медиана межкадрового смещения опорных точек, делённого на межзрачковое
расстояние. Границы терцилей по всем окнам 21 полного видео (до сравнения ошибок):
{c(edges[0], '.5f')} и {c(edges[1], '.5f')}. В ячейке — MAE / доля грубых промахов по опубликованным
окнам с надёжным эталоном; n — все окна терцили.

**T1**

{strata_table('T1')}

**T2**

{strata_table('T2')}

**T3**

{strata_table('T3')}

## Кривые чистоты спектра (текущее умолчание)

Те же 15 абсолютных порогов, что на UBFC-rPPG. Покрытие — доля опубликованных окон; MAE и грубые
промахи — по опубликованным окнам с надёжным эталоном (пул по окнам). Порог не выбирается.

**T1**

{curve_table('T1')}

**T2**

{curve_table('T2')}

**T3**

{curve_table('T3')}

## Отдельно: обрезанные записи s6–s9 (только T1) и s12

| конфигурация | s6–s9 T1: MAE | грубые промахи | покрытие |
|---|---|---|---|
""" + "\n".join(f"| {r['config']} | {c(r['mae'])} | {pct(r['gross_rate'])} | {pct(r['coverage'])} |" for r in trunc) + """

| конфигурация | задание | s12: MAE | грубые промахи | покрытие |
|---|---|---|---|---|
""" + "\n".join(f"| {r['config']} | {r['task']} | {c(r['mae'])} | {pct(r['gross_rate'])} | {pct(r['coverage'])} |" for r in extra) + f"""

## Выгрузки и воспроизведение

`data_check.csv`, `ref_quality.csv`, `sync_lag.csv`, `windows.csv`, `summary.csv`, `per_record.csv`,
`motion_strata.csv`, `purity_curve.csv`, `run_meta.json` — как в задании; сверх задания —
`record_lag.csv` (сдвиг BVP по записям), `hr_series_lag_diagnostic.csv`, `summary_lag_identified.csv`,
`summary_truncated.csv`, `summary_extra_s12.csv`, `pairwise.csv`, `cache_check_ubfc_rppg.json`,
каталоги `no_shift/` (те же таблицы без сдвига) и `shift_from_T1/` (сдвиг T1 перенесён на T2/T3). В `windows.csv` добавлена колонка `set`.
Видео и кадры никуда не копировались.

```bash
PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py build          # кэш по видео, ~2 ч
PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py eval           # все таблицы из кэша
PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py eval --no-shift
PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py eval --shift-from-t1
PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys_report.py
```

Код: коммит {meta['git_commit'][:8]}{' с незакоммиченными изменениями' if meta['git_dirty'] else ''};
хеш исходников и пресетов — в `run_meta.json`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(f"README.md записан, {len(text)} символов")


if __name__ == "__main__":
    main()
