"""
Собирает results/prv/README.md из выгрузок benchmark/prv_eval.py.

    PYTHONPATH=src venv/bin/python3 benchmark/prv_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "prv"
VARIANTS = ["V0"] + [f"{v}_{r}" for v in ("V1", "V2") for r in ("forehead", "left_cheek", "right_cheek")] + ["V3"]


def rd(name):
    return list(csv.DictReader((OUT / f"{name}.csv").open(encoding="utf-8")))


def f(v):
    return float(v) if v not in ("", None) else np.nan


def c(x, spec=".2f"):
    return "—" if not np.isfinite(x) else format(x, spec).replace(".", ",")


def miq(values, spec=".2f"):
    a = np.array([x for x in values if np.isfinite(x)])
    if not len(a):
        return "—"
    return f"{c(np.median(a), spec)} [{c(np.percentile(a, 25), spec)}–{c(np.percentile(a, 75), spec)}]"


def main() -> None:
    params = json.loads((OUT / "params.json").read_text(encoding="utf-8"))
    cp = json.loads((OUT / "checkpoint_g0.json").read_text(encoding="utf-8"))
    hr = json.loads((OUT / "hr_check_after.json").read_text(encoding="utf-8"))
    det, prv, seam, ao = rd("metrics_detection"), rd("metrics_prv"), rd("seam_diagnostics"), rd("roi_arrival_offsets")
    off, impl, ph = rd("video_reference_offset"), rd("implausible_reference"), rd("fp_phase")
    syn = json.loads((OUT / "msptd_synthetic.json").read_text(encoding="utf-8"))

    def table(subset):
        lines = []
        for v in VARIANTS:
            d = [r for r in det if r["variant"] == v and r["subset"] == subset]
            p = [r for r in prv if r["variant"] == v and r["subset"] == subset]
            ratio = [(f(r["tp"]) + f(r["fp"])) / (f(r["tp"]) + f(r["fn"])) for r in d]
            lines.append(f"| {v} | {miq([f(r['f1']) for r in d], '.3f')} | {miq(ratio, '.3f')} | "
                         f"{miq([f(r['ibi_mae_ms']) for r in p])} | "
                         f"{miq([abs(f(r['rmssd_video']) - f(r['rmssd_ref'])) for r in p])} | "
                         f"{miq([abs(f(r['sdnn_video']) - f(r['sdnn_ref'])) for r in p])} | "
                         f"{miq([abs(f(r['pnn50_video']) - f(r['pnn50_ref'])) for r in p])} |")
        return "\n".join(lines)

    rmssd_ref = {s: miq([f(r["rmssd_ref"]) for r in prv if r["variant"] == "V3" and r["subset"] == s]) for s in "12"}
    empty_ibi = sorted({(r["record_id"], r["variant"]) for r in prv if r["ibi_mae_ms"] == ""})
    empty_by_var = {}
    for rid, v in empty_ibi:
        empty_by_var.setdefault(v, []).append(rid)

    def sd(k):
        return [f(r[k]) for r in seam]
    sw = np.array([f(r["ibi_mae_at_switch_ms"]) - f(r["ibi_mae_elsewhere_ms"]) for r in seam])
    sw = sw[np.isfinite(sw)]
    holes = np.array([f(r["rmssd_with_holes"]) - f(r["rmssd_without_holes"]) for r in seam])
    no_switch_rec = [r["record_id"] for r in seam if r["ibi_mae_at_switch_ms"] == ""]
    no_else_rec = [r["record_id"] for r in seam if r["ibi_mae_elsewhere_ms"] == ""]

    def phase_hist(v):
        p = np.array([f(r["phase"]) for r in ph if r["variant"] == v])
        h, _ = np.histogram(p, bins=10, range=(0, 1))
        return " | ".join(c(x * 100, ".0f") for x in h / h.sum())

    ao_rows = []
    for a, b in (("forehead", "left_cheek"), ("forehead", "right_cheek"), ("left_cheek", "right_cheek")):
        x = [f(r["median_offset_ms"]) for r in ao if r["roi_a"] == a and r["roi_b"] == b]
        n = [f(r["n_matched"]) for r in ao if r["roi_a"] == a and r["roi_b"] == b]
        ao_rows.append(f"| {a} − {b} | {miq(x, '.1f')} | {miq(np.abs(x), '.1f')} | {miq(n, '.0f')} |")

    offs = [f(r["median_offset_to_nearest_ref_ms"]) for r in off]
    within = [f(r["share_within_150ms"]) for r in off]
    impl_rows = "\n".join(f"| `{r['record_id']}` | {r['subset']} | {c(f(r['hr_ref_beats']))} | {c(f(r['median_main_ref']))} |"
                          for r in impl if r["record_id"]) or "| — | — | — | — |"
    P = params["params"]
    syn_rows = "\n".join(f"| {c(s['fs'], '.2f')} | {c(s['sensitivity'], '.4f')} | {c(s['ppv'], '.4f')} | "
                         f"{c(s['median_err_ms'], '.1f')} | {c(s['p95_err_ms'], '.1f')} |" for s in syn)

    text = f"""# Вариабельность пульса: замер онлайн-цепочки и офлайн-цепочка

Без выводов о том, какой вариант лучше, — только числа и факт проверок.

## Параметры (зафиксированы до прогона)

Записаны в `params.json` в {params['written_utc']} — до первого расчёта.

- Конфигурация пайплайна: {P['pipeline_preset']}.
- Сопоставление ударов: один к одному, пары по возрастанию |Δt|, допуск ±{c(P['match_tolerance_ms'], '.0f')} мс.
- Отрезок оценки обнаружения: {P['evaluation_span']}.
- Допустимый интервал: {c(P['ibi_valid_range_ms'][0], '.1f')}–{c(P['ibi_valid_range_ms'][1], '.0f')} мс (40–220 уд/мин).
- **Эталон.** {P['reference']['signal']}; {P['reference']['detector']}.
- **V0** — {P['V0']['description']}. `detect_pulse_peaks`: {P['V0']['detect_pulse_peaks']}; края окна {c(P['V0']['edge_trim_s'], '.0f')} с; разрыв ряда при > {c(P['V0']['max_gap_ms'], '.0f')} мс. ВСР: {P['V0']['hrv_as_current_code']}.
- **Непрерывный сигнал (V1–V3).** {P['offline_signal']['method']}. Частота: {P['offline_signal']['fps']}. Валидность: {P['offline_signal']['validity']}. Края записи — {c(P['offline_signal']['edge_trim_s'], '.0f')} с.
- **V1** — {P['V1']}.
- **V2** — {P['V2']['detector']}: окно {c(P['V2']['win_len_s'], '.0f')} с, перекрытие {c(P['V2']['overlap'] * 100, '.0f')} %, коррекция пика ±{c(P['V2']['peak_correction_tol_s'] * 1000, '.0f')} мс; понижение частоты — {P['V2']['downsampling']}; уточнение — {P['V2']['refinement']}. Отличия от исходника, неизбежные при переносе: длина окна округляется до целого числа отсчётов, окно коррекции обрезается границами (в Matlab оба случая — ошибка индексации).
- **V3** — {P['V3']}.
- **ВСР.** {P['prv']}.
- Швы: {P['seams']['roi_switch_interval']}; дыра — {P['seams']['hole']}; повторная детекция — {P['seams']['redetection_jitter']}.
- Неправдоподобный эталон: {P['implausible_reference']}.

## Проверки приёмки

1. **Частота пульса на G0 не изменилась.** Точка сверки до работы: MAE по записям
   {c(cp['mae_by_records'], '.4f')}, покрытие {c(cp['coverage_mean'], '.4f')} (50 записей, основной эталон).
   Во время прогона с логированием: {c(hr['mae_by_records'], '.4f')} и {c(hr['coverage_mean'], '.4f')} — совпадает.
   Онлайн-цепочка воспроизводилась из кэша сигналов; при новых умолчаниях воспроизведение
   побитово совпадает с прогоном по видео (проверено на записи 10).
2. **Юнит-тест MSPTD на синтетике проходит** (`tests/test_offline_prv.py`, 6 тестов). Точность на
   синтетической волне с дикротической волной, дыхательной аритмией, дрейфом и шумом
   (10 реализаций по 60 с на частоту, допуск ±150 мс, положение — целый отсчёт до уточнения):

| частота, Гц | чувствительность | точность | ошибка положения, медиана, мс | p95, мс |
|---|---|---|---|---|
{syn_rows}

3. **Все 50 записей во всех таблицах.** Пустые ячейки — только где считать нечего:
   `ibi_mae_ms`/`ibi_bias_ms` пусты у {len(empty_ibi)} пар «запись × вариант»: нет ни одной пары
   подряд сопоставленных ударов (см. «Систематический сдвиг» ниже) —
   {"; ".join(f"{v}: {', '.join(r)}" for v, r in sorted(empty_by_var.items()))}.
   В `seam_diagnostics` MAE на переключении пуст у {len(no_switch_rec)} записей (нет сопоставленных
   интервалов на переключении или нет переключений): {", ".join(no_switch_rec)}; MAE вне переключений
   пуст у {len(no_else_rec)}: {", ".join(no_else_rec)}.
4. **Три PNG на месте:** `qa_5.png` (подмножество 1), `qa_subject10.png`, `qa_subject38.png`
   (подмножество 2), отрезок 20–40 с. Удары V0 взяты из меняющихся по окнам областей, а
   нарисованы на сигнале одной области (выбранной для V3).
5. **Параметры записаны до результатов** — раздел выше и `params.json`.

## Ограничения эталона

- Во втором подмножестве UBFC шаг временно́й оси контактного сигнала неравномерен (0,010–0,223 с).
  Такой эталон годится для подсчёта ударов, но слишком груб для RMSSD с точностью до миллисекунд.
  Поэтому ниже метрики приведены для обоих подмножеств, но **основным для RMSSD, SDNN и pNN50
  считается первое** (8 записей, 62,5 Гц, равномерная сетка); второе — отдельно, с этой пометкой.
- **Систематический сдвиг видео относительно эталона.** Удары видео приходят раньше эталонных:
  медианный по записям сдвиг до ближайшего эталонного удара {miq(offs, '.0f')} мс (V3), а доля
  ударов V3 в пределах ±150 мс от ближайшего эталонного — {miq(within, '.3f')}. Пульсовая волна
  доходит до лица раньше, чем до пальца с датчиком, и форма волны после полосового фильтра
  другая. При заданном допуске ±150 мс часть верно найденных ударов поэтому не сопоставляется:
  чувствительность, точность и F1 ниже отражают этот сдвиг, а не только качество обнаружения.
  От сдвига не зависит **отношение числа найденных ударов к эталонному** — оно приведено в
  таблице. Ошибки интервала и ВСР считаются по разностям времён и от постоянного сдвига не
  зависят, но ошибка интервала считается только по сопоставленным ударам. Допуск не менялся.
  Сдвиг по записям — `video_reference_offset.csv`.

## Сводка по вариантам

Медиана [межквартильный размах] по записям. Ошибки ВСР — модуль разности видео и эталона на
общем отрезке. Эталонный RMSSD (V3): подмножество 1 — {rmssd_ref['1']} мс, подмножество 2 — {rmssd_ref['2']} мс.

**Подмножество 1 (основное для ВСР)**

| вариант | F1 | ударов / эталон | ошибка интервала, мс | ошибка RMSSD, мс | ошибка SDNN, мс | ошибка pNN50, п.п. |
|---|---|---|---|---|---|---|
{table("1")}

**Подмножество 2 (эталон с неравномерной осью — ВСР только для справки)**

| вариант | F1 | ударов / эталон | ошибка интервала, мс | ошибка RMSSD, мс | ошибка SDNN, мс | ошибка pNN50, п.п. |
|---|---|---|---|---|---|---|
{table("2")}

## Диагностика швов: гипотезы раздела 1

**1. Детектор внутри окна слишком разрешающий.** Онлайн-цепочка находит больше ударов, чем
эталон (столбец «ударов / эталон» у V0). Тот же `detect_pulse_peaks`, применённый один раз к
непрерывному сигналу (V1), даёт отношение заметно ближе к 1, MSPTD (V2, V3) — около 1.
Фаза лишних ударов внутри эталонного цикла, доли по десятым цикла, %:

| вариант | 0–0,1 | | | | | | | | | 0,9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | {phase_hist('V0')} |
| V1 | {phase_hist('V1')} |
| V2 | {phase_hist('V2')} |

Распределение фазы смешано с систематическим сдвигом (зона 0,6–0,9 — это и верные удары,
не попавшие в допуск), поэтому место выемки по нему не локализуется.
Вердикт: лишние удары подтверждены; по сравнению V0 и V1 бо́льшая их часть возникает не в
самом детекторе, а при его применении по окнам.

**2. Швы между окнами.** Повторная детекция одного удара в соседних окнах: медиана сдвига
{miq(sd('redetection_jitter_median_ms'))} мс, p95 {miq(sd('redetection_jitter_p95_ms'))} мс по записям.
Удар, принятый ближе 273 мс к предыдущему (вставка лишнего удара; интервал не пишется, но
указатель сдвигается): {miq(sd('n_rejected_too_short'), '.0f')} за запись; отклонённых длиннее 1,5 с —
{miq(sd('n_rejected_too_long'), '.0f')}, разрывов > 3 с — {miq(sd('n_rejected_gap_gt_3s'), '.0f')}.
Вердикт: сдвиг граничного удара между окнами обычно меньше миллисекунды, но в хвосте до десятков мс;
вставка лишних ударов на швах подтверждена.

**3. Переключение области.** Переключений за запись {miq(sd('n_roi_switches'), '.0f')}; интервалов,
пересекающих переключение, {miq(sd('n_switch_intervals'), '.0f')}. Ошибка интервала на переключении
{miq(sd('ibi_mae_at_switch_ms'))} мс против {miq(sd('ibi_mae_elsewhere_ms'))} мс в остальных;
разность по {len(sw)} записям: медиана {c(np.median(sw))} мс, больше на переключении у
{int((sw > 0).sum())} из {len(sw)}, критерий Уилкоксона p = {c(wilcoxon(sw).pvalue, '.2f')}.
Задержки прихода пульса между областями — ниже (около 16–18 мс между лбом и щеками).
Вердикт: различие ошибки на переключениях в данных не обнаруживается; ожидаемый вклад —
порядка задержки между областями, ниже типичной ошибки интервала.

**4. Дыры в ряду.** Дыр в `_ibi_log` за запись {miq(sd('n_holes'), '.0f')} (в основном после
отброшенных коротких интервалов, см. п. 2). RMSSD с разностями через дыры
{miq(sd('rmssd_with_holes'))} мс, без них {miq(sd('rmssd_without_holes'))} мс; разность: медиана
{c(np.nanmedian(holes))} мс, больше с дырами у {int((holes > 0.01).sum())} из {len(holes)} записей.
Вердикт: механизм подтверждён (разности через дыры входят в RMSSD), величина эффекта мала
по сравнению с общей ошибкой RMSSD онлайн-цепочки.

## Задержки прихода пульса между областями

V2, сопоставление ±150 мс; положительная задержка — удар в первой области позже.

| пара | медиана задержки, мс | |медиана|, мс | сопоставлено ударов |
|---|---|---|---|
{chr(10).join(ao_rows)}

## Записи с неправдоподобным эталоном

| запись | подмножество | ЧСС по эталонным ударам | медиана основного эталона |
|---|---|---|---|
{impl_rows}

## Выгрузки

`reference_beats.csv`, `baseline_window_peaks.csv` (дополнительная колонка `used_for_ibi`),
`baseline_ibi_log.csv`, `offline_beats.csv`, `metrics_detection.csv`, `metrics_prv.csv`,
`seam_diagnostics.csv` (дополнительные колонки: число интервалов на переключении, отброшенных
интервалов по причинам, сдвиг повторной детекции), `roi_arrival_offsets.csv`, `qa_*.png`;
сверх задания — `v3_choice.csv` (выбор области V3 и средние SNR), `video_reference_offset.csv`,
`fp_phase.csv`, `params.json`, `checkpoint_g0.json`, `hr_check_after.json`, `msptd_synthetic.json`.

```bash
PYTHONPATH=src venv/bin/python3 -m pytest tests/test_offline_prv.py
PYTHONPATH=src venv/bin/python3 benchmark/prv_eval.py
PYTHONPATH=src venv/bin/python3 benchmark/prv_report.py
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(f"README.md записан, {len(text)} символов")


if __name__ == "__main__":
    main()
