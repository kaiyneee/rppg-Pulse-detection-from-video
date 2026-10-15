"""
Задание PRV: замер текущей онлайн-цепочки вариабельности (V0) и офлайн-цепочки
на непрерывном сигнале (V1, V2, V3) против эталонных ударов контактной волны.

Онлайн-цепочка не меняется: она воспроизводится из кэша сигналов
(benchmark/trace_cache.py; воспроизведение побитово совпадает с прогоном по
видео) при конфигурации по умолчанию, а её промежуточные величины снимаются
обёртками вокруг RPPGPipeline._update_ibi_log и assess_quality.

Все параметры — в PARAMS; они записываются в results/prv/params.json ДО
первого расчёта.

    PYTHONPATH=src venv/bin/python3 benchmark/prv_eval.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import experiments as ex  # noqa: E402
import reference_full10s as R  # noqa: E402
import trace_cache as tc  # noqa: E402
import rppg.pipeline as pipeline_mod  # noqa: E402
from rppg.config import make_pipeline_config  # noqa: E402
from rppg.hrv.features import ectopic_artifact_mask, hrv_time_domain, refine_peaks_subsample  # noqa: E402
from rppg.hrv.offline_prv import (  # noqa: E402
    MAX_GAP_FRAMES, MSPTD_OVERLAP, MSPTD_TOL_S, MSPTD_WIN_LEN_S, MAX_IBI_MS, MIN_IBI_MS,
    beats_v1, beats_v2, continuous_roi_signal, ibi_series_from_beats, index_to_time_ms,
    match_beats, msptd_beat_detector, prv_metrics, IBISeries,
)

OUT = REPO_ROOT / "results" / "prv"
ROIS = ("forehead", "left_cheek", "right_cheek")
QA_RECORDS = ("5", "subject10", "subject38")   # одна запись подмножества 1, две — подмножества 2
QA_SEGMENT_S = (20.0, 40.0)

PARAMS = {
    "pipeline_preset": "fft_spectral_mean (умолчание)",
    "match_tolerance_ms": 150.0,
    "evaluation_span": "[первый кадр + 1 с, последний кадр - 1 с] записи, одинаково для всех вариантов",
    "ibi_valid_range_ms": [MIN_IBI_MS, MAX_IBI_MS],
    "reference": {
        "signal": "контактная волна, равномерная сетка: N точек между первой и последней меткой записи, "
                  "линейная интерполяция (как в bpm_reference)",
        "detector": "MSPTD (V2) на волне без фильтрации, затем параболическое уточнение (refine_peaks_subsample)",
    },
    "V0": {
        "description": "онлайн-цепочка как есть: пики окна -> _update_ibi_log; удар учтён, если позже последнего учтённого",
        "detect_pulse_peaks": "distance = fps*60/220, prominence = 0.25*std окна, свёртка чередования при отношении >= 1.6",
        "edge_trim_s": 1.0, "max_gap_ms": 3000.0,
        "hrv_as_current_code": "hrv_time_domain с ectopic_artifact_mask(0.4) по ВСЕМУ ряду _ibi_log записи "
                               "(на UBFC онлайн-цепочка ВСР не выдаёт: нужно >= 120 с накопления)",
    },
    "offline_signal": {
        "method": "POS, окно 1.6 с, реконструкция наложением; preprocess_signal с cfg.filt (детренд Тарвайнена "
                  "lambda=3542, Баттерворт 4-го порядка 0.7-4.0 Гц нулевой фазы, z-нормировка) на всей записи",
        "fps": "(N-1)/span по меткам кадров записи",
        "validity": "лицо найдено и ROI валиден по extract_rois (без онлайн-проверки стабильности числа пикселей); "
                    f"пропуски заполняются fill_gaps; удары в провалах длиннее {MAX_GAP_FRAMES} кадров отбрасываются",
        "edge_trim_s": 1.0,
    },
    "V1": "detect_pulse_peaks + refine_peaks_subsample на непрерывном сигнале каждой области",
    "V2": {"detector": "MSPTD, перенос ppg-beats msptd_beat_detector.m (коммит a1f3bef)",
           "win_len_s": MSPTD_WIN_LEN_S, "overlap": MSPTD_OVERLAP, "peak_correction_tol_s": MSPTD_TOL_S,
           "downsampling": "нет (do_ds = 0 в исходнике)",
           "refinement": "refine_peaks_subsample (то же, что у V1)"},
    "V3": "V2 на одной области на всю запись: argmax среднего по окнам SNR области "
          "(snr_by_roi_db онлайн-цепочки, конечные значения)",
    "prv": "RMSSD и pNN50 — по разностям смежных интервалов (оба допустимы, общий удар); SDNN — по допустимым "
           "интервалам; эктопический фильтр не применяется (кроме V0, см. выше); видео и эталон на общем отрезке "
           "[max(первых ударов), min(последних ударов)]",
    "seams": {
        "roi_switch_interval": "интервал _ibi_log, удары которого учтены в окнах с разными областями",
        "hole": "элемент _ibi_log, у которого pt - gap не совпадает с pt предыдущего (допуск 1 мс)",
        "redetection_jitter": "|Δt| одного удара в соседних окнах (сопоставление ±100 мс)",
    },
    "implausible_reference": "|ЧСС по эталонным ударам - медиана основного эталона частоты записи| > 5 уд/мин",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(x, spec=".4f"):
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else format(float(x), spec)
    return x


# --------------------------------------------------------------------------
# Эталон
# --------------------------------------------------------------------------
def reference_beats(trace) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t, x0 = np.asarray(trace.t_ref, float), np.asarray(trace.wave, float)
    tu = np.linspace(t[0], t[-1], len(t))
    x = np.interp(tu, t, x0)
    fs = (len(t) - 1) / (t[-1] - t[0])
    p, _ = msptd_beat_detector(x, fs)
    r = refine_peaks_subsample(x, p)
    return np.interp(r, np.arange(len(tu)), tu) * 1000.0, tu, x, fs


# --------------------------------------------------------------------------
# Часть A: онлайн-цепочка с логированием
# --------------------------------------------------------------------------
def run_online(trace):
    cfg = tc.make_config(trace.fps)
    stash = {"peaks": None, "snr": None}
    log = {"windows": [], "accepted": [], "ibi": [], "rejected": defaultdict(int), "snr": []}
    original_assess = pipeline_mod.assess_quality

    def assess_spy(inputs, qcfg):
        stash["snr"] = dict(inputs.snr_by_roi_db or {})
        return original_assess(inputs, qcfg)

    def setup(pipe):
        original_update = pipe._update_ibi_log

        def update_spy(cross_check, timestamps_ms):
            ts = np.asarray(timestamps_ms, dtype=float)
            times = (np.interp(cross_check.peaks_sub, np.arange(len(ts), dtype=float), ts)
                     if len(cross_check.peaks_sub) else np.array([]))
            used = bool(cross_check.peaks_ok_for_ibi and len(times))
            accepted = []
            if used:   # копия правила дедупликации _update_ibi_log — только для журнала
                last = pipe._last_peak_time_ms
                for pt in times:
                    pt = float(pt)
                    if last is None:
                        accepted.append(pt); last = pt; continue
                    gap = pt - last
                    if gap <= 1e-6:
                        continue
                    accepted.append(pt); last = pt
                    if gap > 3000.0:
                        log["rejected"]["gap_gt_3s"] += 1
                    elif gap < MIN_IBI_MS:
                        log["rejected"]["too_short"] += 1
                    elif gap > MAX_IBI_MS:
                        log["rejected"]["too_long"] += 1
            n_before = len(pipe._ibi_log)
            last_pt_before = pipe._ibi_log[-1][0] if pipe._ibi_log else -np.inf
            original_update(cross_check, timestamps_ms)
            appended = [e for e in list(pipe._ibi_log) if e[0] > last_pt_before]
            stash["peaks"] = (times, used, accepted, appended, n_before)

        pipe._update_ibi_log = update_spy

    def on_result(pipe, res):
        roi = res.debug.best_source if res.debug is not None else None
        widx = len(log["windows"])
        log["windows"].append(roi)
        log["snr"].append(stash["snr"] or {})
        if stash["peaks"] is not None:
            times, used, accepted, appended, _ = stash["peaks"]
            for t in times:
                log.setdefault("window_peaks", []).append((widx, roi, float(t), used))
            for t in accepted:
                log["accepted"].append((float(t), widx, roi))
            log["ibi"].extend((float(p), float(g)) for p, g in appended)
        stash["peaks"], stash["snr"] = None, None

    pipeline_mod.assess_quality = assess_spy
    try:
        rows, _ = tc.replay(trace, cfg, setup=setup, on_result=on_result)
    finally:
        pipeline_mod.assess_quality = original_assess
    log.setdefault("window_peaks", [])
    return rows, log, cfg


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------
def in_span(t, span):
    t = np.asarray(t, float)
    return t[(t >= span[0]) & (t <= span[1])]


def detection(video_ms, ref_ms, tol):
    pairs = match_beats(np.sort(video_ms), np.sort(ref_ms), tol)
    tp = len(pairs); fp = len(video_ms) - tp; fn = len(ref_ms) - tp
    sens = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sens, "ppv": ppv, "f1": f1}, pairs


def ibi_errors(ibi_t_end, ibi_ms, ref_ms, tol):
    """Ошибка интервала: оба удара интервала видео сопоставлены с двумя
    ПОДРЯД идущими эталонными ударами. Возвращает (ошибки, маска)."""
    ref = np.sort(ref_ms)
    starts = np.asarray(ibi_t_end) - np.asarray(ibi_ms)
    beats = np.unique(np.concatenate([starts, ibi_t_end])) if len(ibi_ms) else np.array([])
    pairs = dict(match_beats(beats, ref, tol))
    pos = {round(b, 6): i for i, b in enumerate(beats)}
    errs, mask = [], np.zeros(len(ibi_ms), dtype=bool)
    for k, (s, e) in enumerate(zip(starts, ibi_t_end)):
        i, j = pos.get(round(s, 6)), pos.get(round(e, 6))
        if i in pairs and j in pairs and pairs[j] == pairs[i] + 1:
            errs.append((e - s) - (ref[pairs[j]] - ref[pairs[i]]))
            mask[k] = True
    return np.array(errs), mask


def common_span(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    return (max(np.min(a), np.min(b)), min(np.max(a), np.max(b)))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "params": PARAMS}
    (OUT / "params.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    dev, test = ex.split()
    split_of = {r: "dev" for r in dev} | {r: "test" for r in test}
    main_ref = R.load_reference()
    tol = PARAMS["match_tolerance_ms"]

    out = defaultdict(list)
    hr_check = {"maes": [], "covs": []}
    implausible, qa_data = [], {}

    for rid in sorted(split_of, key=ex._natural_key):
        trace = tc.load_trace(rid)
        subset = trace.subset
        ref_ms, tu, x_ref, fs_ref = reference_beats(trace)
        for k, t in enumerate(ref_ms):
            out["reference_beats"].append({"record_id": rid, "subset": subset, "beat_idx": k, "t_s": fmt(t / 1000, ".4f")})

        rows, log, cfg = run_online(trace)
        # критерий 1: частота пульса не изменилась
        e = [abs(r.bpm - main_ref[(rid, r.timestamp_ms)]) for r in rows
             if r.publishable and (rid, r.timestamp_ms) in main_ref and np.isfinite(r.bpm)]
        hr_check["maes"].append(float(np.mean(e)))
        hr_check["covs"].append(sum(r.publishable for r in rows) / len(rows))

        ts = np.asarray(trace.timestamps_ms, float)
        span = (ts[0] + 1000.0, ts[-1] - 1000.0)
        ref_in = in_span(ref_ms, span)

        # правдоподобие эталона
        ref_series = ibi_series_from_beats(ref_ms)
        hr_ref_beats = 60000.0 / np.mean(ref_series.ibi_ms) if len(ref_series.ibi_ms) else np.nan
        med_main = float(np.median([v for (r_, _t), v in main_ref.items() if r_ == rid]))
        if not np.isfinite(hr_ref_beats) or abs(hr_ref_beats - med_main) > 5:
            implausible.append((rid, subset, hr_ref_beats, med_main))

        # ---- V0 ----
        for widx, roi, t, used in log["window_peaks"]:
            out["baseline_window_peaks"].append({"record_id": rid, "window_idx": widx, "roi": roi,
                                                 "t_s": fmt(t / 1000, ".4f"), "used_for_ibi": fmt(used)})
        ibi_log = log["ibi"]
        adj = [False] + [abs((p - g) - ibi_log[i - 1][0]) <= 1.0 for i, (p, g) in enumerate(ibi_log) if i > 0]
        for (p, g), a in zip(ibi_log, adj):
            out["baseline_ibi_log"].append({"record_id": rid, "pt_ms": fmt(p, ".3f"), "gap_ms": fmt(g, ".3f"),
                                            "adjacent_to_prev": fmt(a)})
        v0_beats = np.array([t for t, _, _ in log["accepted"]])
        variants = {"V0": {"beats": v0_beats,
                           "series": IBISeries(np.array([p for p, _ in ibi_log]), np.array([g for _, g in ibi_log]),
                                               np.array(adj, dtype=bool))}}

        # ---- V1/V2 по областям, V3 ----
        signals = {}
        for roi in ROIS:
            rgb = np.array([f.roi.rgb_by_roi.get(roi) if (f.detected and f.roi is not None and
                            f.roi.rgb_by_roi.get(roi) is not None) else np.zeros(3) for f in trace.frames], float)
            valid = np.array([bool(f.detected and f.roi is not None and f.roi.valid_by_roi.get(roi, False)
                                   and f.roi.rgb_by_roi.get(roi) is not None) for f in trace.frames])
            cs = continuous_roi_signal(ts, rgb, valid, roi, cfg)
            signals[roi] = cs
            bad_spans = []
            if cs.bad.any():
                idx = np.where(cs.bad)[0]
                for run_ in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                    bad_spans.append((ts[run_[0]], ts[run_[-1]]))
            for name, fn in (("V1", beats_v1), ("V2", beats_v2)):
                bt = index_to_time_ms(cs, fn(cs), cfg.hrv.edge_trim_seconds)
                variants[f"{name}_{roi}"] = {"beats": bt, "series": ibi_series_from_beats(bt, bad_spans)}
        snr_mean = {roi: np.nanmean([s.get(roi, np.nan) if np.isfinite(s.get(roi, np.nan)) else np.nan
                                     for s in log["snr"]]) for roi in ROIS}
        v3_roi = max(ROIS, key=lambda r_: -np.inf if not np.isfinite(snr_mean[r_]) else snr_mean[r_])
        variants["V3"] = dict(variants[f"V2_{v3_roi}"])
        out["v3_choice"].append({"record_id": rid, "roi": v3_roi,
                                 **{f"mean_snr_{r_}": fmt(snr_mean[r_], ".3f") for r_ in ROIS}})

        for name, v in variants.items():
            if name.startswith(("V1", "V2")) or name == "V3":
                roi_tag = v3_roi if name == "V3" else name.split("_", 1)[1]
                if name != "V3":
                    for t in v["beats"]:
                        out["offline_beats"].append({"record_id": rid, "roi": roi_tag, "variant": name.split("_")[0],
                                                     "t_s": fmt(t / 1000, ".4f")})
            vb = in_span(v["beats"], span)
            det, _ = detection(vb, ref_in, tol)
            out["metrics_detection"].append({"record_id": rid, "subset": subset, "split": split_of[rid],
                                             "variant": name, **{k: fmt(val) for k, val in det.items()}})
            s = v["series"]
            errs, _ = ibi_errors(s.t_ms, s.ibi_ms, ref_ms, tol)
            cs_span = common_span(v["beats"], ref_ms)
            row = {"record_id": rid, "subset": subset, "split": split_of[rid], "variant": name,
                   "n_ibi": len(s.ibi_ms), "ibi_mae_ms": fmt(np.mean(np.abs(errs)) if len(errs) else np.nan, ".2f"),
                   "ibi_bias_ms": fmt(np.mean(errs) if len(errs) else np.nan, ".2f")}
            if cs_span is not None:
                keep = (s.t_ms - s.ibi_ms >= cs_span[0]) & (s.t_ms <= cs_span[1])
                sv = IBISeries(s.t_ms[keep], s.ibi_ms[keep], s.adjacent[keep])
                if keep.any():
                    sv.adjacent[0] = False
                rr = ibi_series_from_beats(in_span(ref_ms, cs_span))
                if name == "V0":   # как считает текущий код: весь ряд, эктопическая маска
                    hv = hrv_time_domain(sv.ibi_ms, ectopic_artifact_mask(sv.ibi_ms, cfg.hrv.ectopic_max_relative_change))
                    mv = {"rmssd": hv["rmssd_ms"], "sdnn": hv["sdnn_ms"], "pnn50": hv["pnn50_pct"]}
                else:
                    mv = prv_metrics(sv)
                mr = prv_metrics(rr)
                row.update({"rmssd_video": fmt(mv["rmssd"], ".2f"), "rmssd_ref": fmt(mr["rmssd"], ".2f"),
                            "sdnn_video": fmt(mv["sdnn"], ".2f"), "sdnn_ref": fmt(mr["sdnn"], ".2f"),
                            "pnn50_video": fmt(mv["pnn50"], ".2f"), "pnn50_ref": fmt(mr["pnn50"], ".2f")})
            else:
                row.update({k: "" for k in ("rmssd_video", "rmssd_ref", "sdnn_video", "sdnn_ref", "pnn50_video", "pnn50_ref")})
            out["metrics_prv"].append(row)

        # ---- швы V0 ----
        roi_of = {round(t, 6): roi for t, _, roi in log["accepted"]}
        switches = sum(1 for a, b in zip(log["windows"], log["windows"][1:]) if a and b and a != b)
        s0 = variants["V0"]["series"]
        errs0, mask0 = ibi_errors(s0.t_ms, s0.ibi_ms, ref_ms, tol)
        at_switch = np.array([roi_of.get(round(p - g, 6)) != roi_of.get(round(p, 6)) for p, g in ibi_log], dtype=bool)
        err_full = np.full(len(ibi_log), np.nan)
        err_full[mask0] = errs0
        d_all = np.diff(s0.ibi_ms)
        d_adj = d_all[s0.adjacent[1:]] if len(d_all) else d_all
        # повторная детекция одного удара в соседних окнах
        by_w = defaultdict(list)
        for widx, _, t, _u in log["window_peaks"]:
            by_w[widx].append(t)
        jit = []
        for w in sorted(by_w):
            if w - 1 in by_w:
                for i, j in match_beats(np.array(by_w[w]), np.array(by_w[w - 1]), 100.0):
                    jit.append(abs(by_w[w][i] - by_w[w - 1][j]))
        out["seam_diagnostics"].append({
            "record_id": rid, "n_roi_switches": switches,
            "ibi_mae_at_switch_ms": fmt(np.nanmean(np.abs(err_full[at_switch])) if np.isfinite(err_full[at_switch]).any() else np.nan, ".2f"),
            "ibi_mae_elsewhere_ms": fmt(np.nanmean(np.abs(err_full[~at_switch])) if np.isfinite(err_full[~at_switch]).any() else np.nan, ".2f"),
            "n_holes": int(sum(1 for a in adj[1:] if not a)),
            "rmssd_with_holes": fmt(np.sqrt(np.mean(d_all ** 2)) if len(d_all) else np.nan, ".2f"),
            "rmssd_without_holes": fmt(np.sqrt(np.mean(d_adj ** 2)) if len(d_adj) else np.nan, ".2f"),
            "n_switch_intervals": int(at_switch.sum()),
            "n_rejected_too_short": log["rejected"]["too_short"], "n_rejected_too_long": log["rejected"]["too_long"],
            "n_rejected_gap_gt_3s": log["rejected"]["gap_gt_3s"],
            "redetection_jitter_median_ms": fmt(np.median(jit) if jit else np.nan, ".2f"),
            "redetection_jitter_p95_ms": fmt(np.percentile(jit, 95) if jit else np.nan, ".2f"),
        })
        # дикротическая фаза лишних ударов V0 и V1 (гипотеза 1)
        for name in ("V0", f"V1_{v3_roi}", f"V2_{v3_roi}"):
            vb = in_span(variants[name]["beats"], span)
            _, pairs = detection(vb, ref_in, tol)
            matched = {i for i, _ in pairs}
            vb_sorted = np.sort(vb)
            for i, t in enumerate(vb_sorted):
                if i in matched:
                    continue
                k = np.searchsorted(ref_in, t)
                if 0 < k < len(ref_in):
                    out["fp_phase"].append({"record_id": rid, "variant": name.split("_")[0],
                                            "phase": fmt((t - ref_in[k - 1]) / (ref_in[k] - ref_in[k - 1]), ".3f")})

        # ---- задержки прихода пульса между областями ----
        for i_a, a in enumerate(ROIS):
            for b in ROIS[i_a + 1:]:
                ta, tb = np.sort(variants[f"V2_{a}"]["beats"]), np.sort(variants[f"V2_{b}"]["beats"])
                pr = match_beats(ta, tb, tol)
                d = [ta[i] - tb[j] for i, j in pr]
                out["roi_arrival_offsets"].append({"record_id": rid, "roi_a": a, "roi_b": b,
                                                   "median_offset_ms": fmt(np.median(d) if d else np.nan, ".2f"),
                                                   "n_matched": len(pr)})
        if rid in QA_RECORDS:
            qa_data[rid] = {"tu": tu, "x": x_ref, "ref": ref_ms, "cs": signals[v3_roi], "roi": v3_roi,
                            "v0": v0_beats, "v2": variants[f"V2_{v3_roi}"]["beats"], "subset": subset}
        print(f"  {rid:16s} эталон {len(ref_ms):4d} уд., V0 {len(v0_beats):4d}, переключений {switches:3d}, V3={v3_roi}", flush=True)

    for name in ("reference_beats", "baseline_window_peaks", "baseline_ibi_log", "offline_beats",
                 "metrics_detection", "metrics_prv", "seam_diagnostics", "roi_arrival_offsets", "v3_choice", "fp_phase"):
        write_csv(OUT / f"{name}.csv", out[name])
    hr = {"mae_by_records": float(np.mean(hr_check["maes"])), "coverage_mean": float(np.mean(hr_check["covs"]))}
    (OUT / "hr_check_after.json").write_text(json.dumps(hr, indent=2), encoding="utf-8")
    write_csv(OUT / "implausible_reference.csv",
              [{"record_id": r, "subset": s, "hr_ref_beats": fmt(h, ".2f"), "median_main_ref": fmt(m, ".2f")}
               for r, s, h, m in implausible] or [{"record_id": "", "subset": "", "hr_ref_beats": "", "median_main_ref": ""}])
    plot_qa(qa_data)
    print(f"частота пульса G0 во время прогона: MAE {hr['mae_by_records']:.4f}, покрытие {hr['coverage_mean']:.4f}")


def plot_qa(qa_data) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = QA_SEGMENT_S
    for rid, d in qa_data.items():
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        m = (d["tu"] >= a) & (d["tu"] <= b)
        ax[0].plot(d["tu"][m], d["x"][m], lw=0.8, color="0.3")
        rb = d["ref"][(d["ref"] / 1000 >= a) & (d["ref"] / 1000 <= b)] / 1000
        ax[0].plot(rb, np.interp(rb, d["tu"], d["x"]), "rv", ms=6, label="эталонные удары (MSPTD)")
        ax[0].set_title(f"{rid} (подмножество {d['subset']}): контактная волна")
        ax[0].legend(loc="upper right")
        cs = d["cs"]
        tv = cs.timestamps_ms / 1000
        mv = (tv >= a) & (tv <= b)
        ax[1].plot(tv[mv], cs.signal[mv], lw=0.8, color="0.3")
        for beats, style, lab in ((d["v2"], "go", f"V2 ({d['roi']})"), (d["v0"], "bx", "V0 (онлайн)")):
            bb = beats[(beats / 1000 >= a) & (beats / 1000 <= b)] / 1000
            ax[1].plot(bb, np.interp(bb, tv, cs.signal), style, ms=6, label=lab)
        for t in rb:
            ax[1].axvline(t, color="r", lw=0.4, alpha=0.5)
        ax[1].set_title(f"сигнал видео, область {d['roi']} (красные линии — эталонные удары)")
        ax[1].set_xlabel("время, с")
        ax[1].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(OUT / f"qa_{rid}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    run()
