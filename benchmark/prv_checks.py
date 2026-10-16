"""
Проверки результатов PRV по сохранённым ударам (results/prv): полярность
сигналов, сдвиг ударов видео относительно эталона по записям, метрики после
вычитания сдвига своей записи — против вершин и против начал волны эталона.

Вычитание сдвига своей записи — поправка, оценённая по тем же данным; это
диагностика, а не метрика метода.

    PYTHONPATH=src venv/bin/python3 benchmark/prv_checks.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import skew

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import prv_eval as pe  # noqa: E402
import trace_cache as tc  # noqa: E402
import reference_full10s as R  # noqa: E402
from rppg.hrv.features import refine_peaks_subsample  # noqa: E402
from rppg.hrv.offline_prv import (  # noqa: E402
    IBISeries, continuous_roi_signal, ibi_series_from_beats, msptd_beat_detector,
)
from rppg.signal.preprocessing import bandpass_filter  # noqa: E402

PRV = REPO_ROOT / "results" / "prv"
OUT = PRV / "checks"
TOLS = (150.0, 50.0)
XCORR_MAX_S = 0.5


def rd(name):
    return list(csv.DictReader((PRV / f"{name}.csv").open(encoding="utf-8")))


def nearest_offsets(video, ref):
    ref = np.sort(ref)
    k = np.clip(np.searchsorted(ref, video), 1, len(ref) - 1)
    prev, nxt = ref[k - 1], ref[k]
    return np.where(video - prev < nxt - video, video - prev, video - nxt)


def metrics(video, video_series, ref, span, tol):
    vb = pe.in_span(video, span)
    det, _ = pe.detection(vb, pe.in_span(ref, span), tol)
    errs, _ = pe.ibi_errors(video_series.t_ms, video_series.ibi_ms, ref, tol)
    return det, errs


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_peaks = defaultdict(list)
    for r in rd("reference_beats"):
        ref_peaks[r["record_id"]].append(float(r["t_s"]) * 1000)
    v3_roi = {r["record_id"]: r["roi"] for r in rd("v3_choice")}
    v3 = defaultdict(list)
    for r in rd("offline_beats"):
        if r["variant"] == "V2" and r["roi"] == v3_roi[r["record_id"]]:
            v3[r["record_id"]].append(float(r["t_s"]) * 1000)
    v0_log = defaultdict(list)
    for r in rd("baseline_ibi_log"):
        v0_log[r["record_id"]].append((float(r["pt_ms"]), float(r["gap_ms"]), r["adjacent_to_prev"] == "1"))

    pol, offs, met = [], [], []
    for rid in sorted(ref_peaks, key=lambda s: (len(s), s)):
        trace = tc.load_trace(rid)
        subset = trace.subset
        cfg = tc.make_config(trace.fps)
        ts = np.asarray(trace.timestamps_ms, float)
        span = (ts[0] + 1000.0, ts[-1] - 1000.0)

        # эталон: вершины (сохранены) и начала волны (MSPTD onsets + параболическое уточнение)
        t, x0 = np.asarray(trace.t_ref, float), np.asarray(trace.wave, float)
        tu = np.linspace(t[0], t[-1], len(t))
        x = np.interp(tu, t, x0)
        fs = (len(t) - 1) / (t[-1] - t[0])
        _, on = msptd_beat_detector(x, fs)
        onsets = np.interp(refine_peaks_subsample(x, on), np.arange(len(tu)), tu) * 1000.0
        peaks = np.sort(np.array(ref_peaks[rid]))

        # полярность
        roi = v3_roi[rid]
        rgb = np.array([f.roi.rgb_by_roi.get(roi) if (f.detected and f.roi is not None and f.roi.rgb_by_roi.get(roi) is not None)
                        else np.zeros(3) for f in trace.frames], float)
        valid = np.array([bool(f.detected and f.roi is not None and f.roi.valid_by_roi.get(roi, False)
                               and f.roi.rgb_by_roi.get(roi) is not None) for f in trace.frames])
        cs = continuous_roi_signal(ts, rgb, valid, roi, cfg)
        xb = bandpass_filter(x - x.mean(), fs, R.BAND_HZ[0], R.BAND_HZ[1], order=R.FILTER_ORDER)
        ref_on_video = np.interp(ts / 1000.0, tu, xb)
        a = (cs.signal - cs.signal.mean()) / cs.signal.std()
        b = (ref_on_video - ref_on_video.mean()) / ref_on_video.std()
        max_lag = int(round(XCORR_MAX_S * cs.fps))
        best = (0, 0.0)
        for lag in range(-max_lag, max_lag + 1):
            u, v = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
            r_ = float(np.mean(u * v))
            if abs(r_) > abs(best[1]):
                best = (lag, r_)
        pol.append({"record_id": rid, "subset": subset, "roi": roi,
                    "skew_video_v2_signal": f"{skew(cs.signal):.3f}", "skew_reference_raw": f"{skew(x):.3f}",
                    "skew_reference_bandpassed": f"{skew(xb):.3f}",
                    "xcorr_max_abs_r": f"{best[1]:.3f}", "xcorr_lag_s": f"{best[0] / cs.fps:.3f}",
                    "xcorr_sign": "+" if best[1] > 0 else "-"})

        # ряды видео
        v3_beats = np.sort(np.array(v3[rid]))
        log = v0_log[rid]
        v0_beats = np.unique(np.concatenate([[p for p, _, _ in log], [p - g for p, g, _ in log]])) if log else np.array([])
        series = {
            "V3": (v3_beats, ibi_series_from_beats(v3_beats)),
            "V0": (v0_beats, IBISeries(np.array([p for p, _, _ in log]), np.array([g for _, g, _ in log]),
                                       np.array([a_ for _, _, a_ in log], dtype=bool))),
        }
        for vname, (beats, s) in series.items():
            for target, ref in (("peaks", peaks), ("onsets", onsets)):
                d = nearest_offsets(pe.in_span(beats, span), ref)
                med = float(np.median(d)) if len(d) else float("nan")
                offs.append({"record_id": rid, "subset": subset, "variant": vname, "reference": target,
                             "median_offset_ms": f"{med:.1f}",
                             "iqr_offset_ms": f"{np.percentile(d, 75) - np.percentile(d, 25):.1f}" if len(d) else ""})
                for corrected in (False, True):
                    sh = med if corrected else 0.0
                    s_sh = IBISeries(s.t_ms - sh, s.ibi_ms, s.adjacent)
                    for tol in TOLS:
                        det, errs = metrics(beats - sh, s_sh, ref, span, tol)
                        met.append({"record_id": rid, "subset": subset, "variant": vname, "reference": target,
                                    "own_offset_subtracted": int(corrected), "tolerance_ms": int(tol),
                                    "f1": f"{det['f1']:.4f}", "sensitivity": f"{det['sensitivity']:.4f}",
                                    "ppv": f"{det['ppv']:.4f}", "n_ibi_pairs": len(errs),
                                    "ibi_mae_ms": f"{np.mean(np.abs(errs)):.2f}" if len(errs) else "",
                                    "ibi_bias_ms": f"{np.mean(errs):.2f}" if len(errs) else ""})
        print(f"  {rid:16s} сдвиг V3 к вершинам {offs[-4]['median_offset_ms']:>7s} мс, к началам {offs[-3]['median_offset_ms']:>7s} мс", flush=True)

    write(OUT / "polarity.csv", pol)
    write(OUT / "offsets.csv", offs)
    write(OUT / "shifted_metrics.csv", met)
    print("CHECKS_DONE")


if __name__ == "__main__":
    main()
