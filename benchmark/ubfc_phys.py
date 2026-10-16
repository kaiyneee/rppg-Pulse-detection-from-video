"""
Внешняя проверка на UBFC-Phys (T1 покой, T2 речь, T3 арифметика).

    PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py build   # кэш сигналов по видео (часы)
    PYTHONPATH=src venv/bin/python3 benchmark/ubfc_phys.py eval    # все таблицы из кэша

Ничего не настраивается на UBFC-Phys: конфигурации, правило надёжности
эталона, пороги чистоты спектра и терцили движения заданы ниже до прогона.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import degenerate as dg  # noqa: E402
import reference_full10s as R  # noqa: E402
import trace_cache as tc  # noqa: E402
from rppg.config import PRESET_OVERRIDES, FrequencyMethod, make_pipeline_config  # noqa: E402
from rppg.hrv.offline_prv import continuous_roi_signal  # noqa: E402
from rppg.signal.preprocessing import bandpass_filter  # noqa: E402

DATA_DIR = Path.home() / "Desktop" / "проекты" / "статья пульс по видо" / "второй тестовый датасет "
CACHE_DIR = REPO_ROOT / "cache" / "ubfc_phys"
OUT = REPO_ROOT / "results" / "ubfc_phys"

BVP_FS = 64.0
TASKS = ("T1", "T2", "T3")
FULL_SUBJECTS = ("s1", "s2", "s3", "s4", "s5", "s10", "s11")   # основные выводы
TRUNCATED_SUBJECTS = ("s6", "s7", "s8", "s9")                  # только T1, обрезаны
ROIS = ("forehead", "left_cheek", "right_cheek")

# Четыре варианта абляции 2x2 — те же переопределения, что в benchmark/ablation_2x2.py.
CONFIGS = {
    "legacy_welch8": ("legacy_welch8", {}),
    "fft_best_snr": ("legacy_welch8", {"frequency_method": "fft"}),
    "welch8_spectral_mean": ("legacy_welch8", {"roi.spectral_combination": "spectral_mean"}),
    "fft_spectral_mean": ("fft_spectral_mean", {}),
}
DEFAULT_CONFIG = "fft_spectral_mean"
GROSS_BPM = 5.0
RELIABLE_MAX_DIFF_BPM = 5.0          # ref_reliable = |main - alt| <= 5
ALT_DISTANCE_S = 0.33
ALT_PROMINENCE_STD = 0.3
PURITY_THRESHOLDS = [None, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40]
SYNC_MAX_LAG_S = 3.0


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------
def records() -> list[dict]:
    out = []
    for sdir in sorted(DATA_DIR.glob("s*"), key=lambda p: int(p.name[1:])):
        info = (sdir / f"info_{sdir.name}.txt").read_text().split()
        for task in TASKS:
            video = sdir / f"vid_{sdir.name}_{task}.avi"
            if video.exists():
                out.append({"subject": sdir.name, "task": task, "group": info[2], "video": video,
                            "bvp": sdir / f"bvp_{sdir.name}_{task}.csv"})
    return out


def record_set(subject: str) -> str:
    if subject in FULL_SUBJECTS:
        return "full"
    if subject in TRUNCATED_SUBJECTS:
        return "truncated"
    return "extra"   # есть в папке, но не входит в задание


def load_bvp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.loadtxt(path, delimiter=",", ndmin=1).astype(float)
    return np.arange(len(x)) / BVP_FS, x


def key(rec) -> str:
    return f"{rec['subject']}_{rec['task']}"


def load_trace(k: str):
    with (CACHE_DIR / f"{k}.pkl").open("rb") as fh:
        return tc._Unpickler(fh).load()


def cmd_build() -> None:
    import time
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for rec in records():
        path = CACHE_DIR / f"{key(rec)}.pkl"
        if path.exists():
            continue
        t0 = time.time()
        t_ref, bvp = load_bvp(rec["bvp"])
        trace = tc.build_trace_from_video(key(rec), rec["video"], t_ref, bvp, np.full(len(bvp), np.nan),
                                          subset=record_set(rec["subject"]))
        path.write_bytes(pickle.dumps(trace, protocol=pickle.HIGHEST_PROTOCOL))
        print(f"  {key(rec):8s} кадров {len(trace.frames):5d}, окон {len(trace.live):4d}, {time.time() - t0:6.0f} с", flush=True)
    print("BUILD_DONE", flush=True)


# --------------------------------------------------------------------------
# Эталон
# --------------------------------------------------------------------------
def alt_reference(t_ref, bvp_filtered, end_s) -> float | None:
    """Пульс по ударам: find_peaks на отфильтрованной BVP окна, 60 / медиана интервалов."""
    m = (t_ref >= end_s - 10.0) & (t_ref <= end_s)
    x = bvp_filtered[m]
    if len(x) < 16:
        return None
    p, _ = find_peaks(x, distance=int(round(ALT_DISTANCE_S * BVP_FS)), prominence=ALT_PROMINENCE_STD * float(np.std(x)))
    if len(p) < 3:
        return None
    return float(60.0 / np.median(np.diff(t_ref[m][p])))


def bvp_bandpassed(bvp):
    return bandpass_filter(bvp - np.mean(bvp), BVP_FS, R.BAND_HZ[0], R.BAND_HZ[1], order=R.FILTER_ORDER)


# --------------------------------------------------------------------------
# Прогон конфигураций
# --------------------------------------------------------------------------
def config_for(name: str, fps: float, purity: float | None = None):
    preset, over = CONFIGS[name]
    cfg = tc.make_config(fps, preset)
    for k, v in over.items():
        if k == "frequency_method":
            cfg.frequency_method = FrequencyMethod(v)
        else:
            obj, attr = k.split(".")
            setattr(getattr(cfg, obj), attr, v)
    cfg.quality.min_spectral_purity = purity
    return cfg


def run_config(trace, name, purity=None):
    holder: dict = {"fps": None}
    fs_by_ts = {}

    def on_result(pipe, res):
        fs_by_ts[res.timestamp_ms] = holder["fps"]

    with dg.trace_estimate_fps(holder):
        rows, _ = tc.replay(trace, config_for(name, trace.fps, purity), on_result=on_result)
    return rows, fs_by_ts


def window_motion(trace, end_ms) -> float:
    """Медиана по кадрам окна межкадрового смещения опорных точек, делённого на
    межзрачковое расстояние."""
    ts = np.asarray(trace.timestamps_ms)
    idx = np.where((ts >= end_ms - 10000) & (ts <= end_ms))[0]
    vals, prev = [], None
    for i in idx:
        f = trace.frames[i]
        if not f.detected or f.roi is None:
            prev = None
            continue
        pts, ipd = np.asarray(f.roi.stable_tracking_points, float), float(f.roi.interocular_distance_px)
        if prev is not None and ipd > 0:
            vals.append(float(np.mean(np.linalg.norm(pts - prev, axis=1)) / ipd))
        prev = pts
    return float(np.median(vals)) if vals else float("nan")


LAG_MAX_S = 10.0
LAG_STEP_S = 0.25
LAG_MIN_R = 0.5


def video_continuous(trace):
    """Среднее z-нормированных непрерывных сигналов трёх ROI (умолчание)."""
    cfg = make_pipeline_config()
    ts = np.asarray(trace.timestamps_ms, float)
    sigs = []
    for roi in ROIS:
        rgb = np.array([f.roi.rgb_by_roi.get(roi) if (f.detected and f.roi is not None and f.roi.rgb_by_roi.get(roi) is not None)
                        else np.zeros(3) for f in trace.frames], float)
        valid = np.array([bool(f.detected and f.roi is not None and f.roi.valid_by_roi.get(roi, False)
                               and f.roi.rgb_by_roi.get(roi) is not None) for f in trace.frames])
        sg = continuous_roi_signal(ts, rgb, valid, roi, cfg).signal
        sigs.append((sg - sg.mean()) / (sg.std() + 1e-12))
    fps = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0)
    return ts / 1000.0, np.mean(sigs, axis=0), fps


def beat_hr_series(t_s, x, fs):
    """Мгновенный пульс по ударам MSPTD: время середины интервала, уд/мин,
    сглаженный скользящей медианой по 5 ударам."""
    from scipy.signal import medfilt
    from rppg.hrv.features import refine_peaks_subsample
    from rppg.hrv.offline_prv import msptd_beat_detector, MIN_IBI_MS, MAX_IBI_MS
    p, _ = msptd_beat_detector(x, fs)
    bt = np.interp(refine_peaks_subsample(x, p), np.arange(len(t_s)), t_s)
    ibi = np.diff(bt) * 1000.0
    ok = (ibi >= MIN_IBI_MS) & (ibi <= MAX_IBI_MS)
    mid, hr = ((bt[:-1] + bt[1:]) / 2)[ok], (60000.0 / ibi)[ok]
    if len(hr) >= 5:
        hr = medfilt(hr, 5)
    return mid, hr


def estimate_lag(trace) -> tuple[float, float, bool]:
    """Сдвиг BVP относительно видео (эталонное время = время видео + сдвиг)
    по корреляции рядов мгновенного пульса; не зависит от сравниваемых конфигураций."""
    tv, v, fps = video_continuous(trace)
    mv, hv = beat_hr_series(tv, v, fps)
    t_ref = np.asarray(trace.t_ref, float)
    mb, hb = beat_hr_series(t_ref, bvp_bandpassed(np.asarray(trace.wave, float)), BVP_FS)
    if len(hv) < 10 or len(hb) < 10:
        return 0.0, float("nan"), False
    grid = np.arange(mv[0], mv[-1], 0.25)
    vg = np.interp(grid, mv, hv)
    best = (0.0, -np.inf)
    for lag in np.arange(-LAG_MAX_S, LAG_MAX_S + 1e-9, LAG_STEP_S):
        tt = grid + lag
        ok = (tt >= mb[0]) & (tt <= mb[-1])
        if ok.sum() < 40:
            continue
        r = float(np.corrcoef(vg[ok], np.interp(tt[ok], mb, hb))[0, 1])
        if np.isfinite(r) and r > best[1]:
            best = (float(lag), r)
    return best[0], best[1], bool(best[1] >= LAG_MIN_R)


def sync_lag(trace) -> tuple[float, float]:
    """Задержка BVP относительно сигнала видео по взаимной корреляции, ±3 с.
    Сигнал видео — среднее z-нормированных непрерывных сигналов трёх ROI
    (аналог усреднения областей текущего умолчания). Положительная задержка —
    BVP позже видео. Знак корреляции не ограничивается."""
    t_s, v, fps = video_continuous(trace)
    b = np.interp(t_s, trace.t_ref, bvp_bandpassed(np.asarray(trace.wave, float)))
    v, b = (v - v.mean()) / v.std(), (b - b.mean()) / b.std()
    max_lag = int(round(SYNC_MAX_LAG_S * fps))
    best = (0.0, 0.0)
    for lag in range(-max_lag, max_lag + 1):
        a, c = (v[:len(v) - lag], b[lag:]) if lag >= 0 else (v[-lag:], b[:len(b) + lag])
        r = float(np.mean(a * c))
        if abs(r) > abs(best[1]):
            best = (lag / fps, r)
    return best


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------
def agg(rows_, reliable_only=True):
    use = [r for r in rows_ if r["published"] and np.isfinite(r["err"]) and (r["ref_reliable"] or not reliable_only)]
    e = np.array([r["err"] for r in use])
    return use, e


def write(path, rows_):
    if not rows_:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
        w.writeheader()
        w.writerows(rows_)


def fmt(x, spec=".4f"):
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else format(float(x), spec)
    return x


def regression_check() -> dict:
    ref = R.load_reference()
    maes, covs = [], []
    for p in sorted(tc.CACHE_DIR.glob("*.pkl")):
        t = tc.load_trace(p.stem)
        rows, _ = tc.replay(t, tc.make_config(t.fps))
        e = [abs(r.bpm - ref[(p.stem, r.timestamp_ms)]) for r in rows
             if r.publishable and (p.stem, r.timestamp_ms) in ref and np.isfinite(r.bpm)]
        maes.append(np.mean(e)); covs.append(sum(r.publishable for r in rows) / len(rows))
    return {"mae_by_records": float(np.mean(maes)), "coverage_mean": float(np.mean(covs))}


def summarize_block(win, config, task):
    rows_ = [w for w in win if w["config"] == config and w["task"] == task]
    if not rows_:
        return None
    use, e = agg(rows_)
    _, e_all = agg(rows_, reliable_only=False)
    per_rec = {}
    for w in use:
        per_rec.setdefault(w["subject"], []).append(abs(w["err"]))
    sd = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
    return {"config": config, "task": task, "n_windows": len(rows_),
            "n_reliable": sum(1 for w in rows_ if w["ref_reliable"]),
            "coverage": fmt(np.mean([w["published"] for w in rows_])),
            "mae": fmt(np.mean([np.mean(v) for v in per_rec.values()]) if per_rec else np.nan),
            "gross_rate": fmt(np.mean(np.abs(e) > GROSS_BPM) if len(e) else np.nan),
            "ba_bias": fmt(np.mean(e) if len(e) else np.nan),
            "ba_loa_low": fmt(np.mean(e) - 1.96 * sd if len(e) > 1 else np.nan),
            "ba_loa_high": fmt(np.mean(e) + 1.96 * sd if len(e) > 1 else np.nan),
            "ref_bpm_median": fmt(np.median([w["ref"] for w in rows_ if w["ref_reliable"]])
                                  if any(w["ref_reliable"] for w in rows_) else np.nan),
            "mae_all_ref_windows": fmt(np.mean(np.abs(e_all)) if len(e_all) else np.nan),
            "gross_rate_all_ref_windows": fmt(np.mean(np.abs(e_all) > GROSS_BPM) if len(e_all) else np.nan)}


def src_sha256() -> str:
    h = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        h.update(path.relative_to(REPO_ROOT).as_posix().encode()); h.update(path.read_bytes())
    return h.hexdigest()


def cmd_eval(mode: str = "own") -> None:
    """mode: own — свой сдвиг записи (если определён); none — без сдвига;
    from_t1 — свой сдвиг, а где он не определён на T2/T3 — сдвиг T1 того же
    участника (если определён на T1)."""
    shift = mode != "none"
    out_dir = {"own": OUT, "none": OUT / "no_shift", "from_t1": OUT / "shift_from_T1"}[mode]
    t1_lag: dict[str, float] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = regression_check()
    print(f"регрессия UBFC-rPPG G0: MAE {reg['mae_by_records']:.4f}, покрытие {reg['coverage_mean']:.4f}")
    if f"{reg['mae_by_records']:.4f}" != "1.1970" or f"{reg['coverage_mean']:.4f}" != "0.9921":
        raise SystemExit("регрессия не сошлась с точкой сверки PRV — стоп")

    recs = [r for r in records() if (CACHE_DIR / f"{key(r)}.pkl").exists()]
    data_check, ref_quality, sync, win, lags = [], [], [], [], []
    identified = {}
    traces = {}
    for rec in recs:
        k = key(rec)
        trace = load_trace(k)
        traces[k] = trace
        n = len(trace.frames)
        rs = record_set(rec["subject"])
        data_check.append({"subject": rec["subject"], "task": rec["task"], "group": rec["group"],
                           "n_frames_decoded": n, "video_s": fmt(n / trace.fps, ".2f"),
                           "bvp_s": fmt(len(trace.wave) / BVP_FS, ".2f"), "truncated": fmt(rs == "truncated"),
                           "set": rs, "fps_container": fmt(trace.fps, ".4f")})
        bvp = np.asarray(trace.wave, float)
        bvp_f = bvp_bandpassed(bvp)
        lag, lag_r, lag_ok = estimate_lag(trace)
        applied = lag if (shift and lag_ok) else 0.0
        source = "own" if (shift and lag_ok) else "none"
        if rec["task"] == "T1" and lag_ok:
            t1_lag[rec["subject"]] = lag
        if mode == "from_t1" and not lag_ok and rec["task"] != "T1" and rec["subject"] in t1_lag:
            applied, source = t1_lag[rec["subject"]], "T1"
        lags.append({"subject": rec["subject"], "task": rec["task"], "set": rs, "lag_s": fmt(lag, ".2f"),
                     "r_at_lag": fmt(lag_r, ".3f"), "lag_identified": fmt(lag_ok), "applied_shift_s": fmt(applied, ".2f"), "shift_source": source})
        identified[k] = lag_ok
        refs = {}
        for live in trace.live:
            end_s = live["timestamp_ms"] / 1000.0 + applied
            m = R.reference_full10s(trace.t_ref, bvp, end_s)
            a = alt_reference(trace.t_ref, bvp_f, end_s)
            main = None if m is None else m["bpm_ref"]
            refs[live["timestamp_ms"]] = (main, a, main is not None and a is not None and abs(main - a) <= RELIABLE_MAX_DIFF_BPM)
        for i, (ts_, (m_, a_, ok)) in enumerate(sorted(refs.items())):
            ref_quality.append({"subject": rec["subject"], "task": rec["task"], "window_idx": i,
                                "t_start_s": fmt(ts_ / 1000 - 10, ".3f"), "ref_main_bpm": fmt(m_, ".3f"),
                                "ref_alt_bpm": fmt(a_, ".3f"), "ref_reliable": fmt(ok)})
        if rec["task"] == "T1" and rs == "full":
            lag, corr = sync_lag(trace)
            sync.append({"subject": rec["subject"], "lag_s": fmt(lag, ".3f"), "corr": fmt(corr, ".3f")})
        motion = {ts_: window_motion(trace, ts_) for ts_ in refs}
        for cname in CONFIGS:
            rows, fs_by_ts = run_config(trace, cname)
            for i, r in enumerate(rows):
                m_, _a, ok = refs.get(r.timestamp_ms, (None, None, False))
                sqi = r.debug.sqi if r.debug is not None else None
                win.append({"subject": rec["subject"], "task": rec["task"], "group": rec["group"],
                            "set": rs, "config": cname, "window_idx": i, "t_start_s": r.timestamp_ms / 1000 - 10,
                            "bpm": float(r.bpm), "ref": m_ if m_ is not None else np.nan, "ref_reliable": ok,
                            "published": bool(r.publishable),
                            "err": (float(r.bpm) - m_) if (m_ is not None and np.isfinite(r.bpm)) else np.nan,
                            "sqi": sqi, "composite": float(r.sqi_score), "motion": motion.get(r.timestamp_ms, np.nan),
                            "fs": fs_by_ts.get(r.timestamp_ms), "ts": r.timestamp_ms, "key": k})
        print(f"  {k:8s} [{rs}] окон {len(refs)}, надёжных {sum(v[2] for v in refs.values())}", flush=True)

    # терцили движения — по всем окнам 21 полного видео, до сравнения ошибок
    mot = np.array([w["motion"] for w in win if w["set"] == "full" and w["config"] == DEFAULT_CONFIG and np.isfinite(w["motion"])])
    t1, t2 = np.percentile(mot, [100 / 3, 200 / 3])
    for w in win:
        w["tertile"] = "" if not np.isfinite(w["motion"]) else (1 if w["motion"] <= t1 else (2 if w["motion"] <= t2 else 3))

    full = [w for w in win if w["set"] == "full"]
    summary = [b for c_ in CONFIGS for t in TASKS if (b := summarize_block(full, c_, t))]
    trunc = [b for c_ in CONFIGS if (b := summarize_block([w for w in win if w["set"] == "truncated"], c_, "T1"))]
    extra = [b for c_ in CONFIGS for t in TASKS if (b := summarize_block([w for w in win if w["set"] == "extra"], c_, t))]

    per_record = []
    for c_ in CONFIGS:
        for k in traces:
            rows_ = [w for w in win if w["config"] == c_ and w["key"] == k]
            use, e = agg(rows_)
            per_record.append({"config": c_, "subject": rows_[0]["subject"], "task": rows_[0]["task"],
                               "set": rows_[0]["set"], "mae": fmt(np.mean(np.abs(e)) if len(e) else np.nan),
                               "gross_rate": fmt(np.mean(np.abs(e) > GROSS_BPM) if len(e) else np.nan),
                               "coverage": fmt(np.mean([w["published"] for w in rows_]))})

    pairs = []
    for a, b in (("fft_spectral_mean", "legacy_welch8"), ("fft_spectral_mean", "fft_best_snr"),
                 ("fft_best_snr", "legacy_welch8"), ("welch8_spectral_mean", "legacy_welch8")):
        for t in TASKS:
            da = {r["subject"]: float(r["mae"]) for r in per_record if r["config"] == a and r["task"] == t and r["set"] == "full" and r["mae"]}
            db = {r["subject"]: float(r["mae"]) for r in per_record if r["config"] == b and r["task"] == t and r["set"] == "full" and r["mae"]}
            d = np.array([da[s] - db[s] for s in da if s in db])
            pairs.append({"config_a": a, "config_b": b, "task": t, "n_records": len(d),
                          "median_mae_diff_a_minus_b": fmt(np.median(d) if len(d) else np.nan),
                          "n_a_better": int((d < 0).sum())})

    strata = []
    for c_ in CONFIGS:
        for t in TASKS:
            for q in (1, 2, 3):
                rows_ = [w for w in full if w["config"] == c_ and w["task"] == t and w["tertile"] == q]
                _, e = agg(rows_)
                strata.append({"config": c_, "task": t, "motion_tertile": q, "n_windows": len(rows_),
                               "mae": fmt(np.mean(np.abs(e)) if len(e) else np.nan),
                               "gross_rate": fmt(np.mean(np.abs(e) > GROSS_BPM) if len(e) else np.nan)})

    curve = []
    full_keys = [k for k in traces if traces[k].subset == "full"]
    ref_by = {(w["key"], w["ts"]): (w["ref"], w["ref_reliable"]) for w in full if w["config"] == DEFAULT_CONFIG}
    for thr in PURITY_THRESHOLDS:
        by_task = {t: {"n": 0, "pub": 0, "errs": [], "per_rec": {}} for t in TASKS}
        for k in full_keys:
            rows, _ = run_config(traces[k], DEFAULT_CONFIG, thr)
            t = k.split("_")[1]
            for r in rows:
                by_task[t]["n"] += 1
                if r.publishable:
                    by_task[t]["pub"] += 1
                    ref_, ok = ref_by.get((k, r.timestamp_ms), (np.nan, False))
                    if ok and np.isfinite(r.bpm):
                        by_task[t]["errs"].append(r.bpm - ref_)
                        by_task[t]["per_rec"].setdefault(k, []).append(abs(r.bpm - ref_))
        for t in TASKS:
            if not by_task[t]["n"]:
                continue
            e = np.array(by_task[t]["errs"])
            curve.append({"task": t, "threshold": "" if thr is None else thr,
                          "coverage": fmt(by_task[t]["pub"] / by_task[t]["n"]),
                          "mae": fmt(np.mean(np.abs(e)) if len(e) else np.nan),
                          "mae_by_records": fmt(np.mean([np.mean(v) for v in by_task[t]["per_rec"].values()])
                                                if by_task[t]["per_rec"] else np.nan),
                          "n_records_with_output": len(by_task[t]["per_rec"]),
                          "gross_rate": fmt(np.mean(np.abs(e) > GROSS_BPM) if len(e) else np.nan)})

    windows_csv = []
    for w in win:
        s = w["sqi"]
        windows_csv.append({
            "subject": w["subject"], "task": w["task"], "group": w["group"], "truncated": fmt(w["set"] == "truncated"),
            "config": w["config"], "window_idx": w["window_idx"], "t_start_s": fmt(w["t_start_s"], ".3f"),
            "bpm_estimate": fmt(w["bpm"], ".3f"), "bpm_reference": fmt(w["ref"], ".3f"), "ref_reliable": fmt(w["ref_reliable"]),
            "published": fmt(w["published"]), "sqi_composite": fmt(w["composite"]),
            "sqi_snr": fmt(s.snr_score) if s else "", "snr_db": fmt(s.spectral_snr_db, ".3f") if s else "",
            "sqi_roi_agreement": fmt(s.cross_roi_agreement) if s else "",
            "sqi_landmark_stability": fmt(s.landmark_stability) if s else "",
            "sqi_temporal_consistency": fmt(s.temporal_consistency) if s else "",
            "spectral_purity": fmt(s.spectral_purity) if s else "", "motion": fmt(w["motion"], ".6f"),
            "motion_tertile": w["tertile"], "fs_used_hz": fmt(w["fs"], ".4f"), "set": w["set"]})

    ident = [w for w in full if identified.get(w["key"])]
    summary_ident = [b for c_ in CONFIGS for t in TASKS if (b := summarize_block(ident, c_, t))]
    for name, rows_ in (("record_lag", lags), ("summary_lag_identified", summary_ident), ("data_check", data_check), ("ref_quality", ref_quality), ("sync_lag", sync),
                        ("windows", windows_csv), ("summary", summary), ("summary_truncated", trunc),
                        ("per_record", per_record), ("pairwise", pairs), ("motion_strata", strata),
                        ("purity_curve", curve)):
        write(out_dir / f"{name}.csv", rows_)
    if extra:
        write(out_dir / "summary_extra_s12.csv", extra)

    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip() or None
    meta = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "src_sha256": src_sha256(),
            "presets_sha256": hashlib.sha256(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in PRESET_OVERRIDES.items()},
                                                        sort_keys=True).encode()).hexdigest(),
            "configs": {k: {"preset": p, "overrides": o} for k, (p, o) in CONFIGS.items()},
            "default_config": DEFAULT_CONFIG,
            "reference_main": "benchmark/reference_full10s.py (full10s-v1) на BVP",
            "reference_alt": f"find_peaks на BVP после полосового фильтра {R.BAND_HZ} Гц порядка {R.FILTER_ORDER}, "
                             f"distance={ALT_DISTANCE_S} с, prominence={ALT_PROMINENCE_STD}*СКО окна, 60/медиана интервалов",
            "reliable_rule": f"|ref_main_bpm - ref_alt_bpm| <= {RELIABLE_MAX_DIFF_BPM}",
            "gross_error": f"|оценка - эталон| > {GROSS_BPM} уд/мин",
            "purity_thresholds": PURITY_THRESHOLDS, "motion_tertile_edges": [float(t1), float(t2)],
            "regression_ubfc_rppg_g0": reg,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__}}
    meta.update({"shift_mode": mode, "shift_applied": shift, "lag_method": "MSPTD instantaneous HR cross-correlation, +-10 s, step 0.25 s, accept r >= 0.5"})
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EVAL_DONE")


def cmd_lagdiag() -> None:
    """Диагностика: сдвиг ряда пульса BVP относительно ряда пульса видео
    (текущее умолчание), ±15 с с шагом 0,5 с. В основных таблицах ничего
    не сдвигается."""
    rows_out = []
    for rec in records():
        k = key(rec)
        if not (CACHE_DIR / f"{k}.pkl").exists():
            continue
        trace = load_trace(k)
        rows, _ = run_config(trace, DEFAULT_CONFIG)
        est = np.array([r.bpm for r in rows])
        ends = np.array([r.timestamp_ms / 1000.0 for r in rows])
        bvp = np.asarray(trace.wave, float)
        res = []
        for sh in np.arange(-15.0, 15.0 + 1e-9, 0.5):
            ref = np.array([(R.reference_full10s(trace.t_ref, bvp, e + sh) or {"bpm_ref": np.nan})["bpm_ref"] for e in ends])
            ok = np.isfinite(ref) & np.isfinite(est)
            if ok.sum() < 20:
                continue
            res.append((sh, float(np.mean(np.abs(est[ok] - ref[ok]))), float(np.corrcoef(est[ok], ref[ok])[0, 1])))
        best = min(res, key=lambda x: x[1])
        zero = next(x for x in res if abs(x[0]) < 1e-9)
        rows_out.append({"subject": rec["subject"], "task": rec["task"], "set": record_set(rec["subject"]),
                         "best_shift_s": fmt(best[0], ".1f"), "mae_at_best": fmt(best[1], ".2f"), "r_at_best": fmt(best[2], ".2f"),
                         "mae_at_0": fmt(zero[1], ".2f"), "r_at_0": fmt(zero[2], ".2f")})
        print(f"  {k:8s} лучший сдвиг {best[0]:+5.1f} с: MAE {best[1]:5.2f} (r {best[2]:+.2f}); при 0: MAE {zero[1]:5.2f} (r {zero[2]:+.2f})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "hr_series_lag_diagnostic.csv", rows_out)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        cmd_eval("none" if "--no-shift" in sys.argv else ("from_t1" if "--shift-from-t1" in sys.argv else "own"))
    else:
        {"build": cmd_build, "lagdiag": cmd_lagdiag}[sys.argv[1]]()
