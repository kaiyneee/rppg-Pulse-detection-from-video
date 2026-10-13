"""
Стенд экспериментов по точности на кэшированных сигналах UBFC.

Каждый вариант — функция trace -> окна (timestamp_ms, bpm, publishable).
Все варианты оцениваются ОДИНАКОВО: эталон — ЧСС по контактной волне той
же процедурой, что в основной работе (eval_ubfc_subject.reference_bpm_from_wave),
MAE считается по опубликованным окнам внутри записи и усредняется ПО
ЗАПИСЯМ. Отдельно — доля грубых промахов (|ошибка| > 5 уд/мин) и покрытие.

Разбиение на разработочную и тестовую части ЗАФИКСИРОВАНО заранее и не
зависит от ошибок записей:
  - разработочная: шесть прежних настроечных записей DATASET_1 и каждая
    третья запись DATASET_2 по возрастанию номера (14 записей);
  - тестовая: остальные 30, включая after-exercise (высокий пульс).
Решения принимаются только по разработочной части. Тестовая смотрится
один раз в конце, внешняя проверка — PURE.

    PYTHONPATH=src venv/bin/python3 benchmark/experiments.py baseline whitened
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import trace_cache as tc  # noqa: E402
import eval_ubfc_subject as ev  # noqa: E402

REF_CACHE = REPO_ROOT / "cache" / "refs.pkl"
RUN_DIR = REPO_ROOT / "cache" / "experiments"
OUTLIER_BPM = 5.0

DATASET1_DEV = ["5", "6", "7", "8", "10", "11"]


def _natural_key(s: str):
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def split() -> tuple[list[str], list[str]]:
    records = sorted((p.stem for p in tc.CACHE_DIR.glob("*.pkl")), key=_natural_key)
    d2 = sorted([r for r in records if r.startswith("subject")], key=lambda s: int(s[7:]))
    dev = DATASET1_DEV + d2[0::3]
    test = [r for r in records if r not in dev]
    return dev, test


# --------------------------------------------------------------------------
# Эталон по окнам (кэшируется: не зависит от варианта, только от метки окна)
# --------------------------------------------------------------------------
_REFS: dict | None = None


def _refs() -> dict:
    global _REFS
    if _REFS is None:
        _REFS = pickle.loads(REF_CACHE.read_bytes()) if REF_CACHE.exists() else {}
    return _REFS


def reference(trace: tc.RecordTrace, timestamp_ms: int, method: str = "welch") -> float | None:
    """Эталон окна ТОЙ ЖЕ процедурой, что оценка варианта (задача 53):
    reference_bpm_from_wave вызывает estimate_hr с cfg.frequency_method,
    поэтому при смене метода оценки эталон меняется вместе с ним и
    описывает тот же отрезок времени."""
    key = (trace.record_id, int(timestamp_ms)) if method == "welch" else (trace.record_id, int(timestamp_ms), method)
    refs = _refs()
    if key not in refs:
        from rppg.config import FrequencyMethod
        cfg = tc.make_config(trace.fps)
        cfg.frequency_method = FrequencyMethod(method)
        ref, _snr = ev.reference_bpm_from_wave(
            trace.t_ref, trace.wave, timestamp_ms / 1000.0, cfg.window.window_seconds, cfg,
        )
        refs[key] = ref
    return refs[key]


def save_refs() -> None:
    REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    REF_CACHE.write_bytes(pickle.dumps(_refs(), protocol=pickle.HIGHEST_PROTOCOL))


# --------------------------------------------------------------------------
# Варианты
# --------------------------------------------------------------------------
def _rows_from(results, method: str = "welch") -> list[dict]:
    return [{"timestamp_ms": r.timestamp_ms, "bpm": float(r.bpm), "publishable": bool(r.publishable),
             "method": method} for r in results]


def _cfg(trace, **overrides):
    from rppg.config import FrequencyMethod
    cfg = tc.make_config(trace.fps)
    if "frequency_method" in overrides:
        cfg.frequency_method = FrequencyMethod(overrides.pop("frequency_method"))
    for dotted, value in overrides.items():
        obj = cfg
        *path, last = dotted.split("__")
        for part in path:
            obj = getattr(obj, part)
        setattr(obj, last, value)
    return cfg


def make_variant(**overrides):
    method = overrides.get("frequency_method", "welch")

    def fn(trace):
        rows, _ = tc.replay(trace, _cfg(trace, **dict(overrides)))
        return _rows_from(rows, method)
    return fn


def v_baseline(trace):
    rows, _ = tc.replay(trace)
    return _rows_from(rows)


def v_whitened(trace):
    cfg = tc.make_config(trace.fps)
    cfg.frequency.peak_selection = "whitened"
    rows, _ = tc.replay(trace, cfg)
    return _rows_from(rows)


VARIANTS = {
    "baseline": v_baseline,
    "whitened": v_whitened,
    "fft": make_variant(frequency_method="fft"),
    "fft_nocont": make_variant(frequency_method="fft", frequency__continuity_enabled=False),
    "welch_mean": make_variant(roi__spectral_combination="spectral_mean"),
    "fft_mean": make_variant(frequency_method="fft", roi__spectral_combination="spectral_mean"),
    "fft_mean_nocont": make_variant(frequency_method="fft", roi__spectral_combination="spectral_mean",
                                    frequency__continuity_enabled=False),
}


# --------------------------------------------------------------------------
# Оценка
# --------------------------------------------------------------------------
def _full_refs() -> dict:
    global _FULL
    if _FULL is None:
        path = REPO_ROOT / "cache" / "ref_full.pkl"
        _FULL = pickle.loads(path.read_bytes()) if path.exists() else {}
    return _FULL


_FULL: dict | None = None


def score_record(trace, rows) -> dict:
    errs, errs_full, n_pub = [], [], 0
    for r in rows:
        if not r["publishable"]:
            continue
        n_pub += 1
        ref = reference(trace, r["timestamp_ms"], r.get("method", "welch"))
        if ref is None or not np.isfinite(r["bpm"]):
            continue
        errs.append(r["bpm"] - ref)
        rf = _full_refs().get((trace.record_id, r["timestamp_ms"]))
        if rf is not None:
            errs_full.append(r["bpm"] - rf)
    e = np.array(errs)
    ef = np.array(errs_full)
    return {
        "mae_full": float(np.mean(np.abs(ef))) if ef.size else float("nan"),
        "n_outlier_full": int((np.abs(ef) > OUTLIER_BPM).sum()),
        "n_full": int(ef.size),
        "record_id": trace.record_id,
        "n": len(rows),
        "n_pub": n_pub,
        "coverage": n_pub / max(len(rows), 1),
        "mae": float(np.mean(np.abs(e))) if e.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
        "bias": float(np.mean(e)) if e.size else float("nan"),
        "n_scored": int(e.size),
        "n_outlier": int((np.abs(e) > OUTLIER_BPM).sum()),
    }


def aggregate(scores: list[dict]) -> dict:
    mae = np.array([s["mae"] for s in scores if np.isfinite(s["mae"])])
    rmse = np.array([s["rmse"] for s in scores if np.isfinite(s["rmse"])])
    n_scored = sum(s["n_scored"] for s in scores)
    mae_full = np.array([s["mae_full"] for s in scores if np.isfinite(s["mae_full"])])
    return {
        "mae_full": float(mae_full.mean()) if mae_full.size else float("nan"),
        "outlier_full": sum(s["n_outlier_full"] for s in scores) / max(sum(s["n_full"] for s in scores), 1),
        "n_records": len(scores),
        "mae": float(mae.mean()) if mae.size else float("nan"),
        "mae_sd": float(mae.std(ddof=1)) if mae.size > 1 else float("nan"),
        "rmse": float(rmse.mean()) if rmse.size else float("nan"),
        "outlier_rate": sum(s["n_outlier"] for s in scores) / max(n_scored, 1),
        "coverage": float(np.mean([s["coverage"] for s in scores])),
        "bias": float(np.mean([s["bias"] for s in scores if np.isfinite(s["bias"])])),
    }


def run_variant(name: str, records: list[str]) -> dict[str, dict]:
    fn = VARIANTS[name]
    out = {}
    for rid in records:
        trace = tc.load_trace(rid)
        rows = fn(trace)
        out[rid] = {"rows": rows, "score": score_record(trace, rows)}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"{name}.pkl").write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    save_refs()
    return out


def report(name: str, out: dict, dev: list[str], test: list[str], show_test: bool) -> None:
    def line(label, ids):
        a = aggregate([out[r]["score"] for r in ids if r in out])
        return (f"  {label:6s} n={a['n_records']:2d}  MAE(свой эт.) {a['mae']:.3f}  промахи {a['outlier_rate']:6.2%}  "
                f"| MAE(полное окно) {a['mae_full']:.3f}  промахи {a['outlier_full']:6.2%}  "
                f"| покрытие {a['coverage']:6.1%}  смещение {a['bias']:+.3f}")
    print(f"=== {name} ===")
    print(line("dev", dev))
    if show_test:
        print(line("test", test))
        print(line("все", dev + test))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", choices=sorted(VARIANTS))
    ap.add_argument("--test", action="store_true", help="показать тестовую часть (только для финальной проверки)")
    ap.add_argument("--all-records", action="store_true", help="прогнать и тестовую часть (без показа, если нет --test)")
    args = ap.parse_args()

    dev, test = split()
    records = dev + (test if (args.test or args.all_records) else [])
    for name in args.variants:
        t0 = time.time()
        out = run_variant(name, records)
        report(name, out, dev, test, args.test)
        print(f"  ({time.time() - t0:.1f}с)")


if __name__ == "__main__":
    main()
