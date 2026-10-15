"""
Основной эталон ЧСС для оценки на UBFC-rPPG: контактная пульсовая волна на
ПОЛНОМ 10-секундном окне, один фиксированный способ.

Эталон НЕ зависит от настроек оценщика: все параметры заморожены здесь
константами и не читаются из PipelineConfig. Считается один раз для всех
окон и сохраняется в results/reference/ubfc_ref_full10s.csv; оценки
сравниваются с файлом, а не пересчитывают эталон под свой конфиг.

Как считается (для окна, заканчивающегося в момент t_end):
  1. берутся отсчёты контактной волны с меткой t в [t_end - 10 с, t_end];
  2. приводятся к равномерной сетке: линейная интерполяция на N точек
     между первой и последней меткой окна (N — число исходных отсчётов;
     нужно потому, что во втором подмножестве UBFC шаг прибора гуляет от
     0,010 до 0,223 с), частота сетки fs = (N - 1) / (t_last - t_first);
  3. предобработка: детренд Тарвайнена со срезом 0,1 Гц (λ пересчитывается
     под fs волны), полосовой фильтр Баттерворта 4-го порядка 0,7–4,0 Гц
     нулевой фазы, z-нормировка;
  4. спектр: периодограмма всего окна с окном Ханна, дополнение нулями до
     степени двойки с шагом сетки не грубее 0,05 уд/мин;
  5. эталон — частота максимума спектра в полосе 0,7–4,0 Гц (42–240 уд/мин).
Дополнительно сохраняется доля мощности полосы в ±3 уд/мин от пика — мера
чистоты контактного сигнала в окне.

Старый эталон первой статьи (Уэлч, один 8-секундный сегмент) остаётся
только для сверки со статьями — см. scripts/eval_ubfc_subject.py.

    PYTHONPATH=src venv/bin/python3 benchmark/reference_full10s.py      # пересчитать файл
"""

from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.signal.preprocessing import preprocess_signal, tarvainen_lambda_for_cutoff  # noqa: E402

REFERENCE_FILE = REPO_ROOT / "results" / "reference" / "ubfc_ref_full10s.csv"
METHOD_VERSION = "full10s-v1"

WINDOW_S = 10.0
BAND_HZ = (0.7, 4.0)
FILTER_ORDER = 4
DETREND_CUTOFF_HZ = 0.1
FREQ_STEP_BPM = 0.05
PEAK_HALF_WIDTH_BPM = 3.0
MIN_SAMPLES = 16


def reference_full10s(t_ref: np.ndarray, wave: np.ndarray, window_end_s: float) -> dict | None:
    """Эталон окна [window_end_s - 10, window_end_s]; None, если отсчётов
    волны в окне слишком мало для оценки."""
    t_ref = np.asarray(t_ref, dtype=float)
    m = (t_ref >= window_end_s - WINDOW_S) & (t_ref <= window_end_s)
    tw, xw = t_ref[m], np.asarray(wave, dtype=float)[m]
    if len(tw) < MIN_SAMPLES:
        return None
    span = float(tw[-1] - tw[0])
    if not np.isfinite(span) or span <= 0:
        return None
    fs = (len(tw) - 1) / span
    x = np.interp(np.linspace(tw[0], tw[-1], len(tw)), tw, xw)
    y = preprocess_signal(
        x, fs, BAND_HZ[0], BAND_HZ[1], order=FILTER_ORDER, detrend_method="tarvainen",
        tarvainen_lambda=tarvainen_lambda_for_cutoff(DETREND_CUTOFF_HZ, fs), normalize_method="zscore",
    )
    if not np.all(np.isfinite(y)):
        return None
    nfft = int(2 ** np.ceil(np.log2(fs * 60.0 / FREQ_STEP_BPM)))
    win = np.hanning(len(y))
    power = np.abs(np.fft.rfft((y - y.mean()) * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    if not band.any():
        return None
    fb, pb = freqs[band], power[band]
    i = int(np.argmax(pb))
    near = np.abs(fb - fb[i]) * 60.0 <= PEAK_HALF_WIDTH_BPM
    return {
        "bpm_ref": float(fb[i] * 60.0),
        "fs_wave_hz": float(fs),
        "n_samples": int(len(tw)),
        "peak_fraction": float(pb[near].sum() / pb.sum()) if pb.sum() > 0 else 0.0,
    }


FIELDS = ["record_id", "window_end_ms", "bpm_ref", "fs_wave_hz", "n_samples", "peak_fraction", "method_version"]


@lru_cache(maxsize=1)
def load_reference() -> dict[tuple[str, int], float]:
    """(record_id, window_end_ms) -> эталон, уд/мин (только окна с эталоном)."""
    out: dict[tuple[str, int], float] = {}
    with REFERENCE_FILE.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["bpm_ref"] not in ("", "nan"):
                out[(r["record_id"], int(r["window_end_ms"]))] = float(r["bpm_ref"])
    return out


def build() -> None:
    sys.path.insert(0, str(REPO_ROOT / "benchmark"))
    import trace_cache as tc

    rows = []
    for path in sorted(tc.CACHE_DIR.glob("*.pkl"), key=lambda p: p.stem):
        trace = tc.load_trace(path.stem)
        for live in trace.live:
            ref = reference_full10s(trace.t_ref, trace.wave, live["timestamp_ms"] / 1000.0)
            rows.append({
                "record_id": trace.record_id, "window_end_ms": live["timestamp_ms"],
                "bpm_ref": "" if ref is None else f"{ref['bpm_ref']:.4f}",
                "fs_wave_hz": "" if ref is None else f"{ref['fs_wave_hz']:.4f}",
                "n_samples": "" if ref is None else ref["n_samples"],
                "peak_fraction": "" if ref is None else f"{ref['peak_fraction']:.4f}",
                "method_version": METHOD_VERSION,
            })
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE_FILE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["bpm_ref"] != "")
    print(f"окон: {len(rows)}, с эталоном: {n_ok}, файл: {REFERENCE_FILE}")


if __name__ == "__main__":
    build()
