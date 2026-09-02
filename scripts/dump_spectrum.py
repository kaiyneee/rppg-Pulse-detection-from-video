"""
Задача 42, часть 1: прямой взгляд на спектр — единственный способ напрямую
проверить рабочую гипотезу (истинный пульс на ~2.1 Гц/125 BPM, система
выбирает субгармонику ~1.05 Гц/62 BPM), а не гадать по косвенным признакам.

Строит и сохраняет Уэлч-спектр окна (0.5-4.5 Гц, ось BPM) с отмеченными
вертикальными линиями: выбранный пик, его удвоение (x2), его половина (/2).
Печатает численно мощность в этих трёх точках и их отношения.

PSD берётся из RPPGPipeline(debug=True).debug.psd_freqs/psd_values — это
ТОТ ЖЕ Уэлч-спектр лучшего по SQI ROI, который реально видел assess_quality
при вынесении решения об этом окне (не пересчитанный заново с другими
параметрами), и peak_freq_hz — реально выбранная пайплайном частота, а не
argmax по PSD "с нуля" (могла быть скорректирована continuity/rate-limit,
задачи 36/37 — это ВАЖНО для диагностики механизма B).

Использование:
    PYTHONPATH=src python3 scripts/dump_spectrum.py \\
        --video "я 20 приседаний.mov" --windows 9967 max-snr 26969 \\
        --out-dir spectrum_dumps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.config import PipelineConfig
from rppg.config_io import load_config
from rppg.pipeline import RPPGPipeline


def collect_results(video_path: str, model_path: str, config: PipelineConfig) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps
    config.face.model_asset_path = model_path

    results = []
    frame_idx = 0
    with RPPGPipeline(config, debug=True) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                results.append(result)
            frame_idx += 1
    cap.release()
    return results


def find_window(results: list, selector: str):
    with_debug = [
        r for r in results
        if r.debug is not None and (r.debug.bpm_freq_estimate is not None or r.debug.psd_freqs is not None)
    ]
    if not with_debug:
        raise RuntimeError("Ни одно окно не содержит PSD в debug (сигнал вырожден на всей записи?)")
    if selector == "max-snr":
        return max(with_debug, key=lambda r: r.debug.sqi.spectral_snr_db)
    target_ms = int(selector)
    return min(with_debug, key=lambda r: abs(r.timestamp_ms - target_ms))


def dump_spectrum(result, out_path: Path, band_plot_hz: tuple[float, float] = (0.5, 4.5)) -> None:
    debug = result.debug

    # ВАЖНО (найдено при первом прогоне этого скрипта, задача 42): debug.
    # psd_freqs/psd_values — это спектр из quality.compute_psd (nperseg=
    # len(signal)//2, посчитанный ДЛЯ SQI), а НЕ тот спектр, что реально
    # определил bpm — за это отвечает estimate_hr_welch со СВОИМ nperseg=
    # fps*8, другим окном усреднения. На близких по мощности конкурирующих
    # пиках (примерно такая ситуация и есть) эти два спектра могут выбрать
    # РАЗНЫЙ пик. debug.bpm_freq_estimate — это буквально FrequencyEstimate,
    # вернувшийся из estimate_hr для best_source, тот самый расчёт, что
    # породил result.bpm — используем ЕГО, а не SQI-спектр, иначе
    # диагностируем не то, что реально произошло.
    if debug.bpm_freq_estimate is not None:
        freqs = np.asarray(debug.bpm_freq_estimate.freqs_hz)
        psd = np.asarray(debug.bpm_freq_estimate.power)
        peak_hz = debug.bpm_freq_estimate.freq_hz
        spectrum_source = f"estimate_hr({debug.bpm_freq_estimate.method}) — реально определил bpm"
    else:
        freqs = np.asarray(debug.psd_freqs)
        psd = np.asarray(debug.psd_values)
        peak_hz = debug.peak_freq_hz
        spectrum_source = "quality.compute_psd (SQI) — bpm_freq_estimate недоступен, ФОЛБЭК"
    print(f"  источник спектра: {spectrum_source}")

    mask = (freqs >= band_plot_hz[0]) & (freqs <= band_plot_hz[1])
    freqs_plot, psd_plot = freqs[mask], psd[mask]

    double_hz = peak_hz * 2.0
    half_hz = peak_hz / 2.0

    def power_at(f0: float) -> float:
        if f0 < freqs.min() or f0 > freqs.max():
            return float("nan")
        return float(np.interp(f0, freqs, psd))

    p_peak = power_at(peak_hz)
    p_double = power_at(double_hz)
    p_half = power_at(half_hz)

    print(f"\n=== timestamp_ms={result.timestamp_ms}  bpm={result.bpm:.2f}  "
          f"spectral_snr_db={debug.sqi.spectral_snr_db:.2f} dB  best_source={debug.best_source} ===")
    print(f"  выбранный пик:  {peak_hz:.3f} Hz ({peak_hz * 60:.1f} BPM)  мощность={p_peak:.4g}")
    print(f"  удвоение (x2):  {double_hz:.3f} Hz ({double_hz * 60:.1f} BPM)  мощность={p_double:.4g}")
    print(f"  половина (/2):  {half_hz:.3f} Hz ({half_hz * 60:.1f} BPM)  мощность={p_half:.4g}")
    if np.isfinite(p_double) and p_peak > 0:
        print(f"  отношение удвоение/пик = {p_double / p_peak:.3f}")
    if np.isfinite(p_half) and p_peak > 0:
        print(f"  отношение половина/пик = {p_half / p_peak:.3f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(freqs_plot * 60.0, psd_plot, color="black", linewidth=1.2)
    ax.axvline(peak_hz * 60, color="tab:blue", linewidth=2,
               label=f"выбранный пик: {peak_hz * 60:.1f} BPM (P={p_peak:.3g})")
    if np.isfinite(p_double):
        ax.axvline(double_hz * 60, color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"удвоение (x2): {double_hz * 60:.1f} BPM (P={p_double:.3g})")
    if np.isfinite(p_half):
        ax.axvline(half_hz * 60, color="tab:green", linestyle="--", linewidth=1.5,
                   label=f"половина (/2): {half_hz * 60:.1f} BPM (P={p_half:.3g})")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Welch PSD")
    ax.set_title(
        f"timestamp={result.timestamp_ms} мс   итоговый bpm={result.bpm:.1f}   "
        f"SNR={debug.sqi.spectral_snr_db:.2f} dB   source={debug.best_source}"
    )
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  сохранено: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--windows", nargs="+", required=True,
        help="Список: 'max-snr' и/или timestamp в мс (берётся ближайшее реально посчитанное окно)",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else PipelineConfig()
    print(f"Обрабатываю {args.video} ...")
    results = collect_results(args.video, args.model, config)
    print(f"Всего окон с оценкой: {len(results)}")

    out_dir = Path(args.out_dir)
    for w in args.windows:
        result = find_window(results, w)
        name = "max_snr" if w == "max-snr" else f"{w}ms"
        dump_spectrum(result, out_dir / f"spectrum_{name}.png")


if __name__ == "__main__":
    main()
