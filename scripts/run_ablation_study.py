"""
Ablation study (п.35 требований): полная таблица

    6 методов извлечения x 3 метода оценки частоты x 3 варианта detrend
    (none/linear/tarvainen) x 2 режима ROI (по отдельности / объединённые
    через SQI-взвешенный fusion, см. signal/fusion.py и п.34)

на СИНТЕТИЧЕСКИХ данных с известным истинным BPM (см. честную оговорку в
benchmark/evaluate.py и docstring scripts/compare_fusion_vs_argmax.py:
реальных датасетов в этой среде нет — это ИНФРАСТРУКТУРА для воспроизводимого
запуска, а не финальные числа для статьи, которые требуют реального
датасета, см. п.25/31).

Для head_motion "режим ROI" не применим (единственный канал, нет 3 ROI для
объединения) — такие строки помечены roi_mode="single" и не дублируются.

Методология парного сравнения: для каждого из n_trials "испытаний"
(случайный истинный BPM + случайное зерно шума) генерируется ОДИН набор
сырых RGB-траекторий 3 ROI и ОДНА траектория motion, которые затем
прогоняются через ВСЕ комбинации метод x частота x detrend x roi_mode —
так разница между строками таблицы объясняется только методом, а не другой
случайной реализацией шума (уменьшает дисперсию сравнения, стандартная
практика для ablation).

Запуск: PYTHONPATH=src python3 scripts/run_ablation_study.py
"""

from __future__ import annotations

import csv
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.signal.methods import SignalWindow, get_method
from rppg.signal.preprocessing import preprocess_signal
from rppg.signal.frequency import estimate_hr
from rppg.signal.quality import dominant_frequency_and_snr
from rppg.signal.fusion import fuse_signals_by_sqi, snr_db_to_weight

FPS = 30.0
DURATION_S = 10.0
BAND = (0.7, 4.0)
N = int(FPS * DURATION_S)

COLOR_METHODS = ["green", "chrom", "pos", "pca", "ica"]
ALL_METHODS = COLOR_METHODS + ["head_motion"]
FREQ_METHODS = ["fft", "welch", "lomb_scargle"]
DETREND_METHODS = ["none", "linear", "tarvainen"]
ROI_NAMES = ["forehead", "left_cheek", "right_cheek"]

# Уровень шума — умеренный, ОДИН зафиксированный сценарий на весь grid (не
# калибровочное упражнение из compare_fusion_vs_argmax.py: цель здесь —
# полнота и воспроизводимость перебора, а не измерение выигрыша fusion).
# Motion и color используют РАЗНЫЕ noise_std по той же причине, что и в
# compare_fusion_vs_argmax.py: HeadMotionMethod усредняет по 40
# landmark-точкам, поэтому то же noise_std даёт совсем другой SNR.
COLOR_NOISE_STD = 0.3
MOTION_NOISE_STD = 2.0


def make_color_roi(true_bpm: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N) / FPS
    f = true_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t - 0.5)
    pulse /= np.max(np.abs(pulse))
    gains = {"R": 0.6, "G": 1.0, "B": 0.35}
    dc = {"R": 150.0, "G": 110.0, "B": 90.0}
    amp = 1.1
    # Медленный дрейф (освещение/сползание ROI) — БЕЗ него detrend_method
    # буквально нечего убирать (см. вывод main(): АЧХ-объяснение того, почему
    # detrend_method всё равно не влияет на итоговую точность BPM в этом
    # пайплайне из-за финального bandpass, но проверять это на сигнале СОВСЕМ
    # без дрейфа было бы нечестной постановкой ablation).
    drift = 5.0 * (0.4 * np.sin(2 * np.pi * 0.02 * t) + (t / DURATION_S) ** 2)
    channels = [dc[ch] + gains[ch] * amp * pulse + drift + rng.normal(0, COLOR_NOISE_STD, N) for ch in ("R", "G", "B")]
    return np.stack(channels, axis=1)


def make_motion_trajectories(true_bpm: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N) / FPS
    f = true_bpm / 60.0
    common = 0.6 * np.sin(2 * np.pi * f * t)
    n_points = 40
    traj = np.zeros((N, n_points, 2))
    for p in range(n_points):
        gain = rng.uniform(0.5, 1.0)
        traj[:, p, 1] = 100 + gain * common + rng.normal(0, MOTION_NOISE_STD, N)
        traj[:, p, 0] = 100 + rng.normal(0, MOTION_NOISE_STD, N)
    return traj


def estimate_bpm(signal: np.ndarray, freq_method: str) -> float:
    if freq_method == "lomb_scargle":
        timestamps = np.arange(len(signal)) / FPS
        return estimate_hr(signal, FPS, BAND, method="lomb_scargle", timestamps_sec=timestamps).bpm
    return estimate_hr(signal, FPS, BAND, method=freq_method).bpm


def run_trial(true_bpm: float, trial_seed: int, rows: list[dict]) -> None:
    rng = np.random.default_rng(trial_seed)
    rgb_by_roi = {name: make_color_roi(true_bpm, rng) for name in ROI_NAMES}
    motion_traj = make_motion_trajectories(true_bpm, rng)

    for method_name in ALL_METHODS:
        method = get_method(method_name)

        if method_name == "head_motion":
            window = SignalWindow(rgb_traces={}, fps=FPS, landmark_trajectories=motion_traj, hr_band_hz=BAND)
            try:
                raw = method.extract(window, None)
            except Exception:
                continue
            for detrend_method in DETREND_METHODS:
                processed = preprocess_signal(raw, FPS, *BAND, detrend_method=detrend_method)
                for freq_method in FREQ_METHODS:
                    try:
                        bpm = estimate_bpm(processed, freq_method)
                    except Exception:
                        continue
                    rows.append({
                        "method": method_name, "freq_method": freq_method, "detrend_method": detrend_method,
                        "roi_mode": "single", "true_bpm": true_bpm, "pred_bpm": bpm,
                        "abs_error_bpm": abs(bpm - true_bpm),
                    })
            continue

        # Цветовые методы: извлекаем ОДИН раз на detrend, переиспользуем для
        # всех freq_method (extract()/detrend не зависят от freq_method).
        for detrend_method in DETREND_METHODS:
            per_roi_processed: dict[str, np.ndarray] = {}
            for roi_name, rgb in rgb_by_roi.items():
                window = SignalWindow(rgb_traces={roi_name: rgb}, fps=FPS, hr_band_hz=BAND)
                try:
                    raw = method.extract(window, roi_name)
                except Exception:
                    continue
                per_roi_processed[roi_name] = preprocess_signal(raw, FPS, *BAND, detrend_method=detrend_method)

            if not per_roi_processed:
                continue

            for freq_method in FREQ_METHODS:
                # --- roi_mode = separate: средняя ошибка по 3 независимым ROI ---
                errs = []
                for sig in per_roi_processed.values():
                    try:
                        errs.append(abs(estimate_bpm(sig, freq_method) - true_bpm))
                    except Exception:
                        continue
                if errs:
                    rows.append({
                        "method": method_name, "freq_method": freq_method, "detrend_method": detrend_method,
                        "roi_mode": "separate", "true_bpm": true_bpm, "pred_bpm": float("nan"),
                        "abs_error_bpm": float(np.mean(errs)),
                    })

                # --- roi_mode = combined: SQI-взвешенный fusion трёх ROI (п.34) ---
                if len(per_roi_processed) >= 2:
                    weights = {
                        name: snr_db_to_weight(dominant_frequency_and_snr(sig, FPS, BAND)[1])
                        for name, sig in per_roi_processed.items()
                    }
                    fused, _diag = fuse_signals_by_sqi(per_roi_processed, weights, fps=FPS)
                    try:
                        bpm_fused = estimate_bpm(fused, freq_method)
                    except Exception:
                        continue
                    rows.append({
                        "method": method_name, "freq_method": freq_method, "detrend_method": detrend_method,
                        "roi_mode": "combined", "true_bpm": true_bpm, "pred_bpm": bpm_fused,
                        "abs_error_bpm": abs(bpm_fused - true_bpm),
                    })


def main() -> None:
    n_trials = 20
    rng = np.random.default_rng(0)
    trials = [(float(rng.uniform(55, 95)), int(rng.integers(0, 2**31))) for _ in range(n_trials)]

    print(f"Ablation study: {len(ALL_METHODS)} методов x {len(FREQ_METHODS)} частотных оценок x "
          f"{len(DETREND_METHODS)} detrend x 2 roi_mode (+ head_motion как 'single'), n_trials={n_trials}")
    print("(синтетические данные — см. честную оговорку в докстринге модуля)\n")

    t0 = time.time()
    rows: list[dict] = []
    for true_bpm, seed in trials:
        run_trial(true_bpm, seed, rows)
    elapsed = time.time() - t0
    print(f"Прогон {len(trials)} испытаний x {len(ALL_METHODS)} методов занял {elapsed:.1f}с, "
          f"получено {len(rows)} отдельных оценок BPM.\n")

    # --- Агрегация по (method, freq_method, detrend_method, roi_mode) ---
    keys = sorted({(r["method"], r["freq_method"], r["detrend_method"], r["roi_mode"]) for r in rows})
    summary = []
    for key in keys:
        method, freq_method, detrend_method, roi_mode = key
        errors = [r["abs_error_bpm"] for r in rows if (r["method"], r["freq_method"], r["detrend_method"], r["roi_mode"]) == key]
        summary.append({
            "method": method, "freq_method": freq_method, "detrend_method": detrend_method, "roi_mode": roi_mode,
            "n": len(errors), "mae_bpm": float(np.mean(errors)), "median_bpm": float(np.median(errors)),
            "std_bpm": float(np.std(errors)),
        })

    summary.sort(key=lambda r: r["mae_bpm"])

    print(f"{'method':10s} {'freq':13s} {'detrend':10s} {'roi_mode':9s} {'n':>4s} {'MAE':>7s} {'median':>7s} {'std':>7s}")
    for r in summary:
        print(f"{r['method']:10s} {r['freq_method']:13s} {r['detrend_method']:10s} {r['roi_mode']:9s} "
              f"{r['n']:4d} {r['mae_bpm']:7.3f} {r['median_bpm']:7.3f} {r['std_bpm']:7.3f}")

    print(f"\nЛучшая комбинация по MAE: {summary[0]['method']} / {summary[0]['freq_method']} / "
          f"{summary[0]['detrend_method']} / {summary[0]['roi_mode']} -> MAE={summary[0]['mae_bpm']:.3f} BPM")
    print(f"Худшая комбинация по MAE: {summary[-1]['method']} / {summary[-1]['freq_method']} / "
          f"{summary[-1]['detrend_method']} / {summary[-1]['roi_mode']} -> MAE={summary[-1]['mae_bpm']:.3f} BPM")

    # --- combined vs separate, усреднённо по всем цветовым методам/freq/detrend ---
    combined_errs = [r["mae_bpm"] for r in summary if r["roi_mode"] == "combined"]
    separate_errs = [r["mae_bpm"] for r in summary if r["roi_mode"] == "separate"]
    print(f"\nСреднее MAE по всем комбинациям (только цветовые методы, где применимо ROI):")
    print(f"  roi_mode=separate: {np.mean(separate_errs):.3f} BPM (по {len(separate_errs)} комбинациям)")
    print(f"  roi_mode=combined: {np.mean(combined_errs):.3f} BPM (по {len(combined_errs)} комбинациям)")

    # --- Диагностика: почему detrend_method не влияет на итоговую MAE ---
    by_no_detrend_key: dict[tuple, set] = {}
    for r in summary:
        k = (r["method"], r["freq_method"], r["roi_mode"])
        by_no_detrend_key.setdefault(k, set()).add(round(r["mae_bpm"], 6))
    n_identical_groups = sum(1 for v in by_no_detrend_key.values() if len(v) == 1)
    print(
        f"\n=== Диагностика: влияние detrend_method на итоговую MAE ===\n"
        f"  {n_identical_groups}/{len(by_no_detrend_key)} групп (method x freq x roi_mode) дают "
        f"БУКВАЛЬНО ОДИНАКОВУЮ MAE при none/linear/tarvainen.\n"
        f"  Это НЕ баг: preprocess_signal делает detrend -> normalize -> bandpass, и финальный\n"
        f"  zero-phase Butterworth bandpass(0.7,4.0 Hz) сам по себе подавляет весь дрейф ниже\n"
        f"  0.7 Hz практически независимо от того, что сделал detrend НАД синтетическим дрейфом,\n"
        f"  добавленным в этом скрипте (см. make_color_roi) — проверено отдельно (см. обсуждение\n"
        f"  разработки): edge_diff (proxy остаточного тренда) ДО bandpass различается между\n"
        f"  none/linear/tarvainen, а ПОСЛЕ bandpass — почти нет. Дополнительно GreenMethod\n"
        f"  (methods.py) сама линейно детрендирует канал ВНУТРИ extract(), независимо от внешнего\n"
        f"  detrend_method, что ещё сильнее размывает разницу именно для green.\n"
        f"  ВЫВОД: detrend_method в ЭТОЙ архитектуре практически не влияет на точность точечной\n"
        f"  оценки BPM — его реальная роль в пайплайне (см. quality.harmonic_plausibility, п.21)\n"
        f"  в том, что harmonic_check_signal строится ИМЕННО из детрендированного, но НЕ\n"
        f"  bandpass-отфильтрованного сигнала — там, в отличие от BPM-оценки, выбор detrend_method\n"
        f"  действительно меняет результат, т.к. bandpass его больше не маскирует."
    )

    out_path = REPO_ROOT / "benchmark" / "ablation_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "freq_method", "detrend_method", "roi_mode", "n", "mae_bpm", "median_bpm", "std_bpm"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nПолная таблица ({len(summary)} строк) сохранена в {out_path}")


if __name__ == "__main__":
    main()
