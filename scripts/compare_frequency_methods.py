"""
Сравнение frequency_method=welch vs harmonic_sum (см. signal/frequency.py::
estimate_hr_harmonic_sum) на 6 видео задачи 23 — метрика ОКТАВНЫХ ОШИБОК.

ИСПРАВЛЕНО (задача 28): первая версия скрипта сравнивала оценку каждого
окна с МЕДИАНОЙ ПО ВСЕМУ ВИДЕО — при дрейфующем истинном пульсе (например,
у Вероники 60.48 -> 61.43 -> 62.37 -> ... 65.21 BPM за 30 окон) реальная
октавная ошибка возникает от ТЕКУЩЕГО ЛОКАЛЬНОГО значения, а не от
глобальной медианы: 2×медиана видео (139.8 BPM) не совпадала ни с одной
реальной ошибкой (они лежали около 122.86 и 184.29 BPM — это 2× и 3× от
ЛОКАЛЬНЫХ ~61.4 BPM), поэтому старая метрика находила 0% на всех видео —
не потому что ошибок не было, а потому что искала не там. Совпадения
типа 61.42893145161291 × 2 == 122.85786290322582 до 15-го знака в исходных
CSV — прямое доказательство, что это один и тот же бин спектра, умноженный.

Теперь для каждого окна i:
  1. Строится ЛОКАЛЬНАЯ медиана m_i по оценкам этого же метода/видео в
     окрестности ±15с от окна i, САМО окно i исключается.
  2. Окно помечается ошибкой ВВЕРХ, если b_i близко к 2·m_i ИЛИ 3·m_i.
  3. Окно помечается ошибкой ВНИЗ, если b_i близко к m_i / 2.
  4. Допуск — max(2 бина частотной сетки Welch, 3% от сравниваемого
     значения) — жёсткий допуск в один бин (~0.88 BPM) физически не может
     сработать на дрейфующей частоте, это и был источник нулей в задаче 27.
  5. Доли ошибок вверх/вниз считаются РАЗДЕЛЬНО — у welch и harmonic_sum
     это разные болезни (harmonic_sum известен склонностью выбирать f/2,
     см. задачу 29).

Плюс гистограмма log2(BPM) на видео (оба метода рядом) — октавная ошибка
вверх даёт спутниковый кластер точно на +1 (×2) или +1.585 (×3) от
основного кластера по оси log2, ошибка вниз — на -1. Видно глазом, без
всяких порогов.

harmonic_sum НЕ включён по умолчанию (PipelineConfig.frequency_method
остаётся welch).

Использование:
    PYTHONPATH=src python3 scripts/compare_frequency_methods.py
    PYTHONPATH=src python3 scripts/compare_frequency_methods.py --videos a.mp4 b.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rppg.config import PipelineConfig, FrequencyMethod
from rppg.pipeline import RPPGPipeline
from run_benchmark import DEFAULT_VIDEOS

METHODS = ["welch", "harmonic_sum"]
LOCAL_WINDOW_SECONDS = 15.0
MIN_RELATIVE_TOLERANCE = 0.03  # 3% от сравниваемого значения
MIN_BINS_TOLERANCE = 2  # минимум 2 бина частотной сетки


def _grid_step_bpm(fps: float, n_samples: int) -> float:
    """Один шаг частотной сетки Welch (fps/nfft, в BPM) — та же формула,
    что estimate_hr_welch/estimate_hr_harmonic_sum используют внутри (см.
    signal/frequency.py). НЕ путать с физической шириной лепестка (см.
    quality.resolvable_bpm_difference, задача 18) — здесь нужен именно шаг
    ДИСКРЕТНОЙ сетки, на которой сравниваются оценки."""
    nperseg = min(n_samples, max(64, int(fps * 8)))
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    return fps / nfft * 60.0


def run_video_bpm_series(video_path: str, model_path: str, method: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Возвращает (timestamps_s, bpm, fps) по всем окнам со status='ok'."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config = PipelineConfig()
    config.window.assumed_fps = fps
    config.face.model_asset_path = model_path
    config.frequency_method = FrequencyMethod(method)
    # ВАЖНО: continuity-проверка выбора пика (FrequencyConfig.continuity_
    # enabled, включена по умолчанию) — ОТДЕЛЬНЫЙ механизм, который
    # одинаково накладывается поверх ЛЮБОГО метода оценки частоты и активно
    # маскирует именно те октавные ошибки, которые этот скрипт должен
    # измерить "в чистом виде" (подтверждено численно: с continuity=True
    # welch на веронике даёт 0% ошибок, с continuity=False в тех же данных
    # обнаруживаются ТОЧНО те значения, что и в ручном разборе CSV —
    # 61.42893145161291, 122.85786290322582 и т.д., см. задачу 28).
    # Сравнение методов должно быть честным сравнением их СЫРОГО поведения.
    config.frequency.continuity_enabled = False

    timestamps_s: list[float] = []
    bpm_series: list[float] = []
    frame_idx = 0
    with RPPGPipeline(config) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None and result.status == "ok" and not np.isnan(result.bpm):
                timestamps_s.append(timestamp_ms / 1000.0)
                bpm_series.append(result.bpm)
            frame_idx += 1
    cap.release()
    return np.array(timestamps_s), np.array(bpm_series), fps


def local_median_series(timestamps_s: np.ndarray, bpm: np.ndarray, window_s: float = LOCAL_WINDOW_SECONDS) -> np.ndarray:
    """Задача 28: локальная медиана m_i по ОКРЕСТНОСТИ ±window_s секунд
    вокруг окна i, само окно i ИСКЛЮЧЕНО из своей же локальной медианы
    (иначе окно тривиально "подтверждает само себя"). NaN, если в
    окрестности не нашлось других окон (край записи при малом числе
    valid-окон)."""
    n = len(bpm)
    medians = np.full(n, np.nan)
    idx = np.arange(n)
    for i in range(n):
        neighbor_mask = (np.abs(timestamps_s - timestamps_s[i]) <= window_s) & (idx != i)
        if neighbor_mask.any():
            medians[i] = float(np.median(bpm[neighbor_mask]))
    return medians


def _tolerance(target: float, grid_step_bpm: float) -> float:
    return max(MIN_BINS_TOLERANCE * grid_step_bpm, MIN_RELATIVE_TOLERANCE * abs(target))


def octave_error_labels(
    bpm: np.ndarray, local_median: np.ndarray, grid_step_bpm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (is_error_up, is_error_down) — булевы массивы по окнам.
    up: b_i близко к 2·m_i ИЛИ 3·m_i (пик найден на кратной гармонике
    истинной частоты). down: b_i близко к m_i/2 (истинный пик найден как
    его собственная субгармоника — известная болезнь harmonic_sum, см.
    задачу 29)."""
    n = len(bpm)
    up = np.zeros(n, dtype=bool)
    down = np.zeros(n, dtype=bool)
    for i in range(n):
        m = local_median[i]
        if np.isnan(m) or m <= 0:
            continue
        target_2x, target_3x, target_half = 2.0 * m, 3.0 * m, m / 2.0
        if abs(bpm[i] - target_2x) <= _tolerance(target_2x, grid_step_bpm):
            up[i] = True
        elif abs(bpm[i] - target_3x) <= _tolerance(target_3x, grid_step_bpm):
            up[i] = True
        if abs(bpm[i] - target_half) <= _tolerance(target_half, grid_step_bpm):
            down[i] = True
    return up, down


def _plot_log2_histograms(video_name: str, series_by_method: dict[str, np.ndarray], out_path: Path) -> None:
    """Задача 28: гистограмма log2(BPM) — октавная ошибка вверх (×2) даёт
    спутниковый кластер РОВНО на +1 от основного по оси log2 (×3 -> +1.585),
    ошибка вниз (÷2) -> -1. Видно глазом, без всяких порогов."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOR_FILL = "#6da7ec"
    COLOR_EDGE = "#184f95"
    INK_PRIMARY = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    GRIDLINE = "#e1e0d9"

    methods = list(series_by_method.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5), facecolor="#fcfcfb")
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        bpm = series_by_method[method]
        if len(bpm) == 0:
            ax.text(0.5, 0.5, "нет окон", ha="center", va="center", transform=ax.transAxes)
            continue
        log2_bpm = np.log2(bpm)
        ax.hist(log2_bpm, bins=30, color=COLOR_FILL, edgecolor=COLOR_EDGE, linewidth=0.8)
        median_log2 = float(np.median(log2_bpm))
        for offset, label in [(1.0, "×2"), (np.log2(3), "×3"), (-1.0, "÷2")]:
            ax.axvline(median_log2 + offset, color="#d03b3b", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text(median_log2 + offset, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1, label,
                     fontsize=7, color="#d03b3b", ha="center", va="bottom")
        ax.set_title(f"{method}", fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("log2(BPM)", fontsize=8, color=INK_SECONDARY)
        ax.tick_params(labelsize=7, colors=INK_SECONDARY)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"log2(BPM) — {video_name} (задача 28: спутники на ±1/±1.585 = октавные ошибки)",
                 fontsize=11, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--out-dir", type=str, default="compare_frequency_methods_results")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos]
    missing = [v for v in videos if not v.exists()]
    if missing:
        raise SystemExit(
            "Видео не найдены:\n" + "\n".join(f"  {v}" for v in missing) +
            "\n\nПередайте реальные пути через --videos."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = (f"{'видео':<45s}{'метод':<15s}{'n':>6s}{'доля ошибок ×2/×3':>20s}"
              f"{'доля ошибок ÷2':>16s}{'доля любых':>12s}")
    print(header)

    for video in videos:
        series_by_method: dict[str, np.ndarray] = {}
        for method in METHODS:
            timestamps_s, bpm, fps = run_video_bpm_series(str(video), args.model, method)
            series_by_method[method] = bpm
            if len(bpm) == 0:
                print(f"{video.stem:<45s}{method:<15s}{'0':>6s}{'—':>20s}{'—':>16s}{'—':>12s}")
                continue

            n_samples_approx = int(10.0 * fps)  # WindowConfig.window_seconds по умолчанию — 10с
            grid_step = _grid_step_bpm(fps, n_samples_approx)

            local_medians = local_median_series(timestamps_s, bpm)
            up, down = octave_error_labels(bpm, local_medians, grid_step)

            frac_up = float(up.sum()) / len(bpm)
            frac_down = float(down.sum()) / len(bpm)
            frac_any = float((up | down).sum()) / len(bpm)

            print(f"{video.stem:<45s}{method:<15s}{len(bpm):>6d}{frac_up:>20.1%}{frac_down:>16.1%}{frac_any:>12.1%}")

        hist_path = out_dir / f"log2hist_{video.stem}.png"
        _plot_log2_histograms(video.stem, series_by_method, hist_path)
        print(f"  -> гистограмма log2(BPM) сохранена в {hist_path}")


if __name__ == "__main__":
    main()
