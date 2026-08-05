"""
Анализатор структурированного JSONL-лога по окнам (см. structured_log.py,
п.43 требований) — отвечает на вопрос "почему именно эти окна не
опубликованы", а не просто печатает сводную статистику.

Все компоненты SQI (spectral_snr_db, harmonic_score, temporal_consistency,
cross_roi_agreement, landmark_stability, flicker_suspected) уже пишутся в
лог покадрово по окнам — этот скрипт их читает и раскладывает по причинам
отказа, вместо того чтобы require повторный прогон пайплайна.

Публикация (см. quality.assess_quality) требует ОДНОВРЕМЕННО:
    overall_score        >= QualityConfig.min_overall_score_to_publish
    spectral_snr_db       >= QualityConfig.min_spectral_snr_db
    not flicker_suspected

Три условия могут нарушаться одновременно на одном и том же окне — этот
скрипт считает как отдельные (маргинальные) частоты каждого условия, так и
полную комбинаторную разбивку (какая ИМЕННО комбинация встречается чаще
всего), т.к. "overall низкий, потому что низкий snr" и "overall низкий
несмотря на нормальный snr" — разные failure mode с разными починками.

Запуск:
    PYTHONPATH=src python3 scripts/analyze_log.py session.jsonl
    PYTHONPATH=src python3 scripts/analyze_log.py session.jsonl --out report.png
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.config import QualityConfig

# --- Палитра (см. dataviz skill: сериям — категориальный цвет по роли,
# порогам/гейтам — фиксированные status-цвета, никогда не смешивать смысл) --
COLOR_GOOD = "#0ca30c"       # status good — publishable=True
COLOR_CRITICAL = "#d03b3b"   # status critical — publishable=False / порог
COLOR_AUX = "#2a78d6"        # categorical slot 1 — last_valid_bpm (честная ступенька, не отдельное измерение)
COLOR_HIST_FILL = "#6da7ec"  # sequential blue, светлый шаг — заливка гистограмм
COLOR_HIST_EDGE = "#184f95"  # sequential blue, тёмный шаг — контур гистограмм
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

SQI_COMPONENTS = [
    ("sqi_spectral_snr_db", "spectral_snr_db, dB"),
    ("sqi_harmonic_score", "harmonic_score"),
    ("sqi_temporal_consistency", "temporal_consistency"),
    ("sqi_cross_roi_agreement", "cross_roi_agreement"),
    ("sqi_landmark_stability", "landmark_stability"),
]
PERCENTILES = [5, 25, 50, 75, 95]


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if df.empty:
        raise SystemExit(f"{path}: лог пуст (ни одно окно не было оценено).")
    return df


def print_publishable_fraction(df: pd.DataFrame) -> None:
    frac = float(df["publishable"].mean())
    n_total = len(df)
    n_pub = int(df["publishable"].sum())
    print("=== Доля публикуемых окон ===")
    print(f"  publishable=True: {n_pub}/{n_total} ({frac * 100:.1f}%)\n")


def failure_reason_breakdown(df: pd.DataFrame, qcfg: QualityConfig) -> pd.DataFrame:
    """Для каждого НЕопубликованного окна — какое(-ие) именно из 3 условий
    is_reliable не выполнено. Возвращает df с булевыми колонками
    overall_fail/snr_fail/flicker_fail только для publishable=False строк."""
    failed = df.loc[~df["publishable"]].copy()
    failed["overall_fail"] = failed["sqi_overall_score"] < qcfg.min_overall_score_to_publish
    failed["snr_fail"] = failed["sqi_spectral_snr_db"] < qcfg.min_spectral_snr_db
    failed["flicker_fail"] = failed["sqi_flicker_suspected"].astype(bool)
    return failed


def print_failure_histogram(failed: pd.DataFrame, qcfg: QualityConfig) -> None:
    n_failed = len(failed)
    print("=== Гистограмма причин отказа (только среди неопубликованных окон) ===")
    if n_failed == 0:
        print("  Неопубликованных окон нет.\n")
        return

    print(f"  Неопубликованных окон всего: {n_failed}\n")
    print("  По отдельности (маргинально, окно может входить в несколько строк):")
    print(f"    overall_score  < {qcfg.min_overall_score_to_publish:.2f}: "
          f"{int(failed['overall_fail'].sum())} ({failed['overall_fail'].mean() * 100:.1f}%)")
    print(f"    spectral_snr_db < {qcfg.min_spectral_snr_db:.1f} dB: "
          f"{int(failed['snr_fail'].sum())} ({failed['snr_fail'].mean() * 100:.1f}%)")
    print(f"    flicker_suspected: "
          f"{int(failed['flicker_fail'].sum())} ({failed['flicker_fail'].mean() * 100:.1f}%)")

    print("\n  По точным комбинациям (какая ИМЕННО причина или набор причин чаще всего):")
    combo_counts: Counter[tuple[bool, bool, bool]] = Counter(
        zip(failed["overall_fail"], failed["snr_fail"], failed["flicker_fail"])
    )
    for combo in sorted(product([False, True], repeat=3), key=lambda c: -combo_counts.get(c, 0)):
        count = combo_counts.get(combo, 0)
        if count == 0:
            continue
        overall_f, snr_f, flicker_f = combo
        parts = []
        if overall_f:
            parts.append("overall")
        if snr_f:
            parts.append("snr")
        if flicker_f:
            parts.append("flicker")
        name = " + ".join(parts) if parts else "(ни одно условие не нарушено?!)"
        print(f"    {name:<24s} {count:5d} ({count / n_failed * 100:5.1f}%)")
    print()


def print_component_percentiles(df: pd.DataFrame) -> None:
    print("=== Перцентили компонент SQI (5/25/50/75/95) ===")
    header = f"  {'компонента':<28s}" + "".join(f"{p:>9d}%" for p in PERCENTILES)
    print(header)
    for col, label in SQI_COMPONENTS:
        values = df[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n_dropped = len(values) - len(finite)
        if len(finite) == 0:
            print(f"  {label:<28s} (нет конечных значений)")
            continue
        pct = np.percentile(finite, PERCENTILES)
        row = f"  {label:<28s}" + "".join(f"{v:9.2f}" for v in pct)
        if n_dropped:
            row += f"   ({n_dropped} inf/nan исключено)"
        print(row)
    print()


def print_top_warnings(df: pd.DataFrame, top_n: int) -> None:
    print(f"=== Топ-{top_n} текстов warnings по частоте ===")
    all_warnings: list[str] = []
    for w in df["warnings"]:
        if isinstance(w, list):
            all_warnings.extend(w)
    if not all_warnings:
        print("  Warnings в логе нет.\n")
        return
    counts = Counter(all_warnings)
    for text, count in counts.most_common(top_n):
        short = text if len(text) <= 110 else text[:107] + "..."
        print(f"  {count:5d}x  {short}")
    print()


def _draw_component_histogram(ax, df: pd.DataFrame, col: str, label: str, threshold: float | None, threshold_label: str) -> None:
    values = df[col].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    ax.hist(finite, bins=30, color=COLOR_HIST_FILL, edgecolor=COLOR_HIST_EDGE, linewidth=0.8)
    if threshold is not None:
        ax.axvline(threshold, color=COLOR_CRITICAL, linestyle="--", linewidth=1.5,
                    label=f"{threshold_label} = {threshold:g}")
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title(label, fontsize=9, color=INK_PRIMARY)
    ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)


def _draw_failure_bar(ax, failed: pd.DataFrame) -> None:
    if len(failed) == 0:
        ax.text(0.5, 0.5, "Неопубликованных окон нет", ha="center", va="center",
                fontsize=9, color=INK_SECONDARY, transform=ax.transAxes)
        ax.axis("off")
        return
    combo_counts: Counter[tuple[bool, bool, bool]] = Counter(
        zip(failed["overall_fail"], failed["snr_fail"], failed["flicker_fail"])
    )
    rows = []
    for combo, count in combo_counts.items():
        overall_f, snr_f, flicker_f = combo
        parts = [n for n, f in (("overall", overall_f), ("snr", snr_f), ("flicker", flicker_f)) if f]
        rows.append((" + ".join(parts) or "?", count))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    ax.barh(names, counts, color=COLOR_HIST_FILL, edgecolor=COLOR_HIST_EDGE, linewidth=0.8)
    for y, c in enumerate(counts):
        ax.text(c, y, f" {c}", va="center", fontsize=7, color=INK_SECONDARY)
    ax.set_title("Причины отказа (комбинации)", fontsize=9, color=INK_PRIMARY)
    ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)


def _draw_top_warnings_bar(ax, df: pd.DataFrame, top_n: int) -> None:
    all_warnings: list[str] = []
    for w in df["warnings"]:
        if isinstance(w, list):
            all_warnings.extend(w)
    if not all_warnings:
        ax.text(0.5, 0.5, "Warnings в логе нет", ha="center", va="center",
                fontsize=9, color=INK_SECONDARY, transform=ax.transAxes)
        ax.axis("off")
        return
    counts = Counter(all_warnings).most_common(top_n)
    counts.reverse()  # самый частый — сверху при barh
    labels = [t if len(t) <= 42 else t[:39] + "..." for t, _ in counts]
    values = [c for _, c in counts]
    ax.barh(labels, values, color=COLOR_HIST_FILL, edgecolor=COLOR_HIST_EDGE, linewidth=0.8)
    for y, c in enumerate(values):
        ax.text(c, y, f" {c}", va="center", fontsize=7, color=INK_SECONDARY)
    ax.set_title(f"Топ-{top_n} warnings", fontsize=9, color=INK_PRIMARY)
    ax.tick_params(labelsize=6.5, colors=INK_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)


def _draw_timeseries(ax, df: pd.DataFrame) -> None:
    """bpm/publishable на одной оси (см. dataviz: никогда не дублировать
    ось) — публикуемость закодирована ЦВЕТОМ точки, а не второй шкалой.
    Задача 8: здесь больше нет best_effort_bpm (сглаженное "всегда
    доступное" число, фабриковавшееся даже при отсутствии сигнала — см.
    git-историю и PipelineConfig). Вместо него — last_valid_bpm: честная
    ступенчатая линия "последнее РЕАЛЬНО измеренное (publishable=True)
    значение", т.е. ровно то, что видит принимающая система между
    измерениями (см. PTSDPulseFeatures.last_valid_bpm/last_valid_bpm_age_s)."""
    t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    bpm = df["bpm"].to_numpy(dtype=float)
    publishable = df["publishable"].to_numpy(dtype=bool)
    last_valid_bpm = df["bpm"].where(df["publishable"]).ffill().to_numpy(dtype=float)

    ax.step(t, last_valid_bpm, color=COLOR_AUX, linewidth=1.5, linestyle="--", where="post",
            label="last_valid_bpm (последнее реальное измерение, честная ступенька)", zorder=2)
    ax.scatter(t[publishable], bpm[publishable], color=COLOR_GOOD, s=14, zorder=3,
               label="bpm, publishable=True")
    ax.scatter(t[~publishable], bpm[~publishable], color=COLOR_CRITICAL, s=14, zorder=3,
               label="bpm, publishable=False (не уходит в систему ПТСР)")

    ax.set_xlabel("время, с", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("BPM", fontsize=8, color=INK_SECONDARY)
    ax.set_title("BPM во времени: доверенное (цвет = publishable) vs last_valid_bpm", fontsize=9, color=INK_PRIMARY)
    ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)


def make_figure(df: pd.DataFrame, failed: pd.DataFrame, qcfg: QualityConfig, top_n: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 11), facecolor="#fcfcfb", constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.12, hspace=0.05, wspace=0.03)
    gs = fig.add_gridspec(3, 5, height_ratios=[1.1, 1.0, 1.1])

    ax_reasons = fig.add_subplot(gs[0, :2])
    _draw_failure_bar(ax_reasons, failed)

    ax_warnings = fig.add_subplot(gs[0, 2:])
    _draw_top_warnings_bar(ax_warnings, df, top_n)

    thresholds = {
        "sqi_spectral_snr_db": (qcfg.min_spectral_snr_db, "порог публикации"),
        "sqi_harmonic_score": (None, ""),
        "sqi_temporal_consistency": (0.5, "порог warning (quality.py)"),
        "sqi_cross_roi_agreement": (0.5, "порог warning (quality.py)"),
        "sqi_landmark_stability": (qcfg.min_landmark_stability, "порог warning"),
    }
    for i, (col, label) in enumerate(SQI_COMPONENTS):
        ax = fig.add_subplot(gs[1, i])
        thr, thr_label = thresholds[col]
        _draw_component_histogram(ax, df, col, label, thr, thr_label)

    ax_ts = fig.add_subplot(gs[2, :])
    _draw_timeseries(ax_ts, df)

    fig.suptitle("Анализ структурированного лога окон rPPG — почему окна не публикуются",
                 fontsize=13, color=INK_PRIMARY)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"График сохранён в {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_path", type=str, help="Путь к JSONL-логу (см. StructuredWindowLogger)")
    parser.add_argument("--out", type=str, default=None,
                         help="Путь к PNG с графиками (по умолчанию: <log>_analysis.png рядом с логом)")
    parser.add_argument("--top-warnings", type=int, default=10, help="Сколько строк warnings показывать (по умолчанию 10)")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Файл не найден: {log_path}")

    df = load_log(log_path)
    qcfg = QualityConfig()

    print(f"Лог: {log_path} ({len(df)} окон)\n")
    print_publishable_fraction(df)

    failed = failure_reason_breakdown(df, qcfg)
    print_failure_histogram(failed, qcfg)
    print_component_percentiles(df)
    print_top_warnings(df, args.top_warnings)

    try:
        out_path = Path(args.out) if args.out else log_path.with_name(log_path.stem + "_analysis.png")
        make_figure(df, failed, qcfg, args.top_warnings, out_path)
    except ImportError:
        print("matplotlib недоступен — графики не построены (текстовый анализ выше не зависит от него).")


if __name__ == "__main__":
    main()
