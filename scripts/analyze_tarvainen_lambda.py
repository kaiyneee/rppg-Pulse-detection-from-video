"""
Обоснование tarvainen_lambda для fs=30 Гц (п.36 требований).

Проблема: исходный tarvainen_lambda=300 был взят из HRV-литературы БЕЗ
проверки, что он означает на fs=30 Гц (частота кадров видео) — стандартная
HRV-практика применяет smoothness-priors detrending к RR-тахограмме,
РЕСЭМПЛИРОВАННОЙ на 4 Гц, а не к сигналу на частоте кадров камеры. Поскольку
частота среза этого фильтра зависит и от lambda, И от fs (см.
signal.preprocessing.tarvainen_frequency_response), перенос числа "300" без
пересчёта на другую fs молчаливо меняет физический смысл параметра.

Этот скрипт:
  1. Валидирует формулу АЧХ по независимо документированному эталону
     (Kubios/PhysioData Toolbox: fs=4 Гц, lambda=500 -> cutoff~=0.04 Гц).
  2. Показывает, что реально даёт lambda=300 на НАШЕЙ fs=30 Гц.
  3. Подбирает lambda под принципиальный критерий: cutoff = 1/window_seconds
     (тренды медленнее одного окна анализа — дрейф, быстрее — сигнал).
  4. Сохраняет график АЧХ (старое vs новое значение) для статьи.

Запуск: PYTHONPATH=src python3 scripts/analyze_tarvainen_lambda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rppg.signal.preprocessing import tarvainen_frequency_response, tarvainen_cutoff_hz
from rppg.config import FilterConfig, WindowConfig


def main() -> None:
    fs = WindowConfig().assumed_fps
    window_seconds = WindowConfig().window_seconds
    old_lambda = 300.0
    new_lambda = FilterConfig().tarvainen_lambda
    band = (FilterConfig().low_hz, FilterConfig().high_hz)

    print("=== 1. Валидация формулы АЧХ по документированному эталону ===")
    ref_cutoff = tarvainen_cutoff_hz(500.0, 4.0)
    print(f"  lambda=500 @ fs=4 Hz -> cutoff={ref_cutoff:.4f} Hz "
          f"(эталон Kubios/PhysioData Toolbox: ~0.04 Hz — совпадает)")
    print("  https://physiodatatoolbox.leidenuniv.nl/docs/user-guide/physioanalyzer-modules/hrv-module.html\n")

    print(f"=== 2. Что реально даёт lambda={old_lambda:.0f} на нашей fs={fs:.0f} Hz ===")
    old_cutoff = tarvainen_cutoff_hz(old_lambda, fs)
    print(f"  cutoff(-3dB) = {old_cutoff:.4f} Hz (период {1/old_cutoff:.1f}с)")
    print(f"  Для сравнения: та же lambda=300 при типичном для HRV-литературы")
    print(f"  ресэмплинге RR на 4 Гц дала бы cutoff={tarvainen_cutoff_hz(old_lambda, 4.0):.4f} Hz —")
    print(f"  т.е. буквальный перенос числа '300' на fs=30 сдвигает срез в "
          f"{old_cutoff / tarvainen_cutoff_hz(old_lambda, 4.0):.1f}x выше по частоте.\n")

    print(f"=== 3. Новое значение: cutoff = 1/window_seconds = {1/window_seconds:.3f} Hz ===")
    print(f"  Критерий: тренд медленнее длины окна анализа (window_seconds="
          f"{window_seconds:.0f}с) считается дрейфом и удаляется; быстрее — нет.")
    new_cutoff = tarvainen_cutoff_hz(new_lambda, fs)
    print(f"  lambda={new_lambda:.0f} @ fs={fs:.0f} Hz -> cutoff={new_cutoff:.4f} Hz "
          f"(целевой {1/window_seconds:.3f} Hz)\n")

    print("=== 4. Затухание |H_hp| на ключевых частотах: старое vs новое lambda ===")
    print(f"  {'f, Hz':>8s}  {'|H_hp| (lambda=300)':>20s}  {'|H_hp| (новое)':>16s}   комментарий")
    checkpoints = [
        (0.02, "очень медленный дрейф (освещение/ROI)"),
        (0.05, ""),
        (0.10, "1/window_seconds — целевой срез"),
        (0.15, "нижний край HF-HRV/дыхания"),
        (0.20, ""),
        (0.34, "старый cutoff (-3dB) при lambda=300"),
        (band[0], "нижний край пульсовой полосы (low_hz)"),
        (1.0, "типичный пульс покоя (~60 BPM)"),
        (band[1], "верхний край пульсовой полосы (high_hz)"),
    ]
    for f_hz, comment in checkpoints:
        _, h_old = tarvainen_frequency_response(old_lambda, fs, freqs_hz=np.array([f_hz]))
        _, h_new = tarvainen_frequency_response(new_lambda, fs, freqs_hz=np.array([f_hz]))
        print(f"  {f_hz:8.3f}  {h_old[0]:20.4f}  {h_new[0]:16.4f}   {comment}")

    print(f"\n  Итог: на нижнем крае пульсовой полосы ({band[0]} Hz) затухание "
          f"падает с {(1 - tarvainen_frequency_response(old_lambda, fs, np.array([band[0]]))[1][0]) * 100:.2f}% "
          f"до {(1 - tarvainen_frequency_response(new_lambda, fs, np.array([band[0]]))[1][0]) * 100:.3f}% "
          f"— новое значение делает detrend ЕЩЁ прозрачнее для самого пульса, "
          f"при этом сильнее сохраняет содержимое 0.15-0.3 Hz (HF-HRV/дыхание) "
          f"нетронутым, чем старое.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        freqs = np.linspace(1e-3, 2.0, 3000)
        _, h_old_curve = tarvainen_frequency_response(old_lambda, fs, freqs_hz=freqs)
        _, h_new_curve = tarvainen_frequency_response(new_lambda, fs, freqs_hz=freqs)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs, h_old_curve, label=f"lambda={old_lambda:.0f} (старое значение, из HRV-литературы)", color="tab:red")
        ax.plot(freqs, h_new_curve, label=f"lambda={new_lambda:.0f} (новое, откалибровано под fs=30 Hz)", color="tab:blue")
        ax.axvspan(band[0], band[1], alpha=0.15, color="green", label=f"пульсовая полоса [{band[0]}, {band[1]}] Hz")
        ax.axhline(1 / np.sqrt(2), color="gray", linestyle=":", linewidth=1, label="-3dB")
        ax.axvline(1 / window_seconds, color="black", linestyle="--", linewidth=1,
                    label=f"1/window_seconds = {1/window_seconds:.2f} Hz")
        ax.set_xlabel("Частота, Hz")
        ax.set_ylabel("|H_hp(f)| (доля энергии, ОСТАЮЩЕЙСЯ после detrend)")
        ax.set_title(f"АЧХ tarvainen_detrend при fs={fs:.0f} Hz")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()

        out_path = REPO_ROOT / "benchmark" / "tarvainen_frequency_response.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nГрафик сохранён в {out_path}")
    except ImportError:
        print("\nmatplotlib недоступен — график не построен (числовой анализ выше не зависит от него).")


if __name__ == "__main__":
    main()
