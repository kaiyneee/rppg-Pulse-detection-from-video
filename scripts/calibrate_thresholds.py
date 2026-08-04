"""
Задача 10: калибровка порогов SQI по кривой "покрытие vs ошибка".

Делается ТОЛЬКО после задач 3 (референс/датасет) и 4-7 (починка вероятных
причин ложных отказов) — калибровать порог на коде, у которого гейтинг
ещё декоративен или ломается на 1/f-шуме (см. задачи 4/5), значит
калибровать под симптом, а не под причину.

Методология (ANSI/AAMI EC13 — стандарт для измерителей ЧСС: MAE <= 5 BPM):
для каждого кандидатного значения порога QualityConfig.<param> целиком
ПЕРЕПРОГОНЯЕТ RPPGPipeline по датасету (не переиспользует один прогон
постфактум — единственный способ корректно откалибровать компонентный
порог вроде min_spectral_snr_db, а не только итоговый sqi_score, для
которого это уже делает benchmark.evaluate.coverage_vs_error_curve) и
считает (покрытие, MAE по испытуемым). Рабочая точка — МАКСИМАЛЬНОЕ
покрытие при MAE <= --target-mae (по умолчанию 5.0 BPM).

ЧЕСТНАЯ ОГОВОРКА (см. тот же дисклеймер в benchmark/evaluate.py): в этой
среде разработки нет доступа к реальному датасету с ground truth. Поэтому:

  1. Реализована ПОЛНАЯ, рабочая процедура калибровки (sweep_threshold/
     select_operating_point/plot_calibration_curve) — готова к запуску на
     реальном датасете без единой правки кода.
  2. --synthetic прогоняет ТУ ЖЕ процедуру на синтетическом датасете
     (сгенерированном на лету во временную директорию в формате UBFC-rPPG)
     — доказывает, что код калибровки работает и показывает, как выглядит
     результат, но НЕ является реальной калибровкой и НЕ должно
     использоваться как основание менять QualityConfig в проде.
  3. QualityConfig НЕ изменяется автоматически этим скриптом — итоговые
     значения нужно перенести в config.py руками ПОСЛЕ прогона на реальном
     датасете, глядя на сохранённую кривую (см. --out-dir).

Использование:
    # На реальном датасете (после его получения по data use agreement):
    python scripts/calibrate_thresholds.py --root /data/UBFC-rPPG/DATASET_2 \\
        --param min_spectral_snr_db --values -2,-1,0,1,2,3,4,5,6,7,8 --target-mae 5.0

    # Демонстрация на синтетике (доказывает, что код работает):
    python scripts/calibrate_thresholds.py --synthetic --param min_overall_score_to_publish
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from rppg.config import PipelineConfig, QualityConfig
from benchmark.evaluate import (
    DatasetEvaluator,
    UBFCrPPGLoader,
    mae as mae_metric,
)

# ANSI/AAMI EC13 — стандарт точности для автоматических измерителей ЧСС
# (см. критерий приёмки задачи 10): MAE <= 5 BPM (или 10% от истинного
# значения, смотря что больше — здесь используется абсолютный порог как
# консервативный вариант, применимый во всём физиологическом диапазоне).
DEFAULT_TARGET_MAE_BPM = 5.0


@dataclass
class ThresholdPoint:
    value: float
    coverage: float          # доля ВСЕХ окон датасета с publishable=True при этом значении порога
    mae_bpm: float            # MAE по испытуемым (aggregate_per_subject) СРЕДИ publishable=True окон
    n_subjects_covered: int


def sweep_threshold(
    loader: UBFCrPPGLoader,
    root: Path,
    base_config: PipelineConfig,
    param_name: str,
    candidate_values: list[float],
    max_subjects: int | None = None,
) -> list[ThresholdPoint]:
    """Перепрогоняет RPPGPipeline ПО ВСЕМУ ДАТАСЕТУ ОТДЕЛЬНО для каждого
    кандидатного значения QualityConfig.<param_name> — единственный
    корректный способ откалибровать КОМПОНЕНТНЫЙ порог (min_spectral_snr_db,
    min_landmark_stability, harmonic_floor_ratio_threshold и т.п.): в
    отличие от итогового sqi_score, компонентные значения не сохраняются в
    WindowRecord (см. benchmark/evaluate.py), и постфактум пересчитать
    "что было бы, если бы порог был другим" по одному прогону нельзя —
    сами SQI-компоненты (harmonic_score, snr_score после штрафа) не
    зависят от порога, но is_reliable зависит, а публикуемость влияет на
    то, какие окна вообще считаются "прошедшими" для расчёта MAE."""
    points: list[ThresholdPoint] = []
    for value in candidate_values:
        config = copy.deepcopy(base_config)
        if not hasattr(config.quality, param_name):
            raise ValueError(f"QualityConfig не имеет параметра {param_name!r}")
        setattr(config.quality, param_name, value)

        evaluator = DatasetEvaluator(loader, config)
        result = evaluator.run(root, max_subjects=max_subjects)

        n_total = len(result.window_records)
        n_published = sum(1 for r in result.window_records if r.publishable)
        coverage = n_published / n_total if n_total else float("nan")

        points.append(ThresholdPoint(
            value=float(value),
            coverage=coverage,
            mae_bpm=result.published_subject_report.mae_mean,
            n_subjects_covered=result.published_subject_report.n_subjects,
        ))
        print(f"  {param_name}={value:<8.3f} покрытие={coverage:.3f}  "
              f"MAE={result.published_subject_report.mae_mean:.2f} BPM  "
              f"(испытуемых с покрытием: {result.published_subject_report.n_subjects})")
    return points


def select_operating_point(points: list[ThresholdPoint], target_mae_bpm: float = DEFAULT_TARGET_MAE_BPM) -> ThresholdPoint | None:
    """Рабочая точка (задача 10, критерий приёмки): МАКСИМАЛЬНОЕ покрытие
    среди точек с MAE <= target_mae_bpm (ANSI/AAMI EC13). None, если ни
    одно кандидатное значение порога не даёт MAE <= target_mae_bpm вообще —
    это ВАЖНЫЙ результат сам по себе (значит, проблема не в пороге SQI, а
    в чём-то более фундаментальном, см. задачи 4/5), а не повод понижать
    планку молча."""
    valid = [p for p in points if not np.isnan(p.mae_bpm) and p.mae_bpm <= target_mae_bpm]
    if not valid:
        return None
    return max(valid, key=lambda p: p.coverage)


def plot_calibration_curve(points: list[ThresholdPoint], param_name: str, target_mae_bpm: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOR_COVERAGE = "#2a78d6"
    COLOR_MAE = "#d03b3b"
    COLOR_TARGET = "#898781"
    INK_PRIMARY = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    GRIDLINE = "#e1e0d9"

    ordered = sorted(points, key=lambda p: p.value)
    values = [p.value for p in ordered]
    coverage = [p.coverage for p in ordered]
    mae_vals = [p.mae_bpm for p in ordered]

    fig, (ax_cov, ax_mae) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#fcfcfb")

    ax_cov.plot(values, coverage, marker="o", color=COLOR_COVERAGE)
    ax_cov.set_xlabel(param_name, color=INK_SECONDARY)
    ax_cov.set_ylabel("Покрытие (доля окон publishable=True)", color=INK_SECONDARY)
    ax_cov.set_title("Покрытие vs порог", color=INK_PRIMARY)
    ax_cov.grid(color=GRIDLINE, linewidth=0.7)

    ax_mae.plot(values, mae_vals, marker="o", color=COLOR_MAE)
    ax_mae.axhline(target_mae_bpm, color=COLOR_TARGET, linestyle="--",
                    label=f"целевой MAE (ANSI/AAMI EC13) = {target_mae_bpm:.1f} BPM")
    ax_mae.set_xlabel(param_name, color=INK_SECONDARY)
    ax_mae.set_ylabel("MAE, BPM (по испытуемым)", color=INK_SECONDARY)
    ax_mae.set_title("Ошибка vs порог", color=INK_PRIMARY)
    ax_mae.grid(color=GRIDLINE, linewidth=0.7)
    ax_mae.legend(fontsize=8, frameon=False)

    for ax in (ax_cov, ax_mae):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"Калибровка QualityConfig.{param_name} (задача 10)", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- #
# --synthetic: демонстрационный датасет (НЕ реальная калибровка, см. докстринг)
# --------------------------------------------------------------------------- #

def _write_synthetic_ubfc_dataset(root: Path, n_subjects: int = 5, duration_s: float = 18.0, fps: float = 30.0) -> None:
    """Пишет n_subjects синтетических 'испытуемых' в формате UBFCrPPGLoader
    (vid.avi + ground_truth.txt) во временную директорию — РАЗНЫЙ уровень
    шума на испытуемого, чтобы кривая покрытие-vs-MAE была нетривиальной
    (не плоской линией), как и на реальных данных, где качество сигнала
    отличается между людьми/условиями съёмки."""
    import cv2

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_signal_pipeline import _make_synthetic_landmarks_for_pipeline_test  # noqa: F401 (не используется здесь напрямую)

    rng_master = np.random.default_rng(0)
    n = int(fps * duration_s)
    h, w = 480, 640
    true_bpm_by_subject = rng_master.uniform(58.0, 95.0, n_subjects)
    noise_by_subject = np.linspace(0.5, 12.0, n_subjects)  # от почти чистого до сильно зашумлённого

    for i, (true_bpm, noise_std) in enumerate(zip(true_bpm_by_subject, noise_by_subject)):
        subject_dir = root / f"subject{i+1}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(100 + i)

        writer = cv2.VideoWriter(str(subject_dir / "vid.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        cy, cx = h // 2, w // 2
        ry, rx = int(0.28 * h), int(0.20 * w)
        yy, xx = np.ogrid[:h, :w]
        ellipse_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        f_hz = true_bpm / 60.0
        base_bgr = np.array([40.0, 60.0, 100.0])
        channel_amp = np.array([9.0, 15.0, 5.0])

        for frame_idx in range(n):
            t = frame_idx / fps
            pulse = np.sin(2 * np.pi * f_hz * t)
            color = base_bgr + channel_amp * pulse + rng.normal(0, noise_std, 3)
            frame = np.full((h, w, 3), (90, 70, 60), dtype=np.uint8)
            frame[ellipse_mask] = np.clip(color, 0, 255).astype(np.uint8)
            writer.write(frame)
        writer.release()

        t = np.arange(n) / fps
        bpm_series = np.full(n, true_bpm)
        with open(subject_dir / "ground_truth.txt", "w") as f:
            f.write(" ".join(["0.0"] * n) + "\n")
            f.write(" ".join(str(v) for v in bpm_series) + "\n")
            f.write(" ".join(str(v) for v in t) + "\n")
        print(f"  subject{i+1}: true_bpm={true_bpm:.1f}, noise_std={noise_std:.1f}")


def _patch_landmarker_for_synthetic_demo() -> None:
    """--synthetic использует эллиптическую 'кожу' без настоящего лица (см.
    _write_synthetic_ubfc_dataset) — как и в тестах пайплайна (см.
    test_signal_pipeline.py, test_negative_controls.py), MediaPipe не
    находит на ней лицо, поэтому landmarker.detect подменяется на
    фиксированную синтетическую детекцию ГЛОБАЛЬНО для этого процесса, а не
    только для одного объекта пайплайна — DatasetEvaluator создаёт свой
    RPPGPipeline (и свой FaceLandmarkerWrapper) заново на каждого
    испытуемого."""
    from rppg.face.landmarker import FaceLandmarkerWrapper, FaceFrameResult, HeadPose

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_signal_pipeline import _make_synthetic_landmarks_for_pipeline_test

    landmarks = _make_synthetic_landmarks_for_pipeline_test()
    fixed_result = FaceFrameResult(detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True)
    FaceLandmarkerWrapper.detect = lambda self, frame_bgr, ts: fixed_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default=None, help="Корень датасета (UBFC-rPPG DATASET_2)")
    parser.add_argument("--synthetic", action="store_true",
                         help="Демонстрация на синтетическом датасете вместо --root (см. ЧЕСТНУЮ ОГОВОРКУ в докстринге)")
    parser.add_argument("--param", type=str, default="min_overall_score_to_publish",
                         help="Имя поля QualityConfig, которое перебираем (например, min_spectral_snr_db)")
    parser.add_argument("--values", type=str, default=None,
                         help="Через запятую, например: -2,0,2,3,4,5,6,8. По умолчанию — 11 точек "
                              "линейно от текущего значения-3 до значения+5.")
    parser.add_argument("--target-mae", type=float, default=DEFAULT_TARGET_MAE_BPM,
                         help=f"Целевой MAE, BPM (ANSI/AAMI EC13 по умолчанию {DEFAULT_TARGET_MAE_BPM})")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--synthetic-subjects", type=int, default=5, help="Только с --synthetic: сколько 'испытуемых' сгенерировать")
    parser.add_argument("--synthetic-duration", type=float, default=18.0, help="Только с --synthetic: длительность видео каждого испытуемого, с")
    parser.add_argument("--out-dir", type=str, default="calibration_results")
    args = parser.parse_args()

    if not args.synthetic and args.root is None:
        raise SystemExit("Нужен либо --root (реальный датасет), либо --synthetic (демонстрация)")

    base_config = PipelineConfig()
    current_value = getattr(base_config.quality, args.param)
    if args.values:
        candidate_values = [float(v) for v in args.values.split(",")]
    else:
        candidate_values = list(np.linspace(current_value - 3.0, current_value + 5.0, 11))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = UBFCrPPGLoader()

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.synthetic:
            print("=" * 70)
            print("--synthetic: ДЕМОНСТРАЦИЯ процедуры калибровки на синтетическом")
            print("датасете. Результат НЕ является реальной калибровкой (см. докстринг")
            print("вверху файла) — переносить эти числа в QualityConfig нельзя.")
            print("=" * 70)
            root = Path(tmpdir)
            _write_synthetic_ubfc_dataset(root, n_subjects=args.synthetic_subjects, duration_s=args.synthetic_duration)
            _patch_landmarker_for_synthetic_demo()
        else:
            root = Path(args.root)

        print(f"\nПеребираю QualityConfig.{args.param} по значениям: "
              f"{[round(v, 3) for v in candidate_values]}")
        points = sweep_threshold(loader, root, base_config, args.param, candidate_values, max_subjects=args.max_subjects)

    curve_path = out_dir / f"calibration_{args.param}.png"
    plot_calibration_curve(points, args.param, args.target_mae, curve_path)
    print(f"\nКривая сохранена в {curve_path}")

    operating_point = select_operating_point(points, target_mae_bpm=args.target_mae)
    if operating_point is None:
        print(f"\nНИ ОДНО значение {args.param} не даёт MAE <= {args.target_mae} BPM — "
              "проблема не в этом пороге (см. задачи 4/5), понижать target-mae молча НЕ нужно.")
    else:
        print(f"\nРабочая точка (макс. покрытие при MAE <= {args.target_mae} BPM):")
        print(f"  {args.param} = {operating_point.value:.3f}")
        print(f"  покрытие = {operating_point.coverage:.3f}, MAE = {operating_point.mae_bpm:.2f} BPM, "
              f"испытуемых = {operating_point.n_subjects_covered}")
        if args.synthetic:
            print("\n(--synthetic: это значение НЕ переносится в config.py — см. докстринг)")
        else:
            print(f"\n  -> перенесите это значение в QualityConfig.{args.param} в src/rppg/config.py "
                  "и снимите пометку TODO(калибровка) в комментарии рядом с ним.")


if __name__ == "__main__":
    main()
