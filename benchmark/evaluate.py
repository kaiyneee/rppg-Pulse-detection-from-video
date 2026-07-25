"""
Метрики валидации и интерфейс оценки на датасетах
(пункты "Эксперименты" и "Для каждого датасета... метрики" ТЗ).

ЧЕСТНАЯ ОГОВОРКА (важно, чтобы не выдавать желаемое за действительное):
в этой среде разработки нет доступа к сети и, соответственно, нет доступа к
самим датасетам VIPL-HR/UBFC-rPPG/PURE/MMSE-HR (все они распространяются по
data use agreement и весят от сотен MB до десятков GB). Поэтому здесь
реализованы:
  (1) полностью рабочие и протестированные метрики (см. tests/ниже в этом
      файле — doctest-примеры),
  (2) единый интерфейс DatasetEvaluator для подключения любого датасета,
  (3) один конкретный загрузчик — UBFCrPPGLoader — для UBFC-rPPG DATASET_2,
      формат которого достаточно прост и хорошо задокументирован в
      литературе (Bobbia et al., 2019) чтобы реализовать его без доступа
      к самим файлам. Загрузчики для VIPL-HR/PURE/MMSE-HR оставлены как
      явные интерфейсы-заглушки (см. класс DatasetLoader) — их формат
      специфичен для каждого датасета и должен быть сверен с документацией
      после подписания data use agreement, а не реализован "по памяти".

Реальные числа MAE/RMSE/Bland-Altman по датасетам в этом ответе НЕ
приводятся — это были бы придуманные цифры. См.
docs/research_report.md, раздел 9, для полного протокола, который нужно
прогнать самостоятельно (или через Anthropic Research/аналог) после
получения доступа к данным.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, в тех же единицах, что и вход (обычно BPM).

    >>> round(mae(np.array([70, 80, 90]), np.array([72, 78, 95])), 3)
    3.0
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — сильнее штрафует редкие большие промахи
    (например, "октавные" ошибки на удвоенной/половинной частоте пульса),
    чем MAE.

    >>> round(rmse(np.array([70, 80, 90]), np.array([72, 78, 95])), 3)
    3.317
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, % — удобен для сравнения между
    датасетами/популяциями с разной средней ЧСС."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Возвращает (r, p_value). r близкий к 1 означает, что метод верно
    отслеживает МЕЖСУБЪЕКТНУЮ/МЕЖВРЕМЕННУЮ вариацию ЧСС, а не просто в среднем
    близок по абсолютной величине (что не гарантируется даже при низком MAE
    на узком диапазоне ЧСС)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r), float(p)


@dataclass
class BlandAltmanResult:
    mean_diff: float          # bias
    std_diff: float
    upper_loa: float          # верхняя граница согласия (mean + 1.96*std)
    lower_loa: float          # нижняя граница согласия (mean - 1.96*std)
    means: np.ndarray         # (y_true+y_pred)/2, для построения графика
    diffs: np.ndarray         # y_pred - y_true, для построения графика


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> BlandAltmanResult:
    """Bland-Altman анализ согласия. В отличие от корреляции/MAE явно
    показывает (а) систематическое смещение (bias) метода и (б) зависит ли
    разброс ошибки от абсолютного уровня ЧСС (веерность облака точек —
    признак того, что точность метода не постоянна по диапазону ЧСС, что
    особенно важно проверить, если система ПТСР будет видеть как состояние
    покоя, так и учащённый пульс при стрессовой реакции)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    diffs = y_pred - y_true
    means = (y_true + y_pred) / 2.0
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    return BlandAltmanResult(
        mean_diff=mean_diff,
        std_diff=std_diff,
        upper_loa=mean_diff + 1.96 * std_diff,
        lower_loa=mean_diff - 1.96 * std_diff,
        means=means,
        diffs=diffs,
    )


def plot_bland_altman(result: BlandAltmanResult, out_path: str, title: str = "Bland-Altman") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result.means, result.diffs, alpha=0.5, s=15)
    ax.axhline(result.mean_diff, color="black", linestyle="-", label=f"bias={result.mean_diff:.2f}")
    ax.axhline(result.upper_loa, color="red", linestyle="--", label=f"+1.96 SD={result.upper_loa:.2f}")
    ax.axhline(result.lower_loa, color="red", linestyle="--", label=f"-1.96 SD={result.lower_loa:.2f}")
    ax.set_xlabel("Среднее (rPPG, референс) / 2, BPM")
    ax.set_ylabel("Разность (rPPG - референс), BPM")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class EvaluationReport:
    dataset_name: str
    n_samples: int
    mae_bpm: float
    rmse_bpm: float
    mape_pct: float
    pearson_r: float
    pearson_p: float
    bland_altman: BlandAltmanResult

    def summary(self) -> str:
        return (
            f"[{self.dataset_name}] n={self.n_samples}  "
            f"MAE={self.mae_bpm:.2f} BPM  RMSE={self.rmse_bpm:.2f} BPM  "
            f"MAPE={self.mape_pct:.2f}%  Pearson r={self.pearson_r:.3f} (p={self.pearson_p:.1e})  "
            f"BA bias={self.bland_altman.mean_diff:.2f} BPM, "
            f"LoA=[{self.bland_altman.lower_loa:.2f}, {self.bland_altman.upper_loa:.2f}]"
        )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "unnamed") -> EvaluationReport:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]

    r, p = pearson_correlation(y_true, y_pred)
    return EvaluationReport(
        dataset_name=dataset_name,
        n_samples=len(y_true),
        mae_bpm=mae(y_true, y_pred),
        rmse_bpm=rmse(y_true, y_pred),
        mape_pct=mape(y_true, y_pred),
        pearson_r=r,
        pearson_p=p,
        bland_altman=bland_altman(y_true, y_pred),
    )


# --------------------------------------------------------------------------- #
# Интерфейс загрузчиков датасетов
# --------------------------------------------------------------------------- #

@dataclass
class GroundTruthSample:
    video_path: Path
    reference_bpm_timeseries: np.ndarray  # референсный BPM (по времени)
    reference_timestamps_s: np.ndarray
    subject_id: str


class DatasetLoader(ABC):
    """Общий интерфейс. Конкретные загрузчики реализуют parse-логику под
    формат распределения файлов конкретного датасета."""

    name: str = "base"

    @abstractmethod
    def list_samples(self, root: Path) -> list[GroundTruthSample]:
        raise NotImplementedError


class UBFCrPPGLoader(DatasetLoader):
    """
    UBFC-rPPG, DATASET_2 (Bobbia, Macwan, Benezeth, Mansouri, Dubois, 2019,
    Pattern Recognition Letters, "Unsupervised skin tissue segmentation for
    remote photoplethysmography").

    Ожидаемая структура (типовая для публично описанных копий датасета):
        root/
          subject1/vid.avi
          subject1/ground_truth.txt
          subject2/...

    ground_truth.txt: 3 строки чисел через пробел:
        строка 1 — PPG-волна (сырые отсчёты референсного пульсоксиметра),
        строка 2 — референсный BPM по времени,
        строка 3 — метки времени в секундах.

    ВАЖНО: перед использованием сверьте этот формат с README, который идёт
    в архиве датасета именно у вас — разные "зеркала"/версии архива могут
    незначительно отличаться по разделителю/числу строк.
    """

    name = "UBFC-rPPG"

    def list_samples(self, root: Path) -> list[GroundTruthSample]:
        samples = []
        for subject_dir in sorted(root.iterdir()):
            if not subject_dir.is_dir():
                continue
            gt_path = subject_dir / "ground_truth.txt"
            video_path = subject_dir / "vid.avi"
            if not (gt_path.exists() and video_path.exists()):
                continue

            with open(gt_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if len(lines) < 3:
                continue

            bpm_row = np.array([float(v) for v in lines[1].split()])
            time_row = np.array([float(v) for v in lines[2].split()])

            samples.append(GroundTruthSample(
                video_path=video_path,
                reference_bpm_timeseries=bpm_row,
                reference_timestamps_s=time_row,
                subject_id=subject_dir.name,
            ))
        return samples


class DatasetEvaluator:
    """
    Прогоняет RPPGPipeline по всем сэмплам датасета и агрегирует метрики.

    Использование (после того как реально скачан датасет и есть модель):

        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig

        loader = UBFCrPPGLoader()
        evaluator = DatasetEvaluator(loader, PipelineConfig())
        report = evaluator.run(Path("/data/UBFC-rPPG/DATASET_2"))
        print(report.summary())
        plot_bland_altman(report.bland_altman, "ubfc_bland_altman.png")
    """

    def __init__(self, loader: DatasetLoader, pipeline_config):
        self.loader = loader
        self.pipeline_config = pipeline_config

    def run(self, root: Path, max_subjects: int | None = None) -> EvaluationReport:
        import cv2
        from rppg.pipeline import RPPGPipeline

        samples = self.loader.list_samples(root)
        if max_subjects is not None:
            samples = samples[:max_subjects]

        all_true, all_pred = [], []

        for sample in samples:
            cap = cv2.VideoCapture(str(sample.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or self.pipeline_config.window.assumed_fps
            self.pipeline_config.window.assumed_fps = fps

            with RPPGPipeline(self.pipeline_config) as pipeline:
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    ts_ms = int(frame_idx / fps * 1000)
                    result = pipeline.process_frame(frame, ts_ms)
                    if result is not None and result.publishable:
                        ref_bpm = np.interp(
                            ts_ms / 1000.0,
                            sample.reference_timestamps_s,
                            sample.reference_bpm_timeseries,
                        )
                        all_true.append(ref_bpm)
                        all_pred.append(result.bpm)
                    frame_idx += 1
            cap.release()

        return evaluate(np.array(all_true), np.array(all_pred), dataset_name=self.loader.name)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
