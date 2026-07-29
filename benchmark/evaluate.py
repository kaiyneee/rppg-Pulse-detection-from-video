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
from dataclasses import dataclass, field
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


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient (Lin, 1989, Biometrics) —
    п.27 требований: "строже Пирсона". Pearson r инвариантен к линейному
    сдвигу/масштабу — метод, систематически завышающий BPM на 10 (bias), даёт
    r=1.0, если завышение постоянно. CCC дополнительно штрафует именно за
    отклонение от линии y=x (совершенного согласия), а не только за
    линейную связность, поэтому чувствителен и к bias, и к разнице
    дисперсий true/pred, которые Pearson r не видит.

    >>> round(concordance_correlation_coefficient(np.array([70, 80, 90]), np.array([72, 78, 95])), 3)
    0.933
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(2.0 * covariance / denom)


def reference_relative_snr_db(
    signal: np.ndarray,
    fps: float,
    band_hz: tuple[float, float],
    reference_hz: float,
    tol_hz: float = 0.1,
) -> float:
    """
    SNR ОЦЕНЁННОГО СИГНАЛА ОТНОСИТЕЛЬНО РЕФЕРЕНСНОЙ (истинной) частоты пульса
    (п.27 требований) — определение de Haan & Jeanne (2013) / Wang et al.
    (2016, POS): энергия в узких полосах вокруг референсной частоты И её
    2-й гармоники (систолический подъём делает пульсовую волну
    несинусоидальной — вторая гармоника несёт реальную физиологическую
    энергию, а не шум) против остальной энергии рабочей полосы.

    ВАЖНОЕ ОТЛИЧИЕ от quality.dominant_frequency_and_snr: та функция не
    требует ground truth и центрирована на СОБСТВЕННОМ пике сигнала —
    используется для онлайн-гейтинга (система не знает истинного BPM).
    Эта функция для ВАЛИДАЦИИ, где референс есть: сигнал может быть
    узкополосным и уверенным (высокий dominant_frequency_and_snr), но
    сосредоточенным на ЛОЖНОЙ частоте (например, harmonic confusion, см.
    quality.harmonic_plausibility) — тогда эта метрика, в отличие от
    dominant_frequency_and_snr, корректно покажет низкое согласие с
    референсом.
    """
    from scipy.signal import welch

    signal = np.asarray(signal, dtype=float)
    if len(signal) < 8 or np.std(signal) < 1e-10 or reference_hz <= 0:
        return -np.inf

    nperseg = min(len(signal), 256)
    nfft = max(2048, int(2 ** np.ceil(np.log2(nperseg))) * 4)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, nfft=nfft)

    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not band_mask.any():
        return -np.inf
    band_freqs, band_psd = freqs[band_mask], psd[band_mask]

    signal_mask = (np.abs(band_freqs - reference_hz) <= tol_hz) | (
        np.abs(band_freqs - 2.0 * reference_hz) <= tol_hz
    )
    signal_power = float(np.sum(band_psd[signal_mask]))
    total_power = float(np.sum(band_psd))
    if signal_power <= 1e-12:
        return -np.inf
    noise_power = max(total_power - signal_power, 1e-12)
    return 10.0 * np.log10(signal_power / noise_power)


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Непараметрический bootstrap-доверительный интервал (п.28 требований).

    КРИТИЧНО для корректности: values должен быть массивом ОДНО-ЗНАЧЕНИЕ-НА-
    ИСПЫТУЕМОГО (например, MAE каждого отдельного человека, см.
    aggregate_per_subject), а НЕ массивом отдельных окон. Ресэмплирование по
    окнам вместо испытуемых давало бы заниженный (слишком узкий)
    доверительный интервал, потому что окна одного человека сильно
    коррелированы между собой (не независимые наблюдения) — классическая
    псевдорепликация, та же ошибка, что и в пулинге MAE по окнам (см.
    докстринг evaluate() и aggregate_per_subject)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        v = float(statistic(values))
        return v, v

    rng = np.random.default_rng(seed)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        resample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(resample)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_stats, alpha))
    hi = float(np.quantile(boot_stats, 1.0 - alpha))
    return lo, hi


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
    """ВНИМАНИЕ (п.28 требований): если y_true/y_pred, переданные в evaluate(),
    объединяют окна НЕСКОЛЬКИХ испытуемых — это ПСЕВДОРЕПЛИКАЦИЯ (человек с
    длинным видео весит в метрике больше, чем человек с коротким, пропорционально
    числу его окон, а не как "один испытуемый = один голос"). Для валидной
    межсубъектной статистики используйте aggregate_per_subject() /
    SubjectAggregateReport, которые считают метрику НА ЧЕЛОВЕКА, а затем
    среднее±std и bootstrap CI ПО ЛЮДЯМ. Этот класс/функция оставлены для
    быстрых внутрисубъектных проверок (например, все окна ОДНОГО видео) и как
    строительный блок для aggregate_per_subject, а не как финальная метрика
    статьи для многосубъектной выборки."""

    dataset_name: str
    n_samples: int
    mae_bpm: float
    rmse_bpm: float
    mape_pct: float
    pearson_r: float
    pearson_p: float
    ccc: float
    bland_altman: BlandAltmanResult

    def summary(self) -> str:
        return (
            f"[{self.dataset_name}] n={self.n_samples}  "
            f"MAE={self.mae_bpm:.2f} BPM  RMSE={self.rmse_bpm:.2f} BPM  "
            f"MAPE={self.mape_pct:.2f}%  Pearson r={self.pearson_r:.3f} (p={self.pearson_p:.1e})  "
            f"CCC={self.ccc:.3f}  "
            f"BA bias={self.bland_altman.mean_diff:.2f} BPM, "
            f"LoA=[{self.bland_altman.lower_loa:.2f}, {self.bland_altman.upper_loa:.2f}]"
        )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "unnamed") -> EvaluationReport:
    """Пулинг-метрики на ПЕРЕДАННОМ наборе пар (true, pred) без разбивки по
    испытуемым — см. предупреждение о псевдорепликации в docstring
    EvaluationReport, если вызываете это на нескольких испытуемых сразу."""
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
        ccc=concordance_correlation_coefficient(y_true, y_pred),
        bland_altman=bland_altman(y_true, y_pred),
    )


# --------------------------------------------------------------------------- #
# Агрегация ПО ИСПЫТУЕМЫМ, а не по окнам (п.28 требований)
# --------------------------------------------------------------------------- #

@dataclass
class WindowRecord:
    """Одна оценка BPM за одно окно RPPGPipeline — минимальная единица
    для coverage_vs_error_curve (п.30) и stratified_report (п.29).
    Включает НЕПУБЛИКУЕМЫЕ окна тоже (publishable=False) — они нужны, чтобы
    прогонять весь диапазон порогов SQI при построении кривой "покрытие vs
    ошибка", а не только текущий порог по умолчанию."""

    subject_id: str
    timestamp_ms: int
    true_bpm: float
    pred_bpm: float
    sqi_score: float
    publishable: bool


@dataclass
class PerSubjectResult:
    subject_id: str
    n_windows: int
    mae_bpm: float
    rmse_bpm: float
    mape_pct: float
    metadata: dict = field(default_factory=dict)


@dataclass
class SubjectAggregateReport:
    """Метрики, агрегированные СНАЧАЛА по испытуемому, ПОТОМ по выборке
    испытуемых (mean±std + bootstrap CI ПО ЛЮДЯМ, не по окнам) — правильная
    альтернатива пулингу всех окон всех людей в evaluate()/EvaluationReport
    (п.28 требований)."""

    dataset_name: str
    n_subjects: int
    mae_mean: float
    mae_std: float
    mae_ci95: tuple[float, float]
    rmse_mean: float
    rmse_std: float
    rmse_ci95: tuple[float, float]
    per_subject: list[PerSubjectResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.dataset_name}] n_subjects={self.n_subjects}  "
            f"MAE={self.mae_mean:.2f}±{self.mae_std:.2f} BPM "
            f"(95% CI [{self.mae_ci95[0]:.2f}, {self.mae_ci95[1]:.2f}])  "
            f"RMSE={self.rmse_mean:.2f}±{self.rmse_std:.2f} BPM "
            f"(95% CI [{self.rmse_ci95[0]:.2f}, {self.rmse_ci95[1]:.2f}])"
        )


def aggregate_per_subject(
    records: list[WindowRecord],
    dataset_name: str = "unnamed",
    metadata_by_subject: dict[str, dict] | None = None,
) -> SubjectAggregateReport:
    """Считает MAE/RMSE/MAPE ОТДЕЛЬНО для каждого subject_id, затем берёт
    среднее±std и bootstrap 95% CI ПО ИСПЫТУЕМЫМ (см. bootstrap_ci) — это и
    есть правильный протокол п.28 требований, в противовес пулингу всех окон
    (см. EvaluationReport)."""
    by_subject: dict[str, list[WindowRecord]] = {}
    for r in records:
        by_subject.setdefault(r.subject_id, []).append(r)

    per_subject: list[PerSubjectResult] = []
    for subject_id, subj_records in by_subject.items():
        true = np.array([r.true_bpm for r in subj_records])
        pred = np.array([r.pred_bpm for r in subj_records])
        meta = (metadata_by_subject or {}).get(subject_id, {})
        per_subject.append(PerSubjectResult(
            subject_id=subject_id,
            n_windows=len(subj_records),
            mae_bpm=mae(true, pred),
            rmse_bpm=rmse(true, pred),
            mape_pct=mape(true, pred),
            metadata=meta,
        ))

    mae_values = np.array([p.mae_bpm for p in per_subject])
    rmse_values = np.array([p.rmse_bpm for p in per_subject])
    mae_lo, mae_hi = bootstrap_ci(mae_values)
    rmse_lo, rmse_hi = bootstrap_ci(rmse_values)

    return SubjectAggregateReport(
        dataset_name=dataset_name,
        n_subjects=len(per_subject),
        mae_mean=float(np.mean(mae_values)) if len(mae_values) else float("nan"),
        mae_std=float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
        mae_ci95=(mae_lo, mae_hi),
        rmse_mean=float(np.mean(rmse_values)) if len(rmse_values) else float("nan"),
        rmse_std=float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0,
        rmse_ci95=(rmse_lo, rmse_hi),
        per_subject=per_subject,
    )


def stratified_report(
    records: list[WindowRecord],
    metadata_by_subject: dict[str, dict],
    stratum_key: str,
    dataset_name: str = "unnamed",
) -> dict[str, SubjectAggregateReport]:
    """Стратифицированный анализ (п.29 требований): разбивает записи по
    значению metadata_by_subject[subject_id][stratum_key] (например,
    'skin_tone' — см. benchmark/skin_tone.py для объективной ITA-based
    оценки без ручной разметки, 'motion_level', 'illumination', 'glasses',
    'facial_hair', 'sex', 'age_group') и для КАЖДОЙ группы считает
    SubjectAggregateReport — то есть агрегация по испытуемым ВНУТРИ каждой
    страты (см. aggregate_per_subject, п.28), а не по окнам.

    Испытуемые без значения stratum_key в metadata_by_subject молча
    исключаются из результата (а не попадают в фиктивную группу "None") —
    вызывающий код должен явно проверить len(результат) против числа
    испытуемых, если это важно."""
    groups: dict[str, list[WindowRecord]] = {}
    for r in records:
        stratum_value = metadata_by_subject.get(r.subject_id, {}).get(stratum_key)
        if stratum_value is None:
            continue
        groups.setdefault(str(stratum_value), []).append(r)

    return {
        value: aggregate_per_subject(
            group_records,
            dataset_name=f"{dataset_name}[{stratum_key}={value}]",
            metadata_by_subject=metadata_by_subject,
        )
        for value, group_records in groups.items()
    }


# --------------------------------------------------------------------------- #
# Кривая "покрытие vs ошибка" для SQI (п.30 требований)
# --------------------------------------------------------------------------- #

@dataclass
class CoverageErrorPoint:
    sqi_threshold: float
    coverage: float          # доля ВСЕХ окон (по всем испытуемым), прошедших порог
    mae_mean: float          # среднее MAE ПО ИСПЫТУЕМЫМ на прошедших порог окнах
    mae_ci95: tuple[float, float]
    n_subjects_covered: int  # сколько испытуемых имеют хотя бы одно окно выше порога


@dataclass
class CoverageErrorCurve:
    dataset_name: str
    points: list[CoverageErrorPoint]

    def is_monotonic_nonincreasing(self, tol: float = 1e-6) -> bool:
        """True, если MAE не возрастает по мере роста порога SQI (падения
        покрытия) — п.30: "если кривая монотонно убывает — вы доказали, что
        ваш SQI работает". Точки с NaN MAE (нет ни одного испытуемого,
        прошедшего порог) исключаются из проверки."""
        ordered = sorted(self.points, key=lambda p: p.sqi_threshold)
        maes = [p.mae_mean for p in ordered if not np.isnan(p.mae_mean)]
        return all(maes[i] >= maes[i + 1] - tol for i in range(len(maes) - 1))


def coverage_vs_error_curve(
    records: list[WindowRecord],
    dataset_name: str = "unnamed",
    thresholds: np.ndarray | None = None,
) -> CoverageErrorCurve:
    """
    Ключевой график для доказательства того, что SQI-гейтинг — реальный
    вклад, а не декорация (п.30 требований). Для каждого порога SQI:

      coverage = доля ВСЕХ окон (по всем испытуемым), чей sqi_score >= порога;
      MAE считается ПО ИСПЫТУЕМЫМ (см. aggregate_per_subject, п.28) — только
      среди окон ЭТОГО испытуемого, прошедших порог; испытуемые без единого
      прошедшего окна на данном пороге ИСКЛЮЧАЮТСЯ из среднего (не считаются
      как MAE=0 и не считаются как MAE=NaN-обнуляющие всю точку).

    Если результирующая кривая монотонно не возрастает (см.
    CoverageErrorCurve.is_monotonic_nonincreasing) при росте порога — SQI
    действительно коррелирует с реальной точностью на РЕФЕРЕНСНЫХ данных,
    а не только "выглядит разумно" по своей внутренней конструкции.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    total_n = len(records)
    by_subject: dict[str, list[WindowRecord]] = {}
    for r in records:
        by_subject.setdefault(r.subject_id, []).append(r)

    points: list[CoverageErrorPoint] = []
    for thr in thresholds:
        kept = [r for r in records if r.sqi_score >= thr]
        coverage = (len(kept) / total_n) if total_n > 0 else float("nan")

        subject_maes = []
        for subj_records in by_subject.values():
            subj_kept = [r for r in subj_records if r.sqi_score >= thr]
            if not subj_kept:
                continue
            true = np.array([r.true_bpm for r in subj_kept])
            pred = np.array([r.pred_bpm for r in subj_kept])
            subject_maes.append(mae(true, pred))

        if subject_maes:
            mae_mean = float(np.mean(subject_maes))
            mae_lo, mae_hi = bootstrap_ci(np.array(subject_maes))
        else:
            mae_mean, mae_lo, mae_hi = float("nan"), float("nan"), float("nan")

        points.append(CoverageErrorPoint(
            sqi_threshold=float(thr),
            coverage=coverage,
            mae_mean=mae_mean,
            mae_ci95=(mae_lo, mae_hi),
            n_subjects_covered=len(subject_maes),
        ))

    return CoverageErrorCurve(dataset_name=dataset_name, points=points)


def plot_coverage_vs_error(curve: CoverageErrorCurve, out_path: str, title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted(curve.points, key=lambda p: p.coverage)
    coverage = [p.coverage for p in points]
    mae_vals = [p.mae_mean for p in points]
    lo = [p.mae_ci95[0] for p in points]
    hi = [p.mae_ci95[1] for p in points]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, mae_vals, marker="o", color="tab:blue")
    ax.fill_between(coverage, lo, hi, alpha=0.2, color="tab:blue", label="95% bootstrap CI (по испытуемым)")
    ax.set_xlabel("Покрытие (доля окон, допущенных SQI до публикации)")
    ax.set_ylabel("MAE, BPM (среднее по испытуемым)")
    ax.set_title(title or f"Покрытие vs ошибка — {curve.dataset_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Интерфейс загрузчиков датасетов
# --------------------------------------------------------------------------- #

@dataclass
class GroundTruthSample:
    video_path: Path
    reference_bpm_timeseries: np.ndarray  # референсный BPM (по времени)
    reference_timestamps_s: np.ndarray
    subject_id: str
    # Опциональные метки для стратифицированного анализа (п.29 требований):
    # 'skin_tone' (см. benchmark/skin_tone.py), 'motion_level', 'illumination',
    # 'glasses', 'facial_hair', 'sex', 'age_group'. Датасеты редко несут всё
    # это в поставке — заполняется вызывающим кодом (демографический CSV
    # датасета, где он есть, или skin_tone.estimate_skin_tone_bucket).
    metadata: dict = field(default_factory=dict)


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


@dataclass
class DatasetEvaluationResult:
    """Результат DatasetEvaluator.run() — намеренно возвращает ОБА варианта
    агрегации, чтобы контраст между ними был виден в самом коде, а не только
    в докстрингах (п.28 требований):

    * pooled_report      — псевдореплицированный пулинг всех окон всех людей
                            (см. предупреждение в EvaluationReport). НЕ
                            использовать как основную метрику статьи.
    * subject_report     — корректная агрегация по испытуемым (см.
                            aggregate_per_subject). Основная метрика.
    * window_records     — ВСЕ окна (публикуемые и нет, с их sqi_score) —
                            вход для coverage_vs_error_curve (п.30) и
                            stratified_report (п.29).
    """

    dataset_name: str
    pooled_report: EvaluationReport
    subject_report: SubjectAggregateReport
    window_records: list[WindowRecord]

    def summary(self) -> str:
        return (
            f"{self.subject_report.summary()}\n"
            f"  [для контраста, НЕ использовать как основной результат — псевдореплицировано] "
            f"{self.pooled_report.summary()}"
        )


class DatasetEvaluator:
    """
    Прогоняет RPPGPipeline по всем сэмплам датасета и агрегирует метрики.

    Использование (после того как реально скачан датасет и есть модель):

        from rppg.pipeline import RPPGPipeline
        from rppg.config import PipelineConfig

        loader = UBFCrPPGLoader()
        evaluator = DatasetEvaluator(loader, PipelineConfig())
        result = evaluator.run(Path("/data/UBFC-rPPG/DATASET_2"))
        print(result.summary())
        plot_bland_altman(result.pooled_report.bland_altman, "ubfc_bland_altman.png")

        curve = coverage_vs_error_curve(result.window_records, dataset_name=loader.name)
        plot_coverage_vs_error(curve, "ubfc_coverage_vs_error.png")

        # Стратификация по тону кожи (п.29) — метки нужно заполнить заранее
        # в GroundTruthSample.metadata, например через benchmark/skin_tone.py.
        metadata_by_subject = {s.subject_id: s.metadata for s in loader.list_samples(root)}
        by_skin_tone = stratified_report(result.window_records, metadata_by_subject, "skin_tone")
    """

    def __init__(self, loader: DatasetLoader, pipeline_config):
        self.loader = loader
        self.pipeline_config = pipeline_config

    def run(self, root: Path, max_subjects: int | None = None) -> DatasetEvaluationResult:
        import cv2
        from rppg.pipeline import RPPGPipeline

        samples = self.loader.list_samples(root)
        if max_subjects is not None:
            samples = samples[:max_subjects]

        window_records: list[WindowRecord] = []

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
                    # ВСЕ окна с валидным (не NaN) BPM попадают в window_records,
                    # независимо от publishable — coverage_vs_error_curve (п.30)
                    # должна видеть отвергнутые SQI окна тоже, чтобы прогнать
                    # ВЕСЬ диапазон порогов, а не только текущий порог по
                    # умолчанию. "Что публикуется сейчас" — это window_records,
                    # отфильтрованные по .publishable, а не отдельный проход.
                    if result is not None and not np.isnan(result.bpm):
                        ref_bpm = np.interp(
                            ts_ms / 1000.0,
                            sample.reference_timestamps_s,
                            sample.reference_bpm_timeseries,
                        )
                        window_records.append(WindowRecord(
                            subject_id=sample.subject_id,
                            timestamp_ms=ts_ms,
                            true_bpm=float(ref_bpm),
                            pred_bpm=float(result.bpm),
                            sqi_score=result.sqi_score,
                            publishable=result.publishable,
                        ))
                    frame_idx += 1
            cap.release()

        published = [r for r in window_records if r.publishable]
        metadata_by_subject = {s.subject_id: s.metadata for s in samples}

        pooled_report = evaluate(
            np.array([r.true_bpm for r in published]),
            np.array([r.pred_bpm for r in published]),
            dataset_name=self.loader.name,
        )
        subject_report = aggregate_per_subject(
            published, dataset_name=self.loader.name, metadata_by_subject=metadata_by_subject
        )

        return DatasetEvaluationResult(
            dataset_name=self.loader.name,
            pooled_report=pooled_report,
            subject_report=subject_report,
            window_records=window_records,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
