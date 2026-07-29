"""
Тесты для валидационной инфраструктуры benchmark/evaluate.py (п.27-30
требований) и benchmark/skin_tone.py (п.29) — синтетические, поскольку
доступа к реальным датасетам в этой среде нет (см. docstring вверху
benchmark/evaluate.py). Цель — доказать, что сама МАТЕМАТИКА агрегации,
bootstrap CI и кривой "покрытие vs ошибка" реализована верно, а не просто
"выглядит разумно"; реальные числа по датасетам — отдельная задача после
получения доступа к данным (см. docs/research_report.md).

Запуск: PYTHONPATH=src python3 tests/test_benchmark_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# benchmark/ — не пакет под src/ (namespace-пакет без __init__.py), поэтому
# добавляем корень репозитория в sys.path явно, а не полагаемся на то, что
# вызывающий проставит PYTHONPATH правильно для ЭТОГО конкретного файла.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.evaluate import (
    concordance_correlation_coefficient,
    reference_relative_snr_db,
    bootstrap_ci,
    WindowRecord,
    aggregate_per_subject,
    stratified_report,
    coverage_vs_error_curve,
    evaluate,
)
from benchmark.skin_tone import estimate_skin_tone


def test_ccc_penalizes_bias_that_pearson_ignores():
    """Регрессия для п.27: Pearson r инвариантен к постоянному сдвигу —
    метод, систематически завышающий BPM на константу, даёт r=1.0. CCC
    обязан это заметить и дать заметно меньшее значение."""
    print("\n=== Тест: CCC штрафует постоянный bias, который Pearson r не видит ===")
    true = np.array([60.0, 70.0, 80.0, 90.0, 100.0])
    biased_pred = true + 10.0  # идеальная линейная связь, но со сдвигом +10

    from scipy import stats
    r, _ = stats.pearsonr(true, biased_pred)
    ccc = concordance_correlation_coefficient(true, biased_pred)

    print(f"  Pearson r={r:.3f}, CCC={ccc:.3f}")
    assert r > 0.999, "sanity: постоянный сдвиг не должен портить Pearson r"
    assert ccc < 0.85, "CCC должен заметно штрафовать систематический bias, который Pearson r игнорирует"


def test_reference_relative_snr_distinguishes_correct_from_wrong_frequency():
    """Регрессия для п.27: сигнал может быть узкополосным и уверенным
    (высокий 'собственный' SNR), но на НЕВЕРНОЙ частоте — reference_relative_snr_db
    обязан отличать 'сигнал совпадает с референсом' от 'сигнал уверен, но неправ'.

    ВАЖНО: 'неверная' частота здесь НЕ должна быть точной 2-й гармоникой
    референса (2*reference_hz) — по определению этой метрики (de Haan &
    Jeanne, 2013; см. docstring reference_relative_snr_db) энергия у 2-й
    гармоники ЗАСЧИТЫВАЕТСЯ как согласие с референсом, потому что у
    реального PPG-сигнала она несёт физиологическую информацию (см. также
    quality.harmonic_plausibility, п.21, — там ровно эта октавная
    неоднозначность ловится отдельно). Поэтому берём частоту, не кратную
    референсной ни в 2, ни в 0.5 раза."""
    print("\n=== Тест: SNR относительно референса отличает верную частоту от неверной ===")
    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    band = (0.7, 4.0)
    rng = np.random.default_rng(0)

    true_hz = 1.2  # 72 BPM референс
    wrong_hz = 2.0  # заведомо не кратна true_hz (72 vs 120 BPM), не гармоника
    correct_signal = np.sin(2 * np.pi * true_hz * t) + 0.05 * rng.normal(0, 1, n)
    wrong_signal = np.sin(2 * np.pi * wrong_hz * t) + 0.05 * rng.normal(0, 1, n)

    snr_correct = reference_relative_snr_db(correct_signal, fps, band, reference_hz=true_hz)
    snr_wrong = reference_relative_snr_db(wrong_signal, fps, band, reference_hz=true_hz)

    print(f"  сигнал на верной частоте ({true_hz} Hz): SNR={snr_correct:.1f} дБ")
    print(f"  узкополосный сигнал на НЕСВЯЗАННОЙ частоте ({wrong_hz} Hz): SNR={snr_wrong:.1f} дБ")
    assert snr_correct > 0.0, "сигнал точно на референсной частоте должен иметь положительный SNR"
    assert snr_wrong < snr_correct - 5.0, "сигнал на неверной частоте должен иметь заметно более низкий SNR"


def test_bootstrap_ci_brackets_known_mean():
    print("\n=== Тест: bootstrap_ci даёт разумный интервал вокруг известного среднего ===")
    rng = np.random.default_rng(1)
    values = rng.normal(loc=5.0, scale=1.0, size=50)
    lo, hi = bootstrap_ci(values, n_boot=3000, seed=2)
    print(f"  выборочное среднее={np.mean(values):.3f}, bootstrap 95% CI=[{lo:.3f}, {hi:.3f}]")
    assert lo < np.mean(values) < hi, "CI должен покрывать выборочное среднее"
    assert lo < 5.0 < hi, "CI должен покрывать истинное среднее генеральной совокупности (n=50, не крайний случай)"

    single_lo, single_hi = bootstrap_ci(np.array([7.0]))
    assert single_lo == single_hi == 7.0, "единственное значение -> вырожденный интервал в это же значение"


def test_subject_aggregation_avoids_pseudoreplication():
    """Регрессия для п.28: классическая ошибка — считать MAE по всем окнам
    всех людей (человек с длинным видео весит больше). Строим намеренно
    несбалансированный пример: 'лёгкий' испытуемый с МНОГО окон и малой
    ошибкой, 'трудный' испытуемый с МАЛО окон и большой ошибкой. Пулинг
    должен занижать реальную типичную ошибку по сравнению с корректной
    агрегацией по испытуемым."""
    print("\n=== Тест: агрегация по испытуемым не даёт 'длинному видео' лишний вес ===")
    records = []
    # Испытуемый A: 20 окон, ошибка каждого окна = 2 BPM -> MAE_A = 2.
    for i in range(20):
        records.append(WindowRecord(
            subject_id="A", timestamp_ms=i * 1000, true_bpm=70.0, pred_bpm=72.0,
            sqi_score=0.9, publishable=True,
        ))
    # Испытуемый B: 2 окна, ошибка каждого окна = 20 BPM -> MAE_B = 20.
    for i in range(2):
        records.append(WindowRecord(
            subject_id="B", timestamp_ms=i * 1000, true_bpm=70.0, pred_bpm=90.0,
            sqi_score=0.9, publishable=True,
        ))

    pooled = evaluate(
        np.array([r.true_bpm for r in records]),
        np.array([r.pred_bpm for r in records]),
        dataset_name="synthetic",
    )
    subject_report = aggregate_per_subject(records, dataset_name="synthetic")

    print(f"  пулинг по окнам (ПСЕВДОРЕПЛИЦИРОВАНО): MAE={pooled.mae_bpm:.2f} BPM")
    print(f"  агрегация по испытуемым (корректно):    MAE={subject_report.mae_mean:.2f} BPM "
          f"(std={subject_report.mae_std:.2f})")

    assert subject_report.n_subjects == 2
    assert abs(subject_report.mae_mean - 11.0) < 1e-6, "среднее MAE по 2 испытуемым (2 и 20) должно быть 11.0"
    assert pooled.mae_bpm < 5.0, "пулинг занижает ошибку, т.к. 'лёгкий' испытуемый доминирует числом окон"
    assert subject_report.mae_mean > 2 * pooled.mae_bpm, (
        "корректная агрегация должна давать заметно бОльшую (и более честную) ошибку, "
        "чем псевдореплицированный пулинг в этом намеренно несбалансированном примере"
    )


def test_stratified_report_splits_by_metadata():
    print("\n=== Тест: stratified_report разбивает испытуемых по метаданным (п.29) ===")
    records = [
        WindowRecord(subject_id="A", timestamp_ms=0, true_bpm=70.0, pred_bpm=71.0, sqi_score=0.9, publishable=True),
        WindowRecord(subject_id="A", timestamp_ms=1000, true_bpm=70.0, pred_bpm=71.0, sqi_score=0.9, publishable=True),
        WindowRecord(subject_id="B", timestamp_ms=0, true_bpm=70.0, pred_bpm=75.0, sqi_score=0.9, publishable=True),
        WindowRecord(subject_id="C", timestamp_ms=0, true_bpm=70.0, pred_bpm=70.5, sqi_score=0.9, publishable=True),
    ]
    metadata = {
        "A": {"skin_tone": "light"},
        "B": {"skin_tone": "dark"},
        "C": {"skin_tone": "light"},
        # "D" намеренно отсутствует в records -> не должен нигде появиться
    }

    by_tone = stratified_report(records, metadata, stratum_key="skin_tone", dataset_name="synthetic")
    print(f"  группы: {sorted(by_tone.keys())}")
    print(f"  light: {by_tone['light'].summary()}")
    print(f"  dark:  {by_tone['dark'].summary()}")

    assert set(by_tone.keys()) == {"light", "dark"}
    assert by_tone["light"].n_subjects == 2  # A и C
    assert by_tone["dark"].n_subjects == 1   # B


def test_coverage_vs_error_curve_is_monotonic_when_sqi_predicts_error():
    """Синтетическая демонстрация п.30: строим окна, где sqi_score НАРОЧНО
    обратно коррелирует с реальной ошибкой (как и должно быть у рабочего
    SQI), и проверяем, что итоговая кривая 'покрытие vs ошибка'
    действительно монотонно не возрастает по мере ужесточения порога.
    Это НЕ подтверждает, что РЕАЛЬНЫЙ SQI пайплайна так себя ведёт на
    реальных данных (для этого нужен реальный датасет, см. п.25) — только
    то, что код самой кривой корректен и различает 'работающий SQI' от
    'нерабочего'."""
    print("\n=== Синтетическая демонстрация: coverage-vs-error монотонна, когда SQI информативен ===")
    rng = np.random.default_rng(3)
    records = []
    n_subjects, n_windows_per_subject = 12, 25
    for s in range(n_subjects):
        subject_id = f"subj{s}"
        for w in range(n_windows_per_subject):
            error = rng.uniform(0.0, 20.0)  # чем выше error, тем хуже окно
            sqi = float(np.clip(1.0 - error / 20.0 + rng.normal(0, 0.03), 0.0, 1.0))
            true_bpm = 70.0
            pred_bpm = true_bpm + (error if rng.random() < 0.5 else -error)
            records.append(WindowRecord(
                subject_id=subject_id, timestamp_ms=w * 1000,
                true_bpm=true_bpm, pred_bpm=pred_bpm, sqi_score=sqi, publishable=sqi >= 0.5,
            ))

    curve = coverage_vs_error_curve(records, dataset_name="synthetic", thresholds=np.linspace(0.0, 0.9, 10))
    for p in curve.points:
        print(f"  порог={p.sqi_threshold:.2f}  покрытие={p.coverage:.2f}  "
              f"MAE={p.mae_mean:.2f} ({p.n_subjects_covered} исп.)")

    assert curve.is_monotonic_nonincreasing(), (
        "при sqi_score, сконструированном как обратно коррелирующий с ошибкой, "
        "кривая покрытие-vs-ошибка должна монотонно не возрастать"
    )
    low_threshold_mae = curve.points[0].mae_mean
    high_threshold_mae = curve.points[-1].mae_mean
    print(f"  MAE при пороге={curve.points[0].sqi_threshold:.2f}: {low_threshold_mae:.2f}  "
          f"vs при пороге={curve.points[-1].sqi_threshold:.2f}: {high_threshold_mae:.2f}")
    assert high_threshold_mae < low_threshold_mae - 5.0, (
        "самый строгий порог должен давать заметно меньшую MAE, чем самый мягкий"
    )


def test_ita_skin_tone_estimator_orders_light_to_dark_monotonically():
    """Регрессия для п.29: ITA — объективный прокси тона кожи без ручной
    разметки (см. benchmark/skin_tone.py). Проверяем, что для градиента от
    светлого к тёмному цвету кожи ITA монотонно убывает — то есть шкала
    действительно отражает светлота->темнота, а не шум."""
    print("\n=== Тест: ITA монотонно убывает от светлой кожи к тёмной ===")
    swatches_light_to_dark = [
        (255, 224, 196),
        (241, 194, 125),
        (224, 172, 105),
        (198, 134, 66),
        (141, 85, 36),
        (92, 64, 51),
        (58, 40, 32),
    ]
    mask = np.ones((50, 50), dtype=np.uint8)
    ita_values = []
    for rgb in swatches_light_to_dark:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frame[:, :] = rgb[::-1]  # RGB -> BGR
        est = estimate_skin_tone(frame, mask)
        assert est is not None
        ita_values.append(est.ita_degrees)

    print(f"  ITA по градиенту свет->тьма: {[round(v, 1) for v in ita_values]}")
    assert all(ita_values[i] > ita_values[i + 1] for i in range(len(ita_values) - 1)), (
        "ITA должен монотонно убывать от самого светлого к самому тёмному образцу"
    )

    too_few_pixels = estimate_skin_tone(np.zeros((5, 5, 3), dtype=np.uint8), np.ones((5, 5), dtype=np.uint8))
    assert too_few_pixels is None, "недостаточно пикселей -> None, а не ненадёжная оценка"


def run_all():
    tests = [
        test_ccc_penalizes_bias_that_pearson_ignores,
        test_reference_relative_snr_distinguishes_correct_from_wrong_frequency,
        test_bootstrap_ci_brackets_known_mean,
        test_subject_aggregation_avoids_pseudoreplication,
        test_stratified_report_splits_by_metadata,
        test_coverage_vs_error_curve_is_monotonic_when_sqi_predicts_error,
        test_ita_skin_tone_estimator_orders_light_to_dark_monotonically,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed.append((test.__name__, str(e)))
            print(f"  [FAIL] {test.__name__}: {e}")

    print("\n" + "=" * 60)
    if failed:
        print(f"ИТОГ: {len(failed)}/{len(tests)} тестов провалено")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    else:
        print(f"ИТОГ: все {len(tests)} тестов пройдены успешно")


if __name__ == "__main__":
    run_all()
