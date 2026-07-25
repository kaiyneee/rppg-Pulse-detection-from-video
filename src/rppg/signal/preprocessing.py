"""
Подавление шума и препроцессинг (пункт "Подавление шума" ТЗ).

Обоснование выбора каждого метода — см. docs/research_report.md, раздел 4.3.
Коротко:

* detrend (linear / Tarvainen smoothness-priors) убирает медленный дрейф
  (дыхание, медленные изменения освещения, сползание ROI), который иначе
  доминирует в спектре и маскирует пульсовую составляющую.
* normalize (z-score) уравнивает амплитуды R/G/B и разных ROI перед их
  комбинированием (CHROM/POS по построению требуют выравнивания каналов).
* Butterworth bandpass — стандартный выбор в rPPG-литературе (de Haan 2013,
  Wang 2016 и др.) благодаря максимально плоской АЧХ в полосе пропускания
  (в отличие от Чебышева/эллиптического — без пульсаций, которые исказили
  бы форму пульсовой волны и, как следствие, оценку HRV).
* filtfilt (нулевая фаза) обязателен, если сигнал далее используется для
  детекции пиков под HRV — обычный lfilter вносит фазовую задержку,
  разрушающую точное время пиков.
* interpolate_missing — обработка выпадений кадров при окклюзии/потере
  трекинга (пункт "устойчивость к частичной окклюзии").
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, filtfilt, detrend as _scipy_detrend


def linear_detrend(x: np.ndarray) -> np.ndarray:
    """Простое устранение линейного тренда (быстрый базовый вариант)."""
    return _scipy_detrend(x, type="linear")


def tarvainen_detrend(x: np.ndarray, lam: float = 300.0) -> np.ndarray:
    """
    "Smoothness priors" detrending (Tarvainen, Ranta-Aho, Karjalainen, 2002,
    IEEE TBME — "An advanced detrending method with application to HRV analysis").

    Эквивалентно высокочастотному фильтру, реализованному через регуляризованную
    задачу наименьших квадратов:

        z_stat = (I - (I + lam^2 * D2^T D2)^(-1)) * z

    где D2 — оператор второй разности. Параметр lam задаёт "жёсткость" тренда,
    который удаляется: большие lam -> удаляется только очень медленный дрейф.

    Этот метод стандартен в HRV-анализе (используется в Kubios HRV и пакете
    neurokit2) и предпочтительнее наивного линейного детрендинга для окон
    длиннее ~5-10 секунд, где дрейф обычно нелинеен.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return x - np.mean(x)

    identity = sparse.eye(n, format="csc")
    d2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float)
    trend_operator = identity + (lam**2) * (d2.T @ d2)
    trend = spsolve(trend_operator.tocsc(), x)
    return x - trend


def detrend(x: np.ndarray, method: str = "tarvainen", lam: float = 300.0) -> np.ndarray:
    if method == "none":
        return np.asarray(x, dtype=float) - np.mean(x)
    if method == "linear":
        return linear_detrend(x)
    if method == "tarvainen":
        return tarvainen_detrend(x, lam=lam)
    raise ValueError(f"Unknown detrend method: {method}")


def normalize(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if method == "none":
        return x
    if method == "zscore":
        std = np.std(x)
        return (x - np.mean(x)) / std if std > 1e-8 else x - np.mean(x)
    if method == "minmax":
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo) if hi > lo else x - lo
    raise ValueError(f"Unknown normalize method: {method}")


def bandpass_filter(
    x: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Butterworth bandpass, zero-phase (filtfilt).

    low_hz/high_hz по умолчанию 0.7-4.0 Hz (42-240 BPM) — шире классических
    "45-180 BPM" оригинального репозитория, чтобы не отбрасывать тахикардию
    при остром стрессовом ответе (актуально для ПТСР-контекста, см. research_report).
    """
    x = np.asarray(x, dtype=float)
    nyquist = fps / 2.0
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.999)
    if low <= 0:
        raise ValueError("low_hz must be > 0")
    if low >= high:
        raise ValueError("low_hz must be < high_hz (после нормировки на Найквист)")

    b, a = butter(order, [low, high], btype="bandpass")

    # filtfilt требует len(x) > 3 * max(len(a), len(b)); для коротких окон
    # понижаем порядок фильтра, а не роняем пайплайн.
    min_len = 3 * max(len(a), len(b))
    if len(x) <= min_len:
        safe_order = max(1, order // 2)
        b, a = butter(safe_order, [low, high], btype="bandpass")
        min_len = 3 * max(len(a), len(b))
        if len(x) <= min_len:
            # Слишком короткое окно для zero-phase фильтрации — возвращаем
            # как есть (вызывающий код обязан пометить это в SQI).
            return x - np.mean(x)

    return filtfilt(b, a, x)


def interpolate_missing(x: np.ndarray, valid_mask: np.ndarray, max_gap: int = 15) -> tuple[np.ndarray, bool]:
    """
    Линейная интерполяция коротких пропусков (окклюзия, потеря трекинга).

    Возвращает (сигнал, ok). ok=False, если пропуск длиннее max_gap кадров
    подряд — в этом случае интерполяция была бы недостоверной, и окно должно
    быть исключено из анализа выше по пайплайну, а не "залечено" интерполяцией.
    """
    x = np.asarray(x, dtype=float).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if valid_mask.all():
        return x, True
    if not valid_mask.any():
        return x, False

    idx = np.arange(len(x))

    # Найти самый длинный подряд идущий провал.
    invalid = ~valid_mask
    max_run = 0
    run = 0
    for v in invalid:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    if max_run > max_gap:
        return x, False

    x[~valid_mask] = np.interp(idx[~valid_mask], idx[valid_mask], x[valid_mask])
    return x, True


def preprocess_signal(
    x: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
    detrend_method: str = "tarvainen",
    tarvainen_lambda: float = 300.0,
    normalize_method: str = "zscore",
) -> np.ndarray:
    """Полный конвейер: detrend -> normalize -> bandpass. Порядок важен:

    1) detrend снимает низкочастотный дрейф ДО нормализации (иначе дрейф
       исказит оценку std/zscore);
    2) bandpass в конце — дополнительно подавляет то, что detrend не взял,
       и режет высокочастотный шум камеры/квантования.
    """
    x = detrend(x, method=detrend_method, lam=tarvainen_lambda)
    x = normalize(x, method=normalize_method)
    x = bandpass_filter(x, fps, low_hz, high_hz, order=order)
    return x
