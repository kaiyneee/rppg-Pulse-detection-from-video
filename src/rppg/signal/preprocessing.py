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


def tarvainen_detrend(x: np.ndarray, lam: float = 3542.0) -> np.ndarray:
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

    Дефолт lam=3542 (не "стандартные" 300 из HRV-литературы!) откалиброван
    под fs=30 Гц (видео), а не под типичный для HRV-литературы ресэмплинг
    RR-тахограммы на 4 Гц — см. tarvainen_cutoff_hz/tarvainen_frequency_response
    ниже и config.FilterConfig.tarvainen_lambda (п.36 требований).
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


def tarvainen_frequency_response(
    lam: float, fs: float, freqs_hz: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    АЧХ tarvainen_detrend как высокочастотного фильтра (п.36 требований):
    tarvainen_lambda=300 в исходном коде был взят из HRV-литературы БЕЗ
    проверки, что он означает на fs=30 Гц (видео), а не на sampling rate,
    для которого он был откалиброван изначально.

    Вывод формулы: D2 (вторая разность, ядро [1,-2,1]) в частотной области
    циклического (Toeplitz-circulant-приближение, см. ОГОВОРКУ ниже) оператора
    имеет отклик D2(w) = e^{jw} - 2 + e^{-jw} = 2cos(w) - 2 = -4 sin^2(w/2),
    значит |D2(w)|^2 = 16 sin^4(w/2). Тогда полный "trend operator"
    (I + lam^2 D2^T D2) имеет частотный отклик 1 + lam^2 * 16 sin^4(w/2), а
    высокочастотный фильтр z = x - trend имеет отклик

        H_hp(w) = 1 - 1 / (1 + lam^2 * 16 * sin^4(w/2)),   w = 2*pi*f/fs.

    ПРОВЕРКА ФОРМУЛЫ ПО ДОКУМЕНТИРОВАННОМУ РЕФЕРЕНСНОМУ ЗНАЧЕНИЮ: стандартная
    HRV-практика (Kubios HRV / PhysioData Toolbox) ресэмплирует RR-тахограмму
    на 4 Гц и использует lam=500, что документировано как cutoff~=0.04 Hz
    (см. https://physiodatatoolbox.leidenuniv.nl/docs/user-guide/physioanalyzer-modules/hrv-module.html).
    Эта формула для (lam=500, fs=4) даёт cutoff=0.0355 Hz — совпадает с
    точностью до округления источника, что подтверждает формулу.

    ОГОВОРКА: это приближение через циклический (Toeplitz-circulant) оператор
    — реальная tarvainen_detrend решает КОНЕЧНУЮ систему (spsolve на
    Toeplitz, не circulant), и краевые эффекты на границах окна отличаются
    от идеального периодического фильтра. Для оценки ЭФФЕКТИВНОЙ частоты
    среза в середине окна этого достаточно (тот же уровень строгости, что и
    в HRV-литературе, ссылка выше).
    """
    if freqs_hz is None:
        freqs_hz = np.linspace(1e-4, fs / 2.0, 4000)
    omega = 2.0 * np.pi * freqs_hz / fs
    d2_response_sq = 16.0 * np.sin(omega / 2.0) ** 4
    h_hp = 1.0 - 1.0 / (1.0 + (lam**2) * d2_response_sq)
    return freqs_hz, h_hp


def tarvainen_cutoff_hz(lam: float, fs: float, level: float = 1.0 / np.sqrt(2.0)) -> float:
    """Частота среза (-3dB по умолчанию, level=1/sqrt(2)) АЧХ tarvainen_detrend
    — см. tarvainen_frequency_response. Численный поиск корня, а не
    аналитическое решение (гладкая монотонная H_hp(w), поиск устойчив)."""
    from scipy.optimize import brentq

    def f(freq_hz: float) -> float:
        _, h = tarvainen_frequency_response(lam, fs, freqs_hz=np.array([freq_hz]))
        return float(h[0]) - level

    lo, hi = 1e-6, fs / 2.0 - 1e-6
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return float(brentq(f, lo, hi))


def detrend(x: np.ndarray, method: str = "tarvainen", lam: float = 3542.0) -> np.ndarray:
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


def is_gap_acceptable(valid_mask: np.ndarray, max_gap: int = 15) -> bool:
    """True, если самый длинный подряд идущий провал (False) в valid_mask
    не превышает max_gap кадров. Вынесено из interpolate_missing (п.47
    требований): valid_mask на практике общая на все 3 канала R/G/B одного
    ROI (окклюзия определяется на уровне КАДРА, а не канала, см.
    pipeline.py::_estimate_color_roi) — раньше это пересчитывалось 3 раза
    подряд для одной и той же маски."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.all():
        return True
    if not valid_mask.any():
        return False

    invalid = ~valid_mask
    max_run = 0
    run = 0
    for v in invalid:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run <= max_gap


def fill_gaps(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Линейная интерполяция пропусков БЕЗ проверки допустимости их длины
    (см. is_gap_acceptable) — вызывающий код обязан проверить
    is_gap_acceptable САМ (один раз на маску, см. п.47), прежде чем звать
    это на каждом из нескольких каналов с той же маской."""
    x = np.asarray(x, dtype=float).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.all() or not valid_mask.any():
        return x
    idx = np.arange(len(x))
    x[~valid_mask] = np.interp(idx[~valid_mask], idx[valid_mask], x[valid_mask])
    return x


def interpolate_missing(x: np.ndarray, valid_mask: np.ndarray, max_gap: int = 15) -> tuple[np.ndarray, bool]:
    """
    Линейная интерполяция коротких пропусков (окклюзия, потеря трекинга)
    для ОДНОГО сигнала со СВОЕЙ собственной маской.

    Возвращает (сигнал, ok). ok=False, если пропуск длиннее max_gap кадров
    подряд — в этом случае интерполяция была бы недостоверной, и окно должно
    быть исключено из анализа выше по пайплайну, а не "залечено" интерполяцией.

    Если нужно интерполировать НЕСКОЛЬКО каналов/сигналов с ОДНОЙ и той же
    valid_mask (типичный случай — R/G/B одного ROI), вызовите
    is_gap_acceptable(valid_mask) ОДИН раз и затем fill_gaps() для каждого
    канала — эта функция для такого случая пересчитывала бы длину провала
    избыточно (п.47 требований, см. pipeline.py::_estimate_color_roi).
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not is_gap_acceptable(valid_mask, max_gap=max_gap):
        return np.asarray(x, dtype=float).copy(), False
    return fill_gaps(x, valid_mask), True


def preprocess_signal(
    x: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
    detrend_method: str = "tarvainen",
    tarvainen_lambda: float = 3542.0,
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
