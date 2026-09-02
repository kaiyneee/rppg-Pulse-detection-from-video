"""
Структурированное логирование по окнам (п.43 требований): "Каждое окно ->
строка с timestamp, BPM по каждому ROI, компоненты SQI, warnings. Это и
отладка, и материал для анализа failure cases в статье."

Формат — JSON Lines (.jsonl): один JSON-объект на строку, один вызов
RPPGPipeline.process_frame() с непустым результатом -> одна строка.
Стандартный формат для потокового структурированного лога: не требует
парсинга ВСЕГО файла для чтения одной записи, легко читается построчно
любым JSON-парсером или через `pandas.read_json(path, lines=True)` для
последующего анализа failure cases в статье.

ЗАДАЧА 26 (доказанный баг): раньше файл ВСЕГДА открывался в режиме "a"
(дозапись) — повторный прогон на тот же --log путь молча СКЛЕИВАЛ новые
строки со старыми, в т.ч. записанными на другом конфиге (другой порог,
другая полуширина SNR и т.п.). Найдено на реальном файле
veronika_session.jsonl: 204 строки при 68 окнах за прогон — ровно 68×3,
три склеенных прогона, перцентили совпадали до последнего знака (выборка,
повторённая трижды, не меняет перцентили). Теперь:
  1. Файл по умолчанию ПЕРЕЗАПИСЫВАЕТСЯ ("w"); дозапись — только через
     явный StructuredWindowLogger(..., append=True).
  2. Первая строка КАЖДОГО открытия файла — заголовок прогона (build_run_
     header) с параметрами, которые определяют СРАВНИМОСТЬ записей
     (min_spectral_snr_db, полуширина SNR, метод оценки частоты и т.п.) и
     git-хэшем кода. analyze_log.py читает эти заголовки и ОТКАЗЫВАЕТСЯ
     считать статистику, если в одном файле нашлось больше одного
     РАЗЛИЧНОГО заголовка — вместо того чтобы молча посчитать смесь.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Маркер-ключ, отличающий строку заголовка от строки WindowLogRecord в
# JSONL-потоке — обе структуры пишутся построчно в один и тот же файл.
RUN_HEADER_MARKER = "__run_header__"


@dataclass
class WindowLogRecord:
    """Одна строка структурированного лога — одно окно оценки BPM.

    Намеренно содержит компоненты SQI ПО ОТДЕЛЬНОСТИ (не только итоговый
    overall_score), т.к. именно разбивка по компонентам — материал для
    анализа failure cases: "почему именно это окно не опубликовано" виден
    сразу (например spectral_snr_db низкий, но flicker_suspected=False —
    значит дело не в мерцании освещения, а в самом сигнале)."""

    timestamp_ms: int
    bpm: float
    per_roi_bpm: dict[str, float]
    publishable: bool
    method_used: str
    frequency_method_used: str
    sqi_overall_score: float
    sqi_level: str
    sqi_spectral_snr_db: float
    sqi_cross_roi_agreement: float
    sqi_landmark_stability: float
    sqi_temporal_consistency: float
    sqi_harmonic_score: float
    sqi_flicker_suspected: bool
    # Задача 7, п.4: частота/SNR фонового пика КАЖДОГО окна (не только когда
    # flicker_suspected=True) — материал для scripts/analyze_log.py, чтобы
    # показать, насколько ОБОСНОВАННО срабатывает детектор мерцания, а не
    # только сам факт срабатывания. -inf, если фон недоступен/отфильтрован
    # как "нет данных" (см. QualityConfig.flicker_min_raw_std).
    sqi_flicker_background_freq_hz: float
    sqi_flicker_background_snr_db: float
    # Задача 4, п.3: False, если f/2 был ниже рабочей полосы и проверка
    # субгармоники была пропущена как физически неинформативная (обычный
    # случай при BPM покоя < ~84) — см. quality.harmonic_plausibility.
    sqi_subharmonic_check_informative: bool
    respiration_rate_bpm: float | None
    warnings: list[str] = field(default_factory=list)
    # Задача 17: сколько ROI реально участвовало в cross_roi_agreement_score
    # ПОСЛЕ относительной SNR-фильтрации (см. quality.cross_roi_agreement_score)
    # — без этого поля 0.5 "недостаточно участников" неотличимо от 0.5
    # "участники есть, но реально не согласны".
    sqi_cross_roi_n_participants: int = 0
    # Задача 20: горб на 2f выше локального фона — ИНФОРМАЦИОННЫЙ признак
    # (несинусоидальность реального пульса), НЕ штраф и НЕ участвует в
    # harmonic_score/overall (см. quality.harmonic_plausibility).
    sqi_harmonic_support: bool = False
    # Задача 21: детальная причина, ПОЧЕМУ окно получилось таким, каким
    # получилось — "ok" | "no_face" | "no_signal" | "degenerate_signal" |
    # "too_short" (см. pipeline.py::RPPGPipeline._compute_estimate). В
    # отличие от PTSDPulseFeatures.status (только ok/no_signal, стабильный
    # контракт для потребителя), это диагностическое поле только для
    # JSONL/CSV-анализа failure cases (scripts/analyze_log.py).
    window_status: str = "ok"
    # Задача 25: агрегированный BPM для автоматической подачи в систему
    # ПТСР — см. полный контракт потребителя в docstring полей bpm_aggregate*
    # на pipeline.PTSDPulseFeatures (quality==0.0 -> нельзя использовать;
    # status=="stale" -> НЕ новое наблюдение, не независимая точка).
    bpm_aggregate: float = float("nan")
    bpm_aggregate_quality: float = 0.0
    bpm_aggregate_status: str = "no_data"
    bpm_aggregate_n_windows: int = 0
    bpm_aggregate_spread_bpm: float = float("nan")
    # Задача 44: результат кросс-проверки спектральной оценки BPM по пикам
    # того же сигнала — "agreement" | "corrected_to_2f" | "corrected_to_half_f" |
    # "ambiguous" | "insufficient_peaks" (см. hrv.features.CrossCheckResult).
    # cross_check_ratio = mean_hr_from_peaks / bpm_spectral (ДО коррекции),
    # None при verdict="insufficient_peaks".
    cross_check_verdict: str = "insufficient_peaks"
    cross_check_ratio: float | None = None
    cross_check_mean_hr_from_peaks: float | None = None
    # Задача 44 (сглаживание после cross-check): True, если bpm скорректирован
    # НЕ самим cross_check (тот не смог подтвердить по своим пикам в этом
    # окне), а по медиане недавних ПОДТВЕРЖДЁННЫХ окон — см.
    # RPPGPipeline._smooth_bpm_against_trusted_history.
    bpm_was_smoothed: bool = False
    # Задача 47: допущено ли ЭТО окно в опорную историю кросс-проверки/
    # сглаживания (window_status=="ok" И spectral_snr_db>=history_min_snr_db
    # И cross_check_verdict!="ambiguous") — ОТДЕЛЬНО от валидности кадра
    # ROI. history_rejection_reason — почему нет, None если entered_history.
    entered_history: bool = False
    history_rejection_reason: str | None = None
    # Задача 48 (двухпроходный офлайн-режим): True, если решение о
    # коррекции ЭТОГО окна было принято при истории короче
    # FrequencyConfig.history_min_windows (см. RPPGPipeline._smooth_bpm_
    # against_trusted_history / apply_history_ratio_check) — кандидат на
    # пересмотр вторым проходом в scripts/run_on_video.py --two-pass.
    # pass_that_decided — 1 (обычный причинный проход) или 2 (значение
    # пересмотрено вторым проходом по итоговой истории целиком). В
    # потоковом (веб-камера) режиме всегда 1 — там причинность обязательна.
    history_was_insufficient: bool = False
    pass_that_decided: int = 1

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _git_short_hash() -> str | None:
    """Короткий git-хэш HEAD, если репозиторий доступен — None в
    противном случае (установка из архива/pip, git недоступен и т.п.), а
    не исключение: заголовок прогона должен писаться в любой среде."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 - git недоступен -> заголовок без хэша, не падаем
        pass
    return None


def build_run_header(cfg) -> dict:
    """Задача 26: параметры прогона, которые определяют СРАВНИМОСТЬ строк
    лога между собой — если хотя бы один из них отличается между двумя
    прогонами в одном файле, статистика по всему файлу (перцентили,
    доля publishable и т.п.) физически бессмысленна, а не просто "чуть
    менее точна". cfg — PipelineConfig (типа не указан здесь, чтобы не
    тащить import rppg.config в этот модуль ради одной аннотации)."""
    return {
        RUN_HEADER_MARKER: True,
        "min_spectral_snr_db": cfg.quality.min_spectral_snr_db,
        "min_overall_score_to_publish": cfg.quality.min_overall_score_to_publish,
        "snr_peak_half_width_mode": cfg.quality.snr_peak_half_width_mode,
        "snr_peak_half_width_hz": cfg.quality.snr_peak_half_width_hz,
        "snr_peak_half_width_k": cfg.quality.snr_peak_half_width_k,
        "include_second_harmonic_in_snr": cfg.quality.include_second_harmonic_in_snr,
        "frequency_method": cfg.frequency_method.value,
        "method": cfg.method.value,
        "window_seconds": cfg.window.window_seconds,
        "step_seconds": cfg.window.step_seconds,
        "git_hash": _git_short_hash(),
    }


def read_jsonl_records(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Читает JSONL-лог и разделяет строки на (заголовки, окна) по
    RUN_HEADER_MARKER — общий низкоуровневый парсер для scripts/analyze_log.py
    и scripts/record_with_reference.py: ни один из них не должен молча
    трактовать строку заголовка как WindowLogRecord (задача 26). Логи,
    записанные ДО задачи 26 (без заголовка вообще), дают headers=[] —
    вызывающий код решает сам, считать ли это ошибкой."""
    headers: list[dict] = []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get(RUN_HEADER_MARKER):
                headers.append(obj)
            else:
                records.append(obj)
    return headers, records


class StructuredWindowLogger:
    """Пишет один WindowLogRecord на строку в JSONL-файл, и ОДНУ строку
    заголовка (build_run_header) перед первой записью каждого открытия
    файла (см. RUN_HEADER_MARKER и docstring модуля, задача 26).

    Файл открывается ЛЕНИВО при первом вызове log() (а не в __init__) —
    если ни одно окно так и не было оценено (например, видео короче
    min_seconds_before_estimate), не остаётся мусорного пустого файла.
    Каждая запись сразу flush()-ится: при офлайн-обработке видео это
    небольшие лишние syscall'ы, но при работе с веб-камерой в реальном
    времени (см. scripts/run_webcam.py) так лог остаётся читаемым СРАЗУ,
    даже если процесс потом упадёт или будет прерван — предпочтение отдано
    надёжности лога перед чистой пропускной способностью записи.

    ПО УМОЛЧАНИЮ файл ПЕРЕЗАПИСЫВАЕТСЯ (режим "w") — задача 26, доказанный
    баг: старое поведение (всегда "a") молча склеивало записи от разных
    прогонов/конфигов в одном файле. append=True — явный, осознанный опт-ин
    для сценариев, где дозапись действительно нужна (например,
    scripts/run_webcam.py пишет один непрерывный сеанс несколькими
    process_frame() без переоткрытия файла — там append не нужен вовсе,
    т.к. логгер и так открывается один раз за сеанс; append=True нужен,
    только если СОЗНАТЕЛЬНО объединяют несколько отдельных запусков
    процесса в один файл)."""

    def __init__(self, path: str | Path, header: dict | None = None, append: bool = False):
        self.path = Path(path)
        self.header = header
        self.append = append
        self._file = None

    def log(self, record: WindowLogRecord) -> None:
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.append else "w"
            self._file = open(self.path, mode, encoding="utf-8")
            if self.header is not None:
                self._file.write(json.dumps(self.header, ensure_ascii=False) + "\n")
        self._file.write(record.to_json_line() + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "StructuredWindowLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
