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
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    respiration_rate_bpm: float | None
    warnings: list[str] = field(default_factory=list)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class StructuredWindowLogger:
    """Пишет один WindowLogRecord на строку в JSONL-файл.

    Файл открывается ЛЕНИВО при первом вызове log() (а не в __init__) —
    если ни одно окно так и не было оценено (например, видео короче
    min_seconds_before_estimate), не остаётся мусорного пустого файла.
    Каждая запись сразу flush()-ится: при офлайн-обработке видео это
    небольшие лишние syscall'ы, но при работе с веб-камерой в реальном
    времени (см. scripts/run_webcam.py) так лог остаётся читаемым СРАЗУ,
    даже если процесс потом упадёт или будет прерван — предпочтение отдано
    надёжности лога перед чистой пропускной способностью записи."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None

    def log(self, record: WindowLogRecord) -> None:
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
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
