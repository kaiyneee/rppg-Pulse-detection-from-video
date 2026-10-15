"""
YAML/JSON (де)сериализация PipelineConfig (п.42 требований): "один
конфиг-файл на эксперимент... который сохраняется рядом с результатами.
Каждая строчка таблицы в статье должна быть привязана к конкретному
конфигу." save_config/load_config дают именно это — конфиг эксперимента
становится файлом на диске, а не неявным состоянием кода на момент запуска.

РЕАЛИЗОВАНО ЯВНО, ПО КАЖДОМУ ВЛОЖЕННОМУ DATACLASS'У, а не через общую
reflection-based схему (typing.get_type_hints + dataclasses.fields) — это
осознанный выбор, а не недосмотр: get_type_hints() на Python 3.9 падает
на полях вида `onnx_model_path: str | None` (AccelerationConfig) —
PEP 604 `X | None` синтаксис в аннотациях требует РЕАЛЬНОГО вычисления
типа для get_type_hints(), а это работает только на Python >= 3.10, даже
при `from __future__ import annotations` (эта строка делает аннотации
ленивыми/строковыми, но get_type_hints() всё равно их eval()-ит). Мы
таргетируем Python 3.9+ (см. requirements.txt, .github/workflows/ci.yml),
поэтому явные функции ниже — более надёжный выбор, проверенный эмпирически.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rppg.config import (
    PipelineConfig,
    FaceModelConfig,
    ROIConfig,
    FilterConfig,
    WindowConfig,
    QualityConfig,
    HRVConfig,
    AccelerationConfig,
    FusionConfig,
    ExtractionMethod,
    FrequencyMethod,
    ROIName,
    PRESET_FOR_FILES_WITHOUT_PRESET,
    PRESET_OVERRIDES,
    apply_preset,
)


def config_to_dict(config: PipelineConfig) -> dict:
    """PipelineConfig -> обычный dict, пригодный для YAML/JSON.

    dataclasses.asdict() уже рекурсивно разворачивает ВСЕ вложенные
    dataclass'ы (face/roi/filt/window/quality/hrv/accel/fusion) в dict —
    единственное, что asdict НЕ трогает, это значения Enum (оставляет сам
    Enum-объект внутри словаря, что YAML/JSON сериализовать не умеют),
    поэтому три Enum-поля (method, frequency_method, roi.enabled_rois)
    заменяются на их .value вручную ниже.
    """
    data = asdict(config)
    data["method"] = config.method.value
    data["frequency_method"] = config.frequency_method.value
    data["roi"]["enabled_rois"] = [r.value for r in config.roi.enabled_rois]
    return data


def _section(data: dict, name: str, cls: type) -> Any:
    raw = data.get(name)
    return cls(**raw) if raw else cls()


def config_from_dict(data: dict) -> PipelineConfig:
    """Восстанавливает PipelineConfig из dict (обычно — загруженного из
    YAML/JSON файла). Отсутствующие в data ключи/секции берут дефолт
    соответствующего dataclass — частичные конфиги (только изменённые
    относительно дефолта параметры) допустимы и это ОСОЗНАННО удобно для
    экспериментов, где меняется 1-2 параметра."""
    data = data or {}

    roi_data = dict(data.get("roi") or {})
    if "enabled_rois" in roi_data:
        roi_data["enabled_rois"] = tuple(ROIName(v) for v in roi_data["enabled_rois"])
    roi_cfg = ROIConfig(**roi_data) if roi_data else ROIConfig()

    kwargs: dict[str, Any] = dict(
        face=_section(data, "face", FaceModelConfig),
        roi=roi_cfg,
        filt=_section(data, "filt", FilterConfig),
        window=_section(data, "window", WindowConfig),
        quality=_section(data, "quality", QualityConfig),
        hrv=_section(data, "hrv", HRVConfig),
        accel=_section(data, "accel", AccelerationConfig),
        fusion=_section(data, "fusion", FusionConfig),
    )
    if "method" in data:
        kwargs["method"] = ExtractionMethod(data["method"])
    if "frequency_method" in data:
        kwargs["frequency_method"] = FrequencyMethod(data["frequency_method"])

    cfg = PipelineConfig(**kwargs)
    # Пресет: поля, которых нет в файле, достраиваются от пресета, при
    # котором файл записан. Файл без поля preset создан до появления
    # пресетов, то есть при прежних умолчаниях (legacy_welch8), — без этого
    # старые конфиги экспериментов молча начали бы означать другое.
    preset = data.get("preset", PRESET_FOR_FILES_WITHOUT_PRESET)
    present = {k for k in data if k not in ("roi",)} | {f"roi.{k}" for k in roi_data}
    missing = {k for k in PRESET_OVERRIDES.get(preset, {}) if k not in present}
    return apply_preset(cfg, preset, only_missing=missing)


def save_config(config: PipelineConfig, path: str | Path) -> None:
    """Сохраняет config эксперимента на диск — РЯДОМ с результатами (CSV/
    JSON/графиками), см. п.42 требований. Формат определяется по
    расширению: .yaml/.yml -> YAML (человекочитаемо, с комментариями можно
    редактировать руками), иначе JSON."""
    path = Path(path)
    data = config_to_dict(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix in (".yaml", ".yml"):
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_config(path: str | Path) -> PipelineConfig:
    """Загружает PipelineConfig из YAML/JSON файла, сохранённого
    save_config() (или написанного вручную по тому же формату)."""
    path = Path(path)

    if path.suffix in (".yaml", ".yml"):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return config_from_dict(data)
