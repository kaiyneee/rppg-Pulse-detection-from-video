# rPPG-PTSD: модуль бесконтактного измерения пульса и ВСР для исследования физиологических маркеров ПТСР

Модернизированный, переработанный по модулям преемник
[irfan798/head-pulse-track](https://github.com/irfan798/head-pulse-track)
(реализация Balakrishnan et al., CVPR 2013). Полный разбор оригинального
репозитория, обоснование каждого архитектурного решения, научная новизна
и экспериментальный протокол — в **[docs/research_report.md](docs/research_report.md)**.
Этот README — только установка и быстрый старт.

## Возможности

- Детекция лица: MediaPipe **Face Landmarker** (Tasks API, 2023+), 468 точек + поза головы.
- Динамические ROI: лоб, левая щека, правая щека, с occlusion-гейтингом на уровне пикселей.
- 6 методов извлечения сигнала с runtime-переключением: `GREEN`, `CHROM`, `POS`, `PCA`, `ICA`, `HEAD_MOTION`.
- Препроцессинг: Tarvainen smoothness-priors detrending, z-score нормализация, Butterworth bandpass (zero-phase).
- 3 метода частотного анализа: `FFT`, `Welch`, `Lomb-Scargle` (последний — для неравномерно дискретизированных данных после удаления окклюдированных кадров).
- Signal Quality Index (спектральный SNR + межзонное согласие + стабильность трекинга) с жёстким gating: низкое качество → BPM/HRV не публикуются.
- HRV-признаки: Mean HR, SDNN, RMSSD, pNN50, pNN20, LF/HF.
- Опциональное ускорение Numba (POS), заготовка под ONNX Runtime для обучаемого метода.

## Установка

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Модель Face Landmarker скачивается один раз (см. модель ниже: ~4-6 MB)
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

> Для воспроизводимых экспериментов рекомендуется зафиксировать версию модели
> (заменить `float16/latest/` на `float16/1/` или актуальный номер) —
> см. docs/research_report.md, раздел 3.1.

## Быстрый старт

```bash
# Веб-камера, метод POS + Welch (по умолчанию)
PYTHONPATH=src python scripts/run_webcam.py

# Офлайн-обработка видео с сохранением временного ряда в CSV
PYTHONPATH=src python scripts/run_on_video.py --video sample.mp4 --method chrom --freq lomb_scargle --out result.csv
```

Программный API:

```python
from rppg.config import PipelineConfig, ExtractionMethod
from rppg.pipeline import RPPGPipeline
import cv2

config = PipelineConfig(method=ExtractionMethod.POS)
cap = cv2.VideoCapture(0)

with RPPGPipeline(config) as pipeline:
    ok, frame = cap.read()
    result = pipeline.process_frame(frame, timestamp_ms=0)
    if result and result.publishable:
        print(result.bpm, result.hrv.sdnn_ms, result.hrv.rmssd_ms)
    elif result:
        print("низкое качество сигнала:", result.warnings)
```

## Тесты

Полный конвейер (препроцессинг → 6 методов извлечения → 3 метода частотного
анализа → HRV) проверен на синтетических сигналах с известным BPM
(MediaPipe для этого не требуется):

```bash
PYTHONPATH=src python3 tests/test_signal_pipeline.py
```

## Структура проекта

```
src/rppg/
  config.py           конфигурация (датаклассы)
  face/
    landmarker.py      обёртка над MediaPipe Face Landmarker
    roi.py              ROI (лоб/щёки), проверенные индексы landmark-точек
  signal/
    preprocessing.py    detrend, normalize, bandpass
    methods.py           GREEN/CHROM/POS/PCA/ICA/HEAD_MOTION
    quality.py           Signal Quality Index
    frequency.py          FFT/Welch/Lomb-Scargle
  hrv/
    features.py         Mean HR, SDNN, RMSSD, pNN50, LF/HF
  accel/
    fast_ops.py          Numba-ускорение POS (опционально)
  pipeline.py            оркестрация + SQI-gating перед публикацией в ПТСР-систему
scripts/                 CLI: веб-камера, офлайн-видео
benchmark/evaluate.py    MAE/RMSE/Pearson/Bland-Altman + загрузчик UBFC-rPPG
tests/                   тесты на синтетике
docs/research_report.md  полный анализ, архитектура, эксперименты
```

## Ограничения текущей версии (см. research_report.md за подробностями)

- Валидация выполнена на синтетических сигналах; прогон на реальных
  датасетах (VIPL-HR/UBFC-rPPG/PURE/MMSE-HR) требует их скачивания по
  data use agreement и не выполнялся в среде разработки этого кода.
- HRV-признаки вычисляются из межпульсовых, а не межсердечных (ЭКГ)
  интервалов (PRV, а не HRV в строгом смысле) — см. hrv/features.py, docstring.
- Признаки, полезные для ПТСР-скрининга (HRV/RMSSD/LF-HF), являются
  неспецифичным физиологическим коррелятом, а не диагностическим критерием —
  см. research_report.md, раздел 7.
