# rPPG-PTSD: модуль бесконтактного измерения пульса и ВСР для исследования физиологических маркеров ПТСР

[![CI](https://github.com/kaiyneee/rppg-Pulse-detection-from-video/actions/workflows/ci.yml/badge.svg)](https://github.com/kaiyneee/rppg-Pulse-detection-from-video/actions/workflows/ci.yml)

Независимая, модульная реализация классических алгоритмов rPPG/HRV
(Balakrishnan et al., CVPR 2013; de Haan & Jeanne, IEEE TBME 2013; Wang et
al., IEEE TBME 2016; и др. — полный список в [CITATION.cff](CITATION.cff)),
построенная на MediaPipe Face Landmarker (Tasks API). Научной отправной
точкой послужил [irfan798/head-pulse-track](https://github.com/irfan798/head-pulse-track)
— он первым в этой цепочке указал на Balakrishnan et al. (2013) для этой
задачи; тот репозиторий не имеет лицензии (все права защищены его
авторами), и код из него здесь не переиспользуется — см. раздел
"Acknowledgements" ниже. Полный разбор архитектурных решений, научная
новизна и экспериментальный протокол — в
**[docs/research_report.md](docs/research_report.md)**. Этот README —
только установка, быстрый старт и обзор того, что реально есть в коде на
сегодня. См. также [LICENSE](LICENSE) и [CITATION.cff](CITATION.cff).

> **Рамка исследования.** Это модуль бесконтактного измерения физиологических
> маркеров (BPM, HRV), потенциально релевантных для исследований ПТСР — а
> НЕ инструмент диагностики или скрининга. HRV/RMSSD/LF-HF — неспецифичные
> физиологические корреляты, не диагностический критерий (см.
> research_report.md, раздел 7).

## Возможности

- Детекция лица: MediaPipe **Face Landmarker** (Tasks API), landmark-точки + поза головы. Версия модели зафиксирована (`float16/1/`, не `/latest/`) — см. "Установка".
- Динамические ROI: лоб, левая щека, правая щека, с occlusion-гейтингом на уровне пикселей и skin-маской (YCrCb).
- 6 методов извлечения сигнала с runtime-переключением: `GREEN`, `CHROM`, `POS`, `PCA`, `ICA`, `HEAD_MOTION`.
- **SQI-взвешенный fusion** (опционально, `FusionConfig.enabled`): вместо argmax-выбора одного ROI — объединение сигналов 3 ROI и head-motion канала с весами по spectral SNR и выравниванием фазы/знака между модальностями (`src/rppg/signal/fusion.py`). См. `scripts/compare_fusion_vs_argmax.py` для сравнения с argmax на синтетике.
- Препроцессинг: Tarvainen smoothness-priors detrending (λ откалибрована под fs=30 Гц, а не взята из HRV-литературы не глядя — см. `scripts/analyze_tarvainen_lambda.py`), z-score нормализация, Butterworth bandpass (zero-phase).
- 3 метода частотного анализа: `FFT`, `Welch`, `Lomb-Scargle` (последний — для неравномерно дискретизированных данных после удаления окклюдированных кадров).
- **Signal Quality Index**, 4 компоненты: спектральный SNR (со штрафом за гармоники/субгармоники), межзонное согласие, стабильность landmark-точек (нормирована на межзрачковое расстояние — абсолютная, не самоотносительная нормировка), устойчивость BPM между соседними окнами (temporal consistency). Плюс отдельный жёсткий гейт на мерцание освещения через фоновый ROI. Низкое качество → BPM/HRV не публикуются (`publishable=False`).
- HRV-признаки: Mean HR, SDNN, RMSSD, pNN50, pNN20, LF/HF, с суб-сэмпловой интерполяцией пиков и маскированием эктопических интервалов.
- Оценка частоты дыхания по амплитудной модуляции пульсовой волны.
- Опциональное ускорение Numba (POS, ~80x на этой машине — см. `scripts/benchmark_performance.py`), заготовка под ONNX Runtime для обучаемого метода.
- Объективная оценка тона кожи по ITA (без ручной Fitzpatrick-разметки) для стратифицированного анализа — `benchmark/skin_tone.py`.
- Структурированное JSONL-логирование по окнам (`--log`, см. "Быстрый старт") и один YAML/JSON-конфиг на эксперимент, сохраняемый рядом с результатами (`rppg.config_io`).

## Установка

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # точные версии (==) — см. requirements.txt

# Модель Face Landmarker скачивается один раз (~3.6 MB), версия ЗАФИКСИРОВАНА
# (float16/1/, не /latest/ — см. src/rppg/face/landmarker.py::MODEL_URL, п.41)
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

`requirements.txt` зафиксирован точными версиями (`==`), реально протестированными
в этом репозитории на Python 3.9.6 — CI (`.github/workflows/ci.yml`) дополнительно
прогоняет те же пины на Python 3.9/3.11/3.12 как smoke-test совместимости.

## Быстрый старт

```bash
# Веб-камера, метод POS + Welch (по умолчанию)
python scripts/run_webcam.py

# Офлайн-обработка видео с сохранением временного ряда в CSV
python scripts/run_on_video.py --video sample.mp4 --method chrom --freq lomb_scargle --out result.csv

# С конфигом эксперимента и структурированным JSONL-логом по окнам
python scripts/run_on_video.py --video sample.mp4 --config experiment.yaml --out result.csv --log result.jsonl
```

`--out result.csv` дополнительно сохраняет `result.config.yaml` — точный
конфиг, с которым получен именно этот результат (п.42: "каждая строчка
таблицы в статье должна быть привязана к конкретному конфигу").

Программный API:

```python
from rppg.config import PipelineConfig, ExtractionMethod
from rppg.pipeline import RPPGPipeline
import cv2

config = PipelineConfig(method=ExtractionMethod.POS)
cap = cv2.VideoCapture(0)

with RPPGPipeline(config, log_path="session.jsonl") as pipeline:
    ok, frame = cap.read()
    result = pipeline.process_frame(frame, timestamp_ms=0)
    if result and result.publishable:
        print(result.bpm, result.hrv.sdnn_ms, result.hrv.rmssd_ms)
    elif result:
        print("низкое качество сигнала:", result.warnings)
```

## Валидация и бенчмарки

Реальных публичных датасетов (UBFC-rPPG/PURE/COHFACE/VIPL-HR/UBFC-Phys) в
среде разработки этого кода нет — большинство требует Data Use Agreement,
подписываемый исследователем лично (см. `benchmark/evaluate.py`, докстринг).
Поэтому ниже — полностью рабочая ИНФРАСТРУКТУРА валидации, прогнанная на
синтетических данных с известным BPM; реальные числа для статьи требуют
подключения настоящего датасета через `DatasetEvaluator`.

- `benchmark/evaluate.py` — MAE/RMSE/MAPE/Pearson/CCC/Bland-Altman,
  reference-relative SNR, агрегация ПО ИСПЫТУЕМЫМ с bootstrap CI (не по
  окнам — псевдорепликация, см. докстринги), кривая "покрытие vs ошибка"
  для SQI-гейтинга, стратифицированный анализ.
- `benchmark/skin_tone.py` — объективная ITA-оценка тона кожи для
  стратификации без ручной Fitzpatrick-разметки.
- `scripts/run_ablation_study.py` — полная таблица 6 методов × 3 частотных
  оценки × 3 detrend × 2 режима ROI на синтетике; результат сохраняется в
  `benchmark/ablation_results.csv`.
- `scripts/compare_fusion_vs_argmax.py` — SQI-взвешенный fusion против
  argmax-выбора ROI на синтетике с калиброванным по модальностям шумом.
- `scripts/analyze_tarvainen_lambda.py` — АЧХ detrending-фильтра и
  обоснование выбора λ под fs=30 Гц; график в
  `benchmark/tarvainen_frequency_response.png`.
- `scripts/benchmark_performance.py` — замеры производительности (по
  стадиям, Numba cold/warm, CPU/GPU delegate, сквозной FPS) на конкретном
  железе; результат в `benchmark/performance_report.json`.

## Тесты

```bash
pytest                                   # вся папка tests/, конфиг в pytest.ini
pytest tests/test_signal_pipeline.py -v  # подробный вывод по одному файлу
```

Сигнальный уровень (детрендинг → 6 методов извлечения → 3 метода частотного
анализа → HRV → SQI, включая fusion/harmonic/flicker) проверен на
синтетических, но физиологически правдоподобных данных с известным BPM.
Отдельно — сквозной тест ЧЕРЕЗ `RPPGPipeline.process_frame` целиком (на
кадрах с искусственно пульсирующим цветом "кожи") и негативные контроли
(статичное фото, манекен/шум, белый шум, полностью окклюдированное лицо —
во всех случаях `publishable == False`). MediaPipe и модель в этой среде
доступны — pipeline-уровневые тесты запускают его по-настоящему, но
грациозно пропускаются, если в другой среде их нет.

CI (`.github/workflows/ci.yml`) прогоняет тот же `pytest` на push/PR на
Python 3.9/3.11/3.12.

## Структура проекта

```
src/rppg/
  config.py            конфигурация (датаклассы), включая FusionConfig
  config_io.py          YAML/JSON (де)сериализация PipelineConfig (п.42)
  structured_log.py     JSONL-лог по окнам: BPM/SQI-компоненты/warnings (п.43)
  face/
    landmarker.py       обёртка над MediaPipe Face Landmarker
    roi.py                ROI (лоб/щёки) + фоновый ROI для детектора мерцания
  signal/
    preprocessing.py    detrend (+ АЧХ-анализ), normalize, bandpass
    methods.py            GREEN/CHROM/POS/PCA/ICA/HEAD_MOTION
    quality.py             Signal Quality Index (4 компоненты + harmonic/flicker гейты)
    fusion.py               SQI-взвешенное объединение сигналов ROI/модальностей
    frequency.py             FFT/Welch/Lomb-Scargle
  hrv/
    features.py          Mean HR, SDNN, RMSSD, pNN50, LF/HF
  accel/
    fast_ops.py           Numba-ускорение POS (опционально, с numpy fallback)
  pipeline.py              оркестрация + SQI-gating перед публикацией в ПТСР-систему
scripts/                   CLI: веб-камера, офлайн-видео, ablation/fusion/performance/λ-анализ
benchmark/
  evaluate.py              метрики + загрузчик UBFC-rPPG + subject-level агрегация
  skin_tone.py              ITA-оценка тона кожи
tests/                     pytest, сигнальный + pipeline-уровень
docs/research_report.md    полный анализ, архитектура, эксперименты
```

## Ограничения текущей версии (см. research_report.md за подробностями)

- Валидация выполнена на синтетических сигналах; прогон на реальных
  датасетах (VIPL-HR/UBFC-rPPG/PURE/MMSE-HR/UBFC-Phys) требует их
  скачивания по data use agreement и не выполнялся в среде разработки
  этого кода — см. раздел "Валидация и бенчмарки" выше.
- Результаты SQI-взвешенного fusion и ablation-таблица получены на
  синтетике; величина реального выигрыша (или его отсутствие) на реальных
  видео не подтверждена.
- HRV-признаки вычисляются из межпульсовых, а не межсердечных (ЭКГ)
  интервалов (PRV, а не HRV в строгом смысле) — см. hrv/features.py, docstring.
- Признаки, полезные для ПТСР-скрининга (HRV/RMSSD/LF-HF), являются
  неспецифичным физиологическим коррелятом, а не диагностическим критерием —
  см. research_report.md, раздел 7.

## Acknowledgements

Научной отправной точкой для этого проекта послужил
[irfan798/head-pulse-track](https://github.com/irfan798/head-pulse-track) —
он первым в этой цепочке указал на классическую работу Balakrishnan,
Durand & Guttag (CVPR 2013, "Detecting Pulse from Head Motions in Video")
как на подход к задаче. Этот код построен независимо: MediaPipe Face
Landmarker (Tasks API) вместо legacy Face Mesh, все 6 методов извлечения
сигнала реализованы напрямую по первоисточникам (см. полный список в
[CITATION.cff](CITATION.cff)), без переиспользования кода irfan798/head-pulse-track.
Тот репозиторий не несёт лицензии (все права на него защищены его
авторами по умолчанию) — соответственно, здесь нет никакого его кода, а
есть только признание научного приоритета указания на нужный алгоритм.

Дополнительные алгоритмические источники: de Haan & Jeanne (2013, CHROM),
Wang et al. (2016, POS), Poh, McDuff & Picard (2010-2011, ICA),
Lewandowska et al. (2011, PCA), Verkruysse, Svaasand & Nelson (2008,
GREEN), Tarvainen, Ranta-aho & Karjalainen (2002, detrending) — точные
библиографические ссылки в [CITATION.cff](CITATION.cff) и в докстринге
каждого соответствующего модуля.
