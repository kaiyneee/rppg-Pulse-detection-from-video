# Метаданные петли условия B4

Продолжение `README.md`. Ни 550 прогонов, ни сам B4 не пересчитывались:
видео только сканируются ради числа кадров и меток контейнера, а эталон
фрагмента считается по контактной волне. Детектор пиков пайплайна не
использовался — только `scipy.signal.find_peaks` с параметрами из задания
(`distance` = 0,33 с, то есть 180 уд/мин; `prominence` = 0,3·СКО).

## Цитаты кода

Построение петли, `src/rppg/degrade.py`:

```python
164      cap = cv2.VideoCapture(video_path)
165      if not cap.isOpened():
166          raise RuntimeError(f"не удалось открыть видео для подготовки: {video_path}")
167      n_loop = int(round(LOOP_SECONDS * fps))
168      sampled: list[np.ndarray] = []
169      idx = 0
170      while True:
171          ok = cap.grab()
172          if not ok:
173              break
174          take_loop = needs_loop and idx < n_loop
175          take_sample = needs_frozen and idx % subsample == 0
176          if take_loop or take_sample:
177              ok, frame = cap.retrieve()
178              if not ok:
179                  break
180              if take_loop:
181                  prep.loop_frames.append(frame.copy())
182              if take_sample:
183                  sampled.append(frame)
184          idx += 1
185      cap.release()
186      prep.n_frames_seen = idx
       ...
246          if cond == "B4":
247              frames = self.prep.loop_frames
248              return frames[idx % len(frames)].astype(np.float32)
```

Временна́я ось офлайн-прогона, `benchmark/degenerate.py`:

```python
231      cap = cv2.VideoCapture(str(video_path))
232      if not cap.isOpened():
233          raise ev.SubjectDataError(f"не удалось открыть видео {video_path}")
234      fps = cap.get(cv2.CAP_PROP_FPS) or PipelineConfig().window.assumed_fps
235  
236      cfg = PipelineConfig()
237      cfg.face.model_asset_path = str(ev.MODEL_PATH)
238      cfg.window.assumed_fps = float(fps)
       ...
267          frame_idx = 0
268          while True:
269              ok, frame = cap.read()
270              if not ok:
271                  break
272              degraded = degrader(frame, frame_idx)
273              result = pipeline.process_frame(degraded, int(frame_idx / fps * 1000))
274              if result is not None:
275                  rows.append(result)
276              frame_idx += 1
```

Вывод частоты дискретизации из меток, `src/rppg/pipeline.py`:

```python
776      def _estimate_fps(self) -> float:
777          ts = np.array(self._buf.timestamps_ms, dtype=float)
778          if len(ts) < 2:
779              return self.cfg.window.assumed_fps
780          dt = np.diff(ts) / 1000.0
781          dt = dt[dt > 1e-3]
782          if len(dt) == 0:
783              return self.cfg.window.assumed_fps
784          return float(1.0 / np.median(dt))
```

## Ответы на три вопроса

**1. Число кадров в петле.** `n_loop = int(round(LOOP_SECONDS * fps))`,
строка 167 `degrade.py`, где `LOOP_SECONDS = 3.0`, а `fps` — частота
КОНТЕЙНЕРА (`cap.get(cv2.CAP_PROP_FPS)`, строка 234 `degenerate.py`).
Округление арифметическое (`round`), не усечение. Отбор идёт по индексу
кадра (`idx < n_loop`, строка 174), а не по временны́м меткам. Правило
выполняется для всех 50 записей.
Фактические значения: 70 кадров у 3 записей, 86 кадров у 9 записей, 87 кадров у 2 записей, 88 кадров у 6 записей, 89 кадров у 19 записей, 90 кадров у 11 записей.

**2. Временны́е метки на выходе B4 — синтезированные по частоте кадров,
сквозные по всей записи.** Кадр петли повторяется по содержимому
(`frames[idx % len(frames)]`, строка 248 `degrade.py`), но метка берётся не
из петли и не из контейнера: прогон подаёт
`int(frame_idx / fps * 1000)` (строка 273 `degenerate.py`), где `frame_idx`
монотонно растёт по всей записи. Метки исходной записи по порядку при этом
численно совпали бы с синтезированными: расхождение
`fps_container` и `fps_effective` по меткам контейнера равно
0,000000 на всех 50 записях. Существенно другое —
целочисленное усечение до миллисекунд, см. пункт 3.

**3. Уэлч получает частоту, выведенную ИЗ МЕТОК, а не `assumed_fps = 30.0`.**
`RPPGPipeline._estimate_fps` (строки 776–784) возвращает
`1 / median(diff(timestamps))`, и к `assumed_fps` обращается только если
меток меньше двух или все шаги нулевые — на этих данных такого не бывает.
Поскольку метки усечены до целых миллисекунд, медиана шага тоже целое число
миллисекунд, и `fs_used_hz` принимает лишь несколько дискретных значений:

| fs_used_hz | записей | медиана шага, мс |
|---|---|---|
| 23,255814 | 3 | 43 |
| 28,571429 | 10 | 35 |
| 29,411765 | 27 | 34 |
| 30,303030 | 10 | 33 |

Колонка `time_base_used` в выгрузке равна `timestamps` у всех 50 записей.

## Период петли

`loop_period_s` (разность поданных меток между выходными кадрами
`loop_n_frames` и 0) лежит в пределах 2,9840…3,0160 с
при медиане 2,9995 с. Ровно 3,000 с он составляет у
1 записи из 50 (`subject5`),
у остальных 49 отличается.

## Списки по критериям приёмки

**Критерий 3, `ref_n_peaks` вне диапазона 2…8:**
нет таких записей; фактический диапазон 2…6 пиков.

**Критерий 4, `|ref_hr_loop_bpm − ref_hr_g0_window0_bpm| > 15`:**
записей с таким расхождением — 3.

| запись | пульс петли, уд/мин | G0 окно 0, уд/мин | разница |
|---|---|---|---|
| `subject14` | 99,8 | 66,3 | 33,5 |
| `8` | 90,4 | 65,5 | 24,8 |
| `subject41` | 80,7 | 96,8 | 16,0 |

## Выгрузки

- `b4_loop_meta.csv` — 50 строк, по одной на запись, пустых значений нет.
- `b4_ref_peaks.csv` — 225 строк: сырые пики контактной волны на
  интервале петли, по которым посчитан `ref_hr_loop_bpm`.

Интервал для эталона взят по меткам ВИДЕО (реальное время записи, к которому
привязана контактная волна) — от первого кадра петли до кадра
`loop_n_frames − 1` включительно. Приведение к равномерной сетке повторяет
процедуру основной работы (`scripts/eval_ubfc_subject.py::reference_bpm_from_wave`):
сохраняются границы интервала и число отсчётов, интерполяция линейная.

## Воспроизведение

```bash
PYTHONPATH=src venv/bin/python3 benchmark/followup_b4_loop_meta.py
PYTHONPATH=src venv/bin/python3 benchmark/followup_b4_report.py
```
