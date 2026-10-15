"""
Оценивает ОДНОГО испытуемого UBFC-rPPG и накапливает результат в общий
файл — предназначен для последовательных прогонов "скачал папку
испытуемого -> прогнал -> удалил папку -> скачал следующую", поэтому
результаты предыдущих запусков НИКОГДА не теряются и не перезаписываются
целиком: правится только строка текущего испытуемого.

Вызов (ровно один аргумент, всё остальное скрипт определяет сам):
    PYTHONPATH=src python3 scripts/eval_ubfc_subject.py --subject-dir "/Users/ylhjn/Downloads/8-gt"

Не меняет НИЧЕГО в самом пайплайне — это инструмент оценки, не правка
алгоритма (RPPGPipeline вызывается с конфигом по умолчанию, как есть).

Формат эталона определяется по имени файла в папке испытуемого:
  - gtdump.xmp        — CSV без заголовка, 4 колонки: время_мс, ЧСС(уд/мин),
                         сатурация, отсчёт PPG (подтверждено по реальному
                         файлу — см. git history этого скрипта).
  - ground_truth.txt  — формат DATASET_2 UBFC-rPPG. НЕ РЕАЛИЗОВАН: файлы
                         readme.txt/ubfcrppg_data_processor.py из корня
                         UBFC_DATASET, из которых нужно брать точную
                         структуру, на этой машине не найдены нигде —
                         угадывать формат эталона недопустимо (см. итог
                         задачи по этому скрипту). Даёт понятную ошибку.

Видео ищется как vid.avi ИЛИ vid-001.avi (оба варианта встречаются в
реально скачанных архивах испытуемых) — иначе первый файл по маске vid*.avi.

Метрики (MAE/RMSE/MAPE/корреляция Пирсона) считаются в ДВУХ разрезах: по
всем окнам с валидной парой (предсказание, эталон) и отдельно только по
окнам с publishable=True — коэффициент покрытия и диапазон/медиана эталона
задаются ОДНИМ числом на испытуемого (не дублируются по разрезам: это
чисто описательные величины записи целиком, а не следствие ошибки модели).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from rppg.config import DEFAULT_PRESET, PRESET_OVERRIDES, PipelineConfig, make_pipeline_config
from rppg.pipeline import RPPGPipeline, apply_second_pass
from rppg.signal.preprocessing import preprocess_signal, tarvainen_cutoff_hz, tarvainen_lambda_for_cutoff
from rppg.signal.frequency import estimate_hr
from rppg.signal.quality import dominant_frequency_and_snr
from reference_full10s import reference_full10s  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Однопроходные и двухпроходные результаты пишутся в РАЗНЫЕ файлы: смешивать
# их в одной таблице нельзя (это две разные методики измерения), а сравнение
# "что дал второй проход" — основная причина, по которой обе нужны рядом.
# main() переставляет эти пути на *_twopass при --two-pass.
RESULTS_CSV = REPO_ROOT / "ubfc_results.csv"
WINDOWS_DIR = REPO_ROOT / "ubfc_windows"
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

# Задача: диапазон эталона у испытуемого меньше этого числа ударов ->
# корреляция Пирсона считается по шуму почти постоянного сигнала и
# помечается неинформативной, а не просто выводится как есть.
MIN_INFORMATIVE_RANGE_BPM = 5.0

RESULTS_FIELDS = [
    "subject",
    "n_windows", "n_windows_publishable", "coverage",
    "gt_min_bpm", "gt_max_bpm", "median_snr_db",
    "mae_all", "rmse_all", "mape_all_pct", "pearson_r_all", "pearson_r_all_informative",
    "mae_pub", "rmse_pub", "mape_pub_pct", "pearson_r_pub", "pearson_r_pub_informative",
    # Основной эталон (полное 10-секундное окно, фиксированный способ, см.
    # benchmark/reference_full10s.py) и пресет, которым получена строка.
    "mae_pub_full10s", "preset",
]

WINDOW_FIELDS = [
    # Задача 53: bpm_ref_wave — эталон из контактной ВОЛНЫ, посчитанный той же
    # процедурой, что и оценка (участвует в метриках); bpm_ref_device — прежняя
    # колонка ЧСС прибора, оставлена рядом, т.к. расхождение между ними —
    # самостоятельный результат; ref_wave_snr_db — качество самой волны в окне.
    "second", "bpm_pred", "bpm_ref_wave", "bpm_ref_device", "ref_wave_snr_db", "bpm_ref_full10s",
    "error", "sqi_score", "publishable",
    # Задача 51, пункт 1: отдельные компоненты SQI — нужны для проверки,
    # какая (если хоть одна) из них реально предсказывает ошибку на уровне
    # ОКНА, а не только участвует в свёрнутом sqi_score. Все берутся из
    # PTSDPulseFeatures.confidence_breakdown — уже посчитаны пайплайном,
    # пайплайн не трогается.
    "spectral_snr_db", "harmonic_score", "cross_roi_agreement",
    "landmark_stability", "temporal_consistency", "flicker_suspected",
]


class SubjectDataError(RuntimeError):
    """Видео/эталон не найдены или формат эталона не поддержан — по этой
    ошибке скрипт обязан завершиться БЕЗ единой записи в ubfc_results.csv."""


def resolve_subject_id(subject_dir: Path) -> str:
    name = subject_dir.name
    return name[:-3] if name.endswith("-gt") else name


def find_video(subject_dir: Path) -> Path:
    for name in ("vid.avi", "vid-001.avi"):
        candidate = subject_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(subject_dir.glob("vid*.avi"))
    if matches:
        return matches[0]
    raise SubjectDataError(
        f"Видео не найдено в {subject_dir} (искали vid.avi, vid-001.avi, vid*.avi)"
    )


def find_ground_truth(subject_dir: Path) -> Path:
    gtdump = subject_dir / "gtdump.xmp"
    if gtdump.exists():
        return gtdump
    gt_txt = subject_dir / "ground_truth.txt"
    if gt_txt.exists():
        return gt_txt
    raise SubjectDataError(
        f"Файл эталона не найден в {subject_dir} (искали gtdump.xmp, ground_truth.txt)"
    )


def load_reference_waveform(gt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (t_sec, ppg_wave, hr_device) — время от начала записи,
    КОНТАКТНАЯ ПУЛЬСОВАЯ ВОЛНА и ЧСС, посчитанная самим прибором.

    Задача 53. Эталоном служит ВОЛНА, а не колонка ЧСС. Авторы UBFC-rPPG
    пишут в readme.txt дословно: "heart rate values provided by the sensor
    were not used for evaluating our heart rate estimation method. Instead,
    our evaluation compared heart rate estimations derived from the remote
    PPG signal with estimations calculated from the contact PPG signal."
    Сравнение с колонкой прибора несопоставимо ни с публикацией авторов, ни
    с работами, которые на этот датасет ссылаются (rPPG-Toolbox и др.).

    Формат ВЗЯТ ИЗ readme.txt и ubfcrppg_data_processor.py, приложенных к
    датасету (а не выведен из данных):
      DATASET_1, gtdump.xmp     — колонки: 1 время (мс), 2 ЧСС, 3 SpO2, 4 волна
      DATASET_2, ground_truth.txt — строки:  1 волна, 2 ЧСС, 3 время (с)
    hr_device возвращается только для того, чтобы сохранить её в пооконный
    лог рядом с новым эталоном (колонка bpm_ref_device) — расхождение между
    ними самостоятельный результат, но в метриках она больше не участвует.
    """
    if gt_path.name == "gtdump.xmp":
        data = np.loadtxt(gt_path, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 4:
            raise SubjectDataError(
                f"{gt_path}: ожидалось >=4 колонок (время_мс, ЧСС, SpO2, волна), получено {data.shape}"
            )
        t_ms, hr_device, wave = data[:, 0], data[:, 1], data[:, 3]
        return (t_ms - t_ms[0]) / 1000.0, wave, hr_device

    if gt_path.name == "ground_truth.txt":
        rows = np.loadtxt(gt_path)
        if rows.ndim != 2 or rows.shape[0] < 3:
            raise SubjectDataError(
                f"{gt_path}: ожидалось 3 строки (волна, ЧСС, время), получено {rows.shape}"
            )
        wave, hr_device, t_sec = rows[0], rows[1], rows[2]
        if not np.all(np.diff(t_sec) >= 0):
            raise SubjectDataError(f"{gt_path}: строка времени не монотонна — формат не тот, что ожидался")
        return t_sec - t_sec[0], wave, hr_device

    raise SubjectDataError(f"Неизвестный формат файла эталона: {gt_path.name}")


def reference_bpm_from_wave(
    t_sec: np.ndarray,
    wave: np.ndarray,
    window_end_s: float,
    window_seconds: float,
    cfg: PipelineConfig,
    method: str = "welch",
) -> tuple[float | None, float | None]:
    """Эталонная ЧСС окна, посчитанная из контактной волны ТОЙ ЖЕ
    процедурой, что и rPPG-оценка. Возвращает (bpm, snr_db волны).

    Задача 53. Принципиально, что процедура одна и та же: иначе сравниваются
    не два измерения одной величины, а два разных её определения (окно 10 с
    спектральной оценки против внутреннего сглаженного алгоритма прибора с
    собственным запаздыванием). Переиспользуются РОВНО те же функции, что
    работают в пайплайне, — preprocess_signal, estimate_hr,
    dominant_frequency_and_snr, — а не параллельная реализация.

    Частота дискретизации берётся из СОБСТВЕННОЙ временной оси волны (в
    DATASET_1 это ~62 Гц, в DATASET_2 ~30 Гц), а не из fps видео.

    tarvainen_lambda пересчитывается под эту fs: λ — не физическая величина,
    та же λ на другой fs даёт другую частоту среза (на 62 Гц λ=3542 дала бы
    0.207 Гц вместо 0.1 Гц, вдвое более агрессивный фильтр). Переносится
    ЧАСТОТА СРЕЗА, а не число — только так препроцессинг действительно "тот
    же" (см. tarvainen_lambda_for_cutoff).

    Механизмы непрерывности/физиологического предела скорости (continuity,
    rate limit) НЕ применяются: они подтягивают оценку к её собственной
    недавней истории, что для эталона недопустимо — он должен быть
    независимым измерением, а не сглаженным по своему же прошлому.
    """
    mask = (t_sec >= window_end_s - window_seconds) & (t_sec <= window_end_s)
    n = int(mask.sum())
    if n < 16:
        return None, None

    t_win, x_raw = t_sec[mask], np.asarray(wave[mask], dtype=float)
    span = float(t_win[-1] - t_win[0])
    if not np.isfinite(span) or span <= 0:
        return None, None
    if not np.isfinite(np.std(x_raw)) or np.std(x_raw) < 1e-12:
        return None, None

    # Приведение к РАВНОМЕРНОЙ сетке перед Уэлчем. В DATASET_1 волна и так
    # равномерна (шаг 0.016 с, std 0.0003 — 62.5 Гц), а вот в DATASET_2
    # временная ось прибора СИЛЬНО неравномерна: шаг гуляет от 0.010 до
    # 0.223 с при медиане ~0.040. Уэлч определён только на равномерной
    # сетке, и подстановка fs=1/median(dt) там искажала спектр эталона —
    # проверено: без этого шага DATASET_1 резко улучшался, а DATASET_2
    # portился (subject48 1.67 -> 12.66), то есть ошибка была не в смене
    # эталона, а в обработке.
    #
    # Именно интерполяция на равномерную сетку, а НЕ переход на
    # Ломба-Скаргла (который корректен на неравномерных данных, см.
    # frequency.py): если DATASET_2 считать другим методом, чем DATASET_1,
    # два подмножества снова окажутся несопоставимы между собой — ровно та
    # проблема, ради устранения которой эталон и переводится на волну.
    # Метод остаётся один (Уэлч), одинаковый и для эталона, и для rPPG.
    #
    # Частота сетки — средняя частота отсчётов в окне ((n-1)/span): она не
    # добавляет отсчётов сверх реально измеренных и не выбрасывает их.
    fs = float(len(t_win) - 1) / span
    if not np.isfinite(fs) or fs <= 0:
        return None, None
    t_uniform = np.linspace(t_win[0], t_win[-1], len(t_win))
    x = np.interp(t_uniform, t_win, x_raw)

    band = (cfg.filt.low_hz, cfg.filt.high_hz)
    lam = cfg.filt.tarvainen_lambda
    if cfg.filt.detrend_method == "tarvainen":
        target_cutoff = tarvainen_cutoff_hz(cfg.filt.tarvainen_lambda, cfg.window.assumed_fps)
        lam = tarvainen_lambda_for_cutoff(target_cutoff, fs)

    processed = preprocess_signal(
        x, fs, band[0], band[1],
        order=cfg.filt.filter_order, detrend_method=cfg.filt.detrend_method,
        tarvainen_lambda=lam, normalize_method=cfg.filt.normalize_method,
    )
    # Эталон ПЕРВОЙ СТАТЬИ: метод фиксирован (по умолчанию Уэлч с одним
    # 8-секундным сегментом) и НЕ берётся из cfg.frequency_method — после
    # смены умолчаний оценщика (пресет fft_spectral_mean) эталон статьи
    # иначе молча сменил бы определение. Используется только для сверки со
    # статьями; основной эталон — benchmark/reference_full10s.py.
    est = estimate_hr(processed, fs, band, method=method)
    _, snr_db = dominant_frequency_and_snr(processed, fs, band)
    bpm = float(est.bpm) if np.isfinite(est.bpm) else None
    return bpm, (float(snr_db) if np.isfinite(snr_db) else None)


def reference_median_for_window(t_sec: np.ndarray, hr: np.ndarray, window_end_s: float, window_seconds: float):
    """Медиана ЧСС ПРИБОРА за окно.

    Задача 53: это больше НЕ эталон для метрик (им стала оценка из
    контактной волны, см. reference_bpm_from_wave) — только справочная
    колонка bpm_ref_device в пооконном логе, чтобы расхождение между двумя
    определениями эталона было видно прямо в данных.

    Прежний фильтр достоверности (30-220 уд/мин) здесь УБРАН вместе с
    переходом на волну: он вводился post hoc из-за того, что у трёх записей
    прибор выдавал 1-3 удара в минуту. Это сбои алгоритма расчёта ЧСС внутри
    прибора, а не отсутствие сигнала — сама волна в тех же окнах пригодна.
    Признаком непригодного окна теперь служит низкий SNR САМОЙ ВОЛНЫ
    (ref_wave_snr_db), а не неправдоподобное значение ЧСС.
    """
    mask = (t_sec >= window_end_s - window_seconds) & (t_sec <= window_end_s)
    if not mask.any():
        return None
    return float(np.median(hr[mask]))


def run_pipeline(video_path: Path, two_pass: bool = False, preset: str = DEFAULT_PRESET) -> list:
    """two_pass=True — штатный режим ОФЛАЙН-обработки (см.
    rppg.pipeline.apply_second_pass и scripts/run_on_video.py): после
    причинного первого прохода окна, которым не хватило истории, ещё раз
    сверяются с итоговой историей за всю запись. Требует debug=True, иначе
    флаг history_was_insufficient недоступен."""
    config = make_pipeline_config(preset)
    config.face.model_asset_path = str(MODEL_PATH)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SubjectDataError(f"Не удалось открыть видео {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps

    rows = []
    frame_idx = 0
    with RPPGPipeline(config, debug=two_pass) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None:
                rows.append(result)
            frame_idx += 1

        if two_pass:
            stats = apply_second_pass(
                rows, pipeline._trusted_bpm_history, pipeline.cfg.frequency.history_min_windows,
            )
            print(f"  второй проход: пересмотрено {stats.n_reconsidered}, изменено {stats.n_changed} "
                  f"(кандидат {stats.n_changed_by_candidate}, флип x2 {stats.n_changed_by_ratio})")
    cap.release()
    return rows, config.window.window_seconds


def build_window_records(
    rows: list, window_seconds: float, t_ref: np.ndarray, wave: np.ndarray,
    hr_device: np.ndarray, cfg: PipelineConfig,
) -> list[dict]:
    records = []
    for r in rows:
        window_end_s = r.timestamp_ms / 1000.0
        ref, ref_snr = reference_bpm_from_wave(t_ref, wave, window_end_s, window_seconds, cfg)
        ref_device = reference_median_for_window(t_ref, hr_device, window_end_s, window_seconds)
        ref_full = reference_full10s(t_ref, wave, window_end_s)
        pred = r.bpm
        has_pred = pred is not None and np.isfinite(pred)
        error = (pred - ref) if (has_pred and ref is not None) else None
        cb = r.confidence_breakdown
        records.append({
            "second": window_end_s,
            "bpm_pred": pred if has_pred else None,
            "bpm_ref_wave": ref,
            "bpm_ref_device": ref_device,
            "bpm_ref_full10s": None if ref_full is None else ref_full["bpm_ref"],
            "ref_wave_snr_db": ref_snr,
            "error": error,
            "sqi_score": r.sqi_score,
            "publishable": r.publishable,
            "spectral_snr_db": cb.get("spectral_snr_db"),
            "harmonic_score": cb.get("harmonic_score"),
            "cross_roi_agreement": cb.get("cross_roi_agreement"),
            "landmark_stability": cb.get("landmark_stability"),
            "temporal_consistency": cb.get("temporal_consistency"),
            "flicker_suspected": cb.get("flicker_suspected"),
        })
    return records


def _pearson_with_informativeness(pred: np.ndarray, ref: np.ndarray) -> tuple[float | None, bool]:
    gt_range = float(ref.max() - ref.min()) if len(ref) else 0.0
    informative = gt_range >= MIN_INFORMATIVE_RANGE_BPM
    if len(pred) < 2 or np.std(pred) < 1e-9 or np.std(ref) < 1e-9:
        return None, informative
    r = float(np.corrcoef(pred, ref)[0, 1])
    return r, informative


def _error_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    if len(pred) == 0:
        return dict(mae=None, rmse=None, mape_pct=None, pearson_r=None, pearson_r_informative=False)
    err = pred - ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape_pct = float(np.mean(np.abs(err) / ref) * 100.0) if np.all(ref > 0) else None
    r, informative = _pearson_with_informativeness(pred, ref)
    return dict(mae=mae, rmse=rmse, mape_pct=mape_pct, pearson_r=r, pearson_r_informative=informative)


def compute_subject_summary(subject: str, records: list[dict], preset: str = DEFAULT_PRESET) -> dict:
    valid_all = [w for w in records if w["bpm_pred"] is not None and w["bpm_ref_wave"] is not None]
    valid_pub = [w for w in valid_all if w["publishable"]]

    pred_all = np.array([w["bpm_pred"] for w in valid_all], dtype=float)
    ref_all = np.array([w["bpm_ref_wave"] for w in valid_all], dtype=float)
    pred_pub = np.array([w["bpm_pred"] for w in valid_pub], dtype=float)
    ref_pub = np.array([w["bpm_ref_wave"] for w in valid_pub], dtype=float)

    snr_values = np.array(
        [w["spectral_snr_db"] for w in records if w["spectral_snr_db"] is not None and np.isfinite(w["spectral_snr_db"])],
        dtype=float,
    )

    n_windows = len(records)
    n_pub = sum(1 for w in records if w["publishable"])

    m_all = _error_metrics(pred_all, ref_all)
    m_pub = _error_metrics(pred_pub, ref_pub)

    return {
        "subject": subject,
        "n_windows": n_windows,
        "n_windows_publishable": n_pub,
        "coverage": (n_pub / n_windows) if n_windows else 0.0,
        "gt_min_bpm": float(ref_all.min()) if len(ref_all) else None,
        "gt_max_bpm": float(ref_all.max()) if len(ref_all) else None,
        "median_snr_db": float(np.median(snr_values)) if len(snr_values) else None,
        "mae_all": m_all["mae"], "rmse_all": m_all["rmse"], "mape_all_pct": m_all["mape_pct"],
        "pearson_r_all": m_all["pearson_r"], "pearson_r_all_informative": m_all["pearson_r_informative"],
        "mae_pub": m_pub["mae"], "rmse_pub": m_pub["rmse"], "mape_pub_pct": m_pub["mape_pct"],
        "pearson_r_pub": m_pub["pearson_r"], "pearson_r_pub_informative": m_pub["pearson_r_informative"],
        "mae_pub_full10s": _mae_full10s(records),
        "preset": preset,
    }


def _mae_full10s(records: list[dict]) -> float | None:
    """MAE по опубликованным окнам против основного эталона (полное окно)."""
    errs = [abs(w["bpm_pred"] - w["bpm_ref_full10s"]) for w in records
            if w["publishable"] and w["bpm_pred"] is not None and w.get("bpm_ref_full10s") is not None]
    return float(np.mean(errs)) if errs else None


def write_window_csv(subject: str, records: list[dict]) -> Path:
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WINDOWS_DIR / f"{subject}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        for w in records:
            writer.writerow({k: w[k] for k in WINDOW_FIELDS})
    return out_path


def upsert_results_csv(summary: dict) -> None:
    """Читает ubfc_results.csv целиком (если есть), заменяет строку ТОЛЬКО
    текущего испытуемого (или добавляет новую), пишет обратно. Все прочие
    строки переносятся дословно — задача явно запрещает трогать что-либо,
    кроме строки текущего испытуемого."""
    existing_rows: list[dict] = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    existing_rows = [row for row in existing_rows if row.get("subject") != summary["subject"]]
    existing_rows.append({k: summary[k] for k in RESULTS_FIELDS})

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def _fmt(x, spec=".2f") -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def print_subject_summary(summary: dict) -> None:
    print(f"\n=== Испытуемый {summary['subject']} ===")
    print(f"  окон: {summary['n_windows']} (publishable: {summary['n_windows_publishable']}, "
          f"покрытие: {summary['coverage'] * 100:.1f}%)")
    print(f"  эталон: {_fmt(summary['gt_min_bpm'], '.1f')}-{_fmt(summary['gt_max_bpm'], '.1f')} BPM, "
          f"медиана spectral_snr_db: {_fmt(summary['median_snr_db'])}")
    print(f"  {'разрез':<14s}{'MAE':>8s}{'RMSE':>8s}{'MAPE%':>8s}{'r':>8s}")
    r_all = _fmt(summary['pearson_r_all']) + ("" if summary['pearson_r_all_informative'] else " (неинф.)")
    r_pub = _fmt(summary['pearson_r_pub']) + ("" if summary['pearson_r_pub_informative'] else " (неинф.)")
    print(f"  {'все окна':<14s}{_fmt(summary['mae_all']):>8s}{_fmt(summary['rmse_all']):>8s}"
          f"{_fmt(summary['mape_all_pct']):>8s} {r_all:>14s}")
    print(f"  {'publishable':<14s}{_fmt(summary['mae_pub']):>8s}{_fmt(summary['rmse_pub']):>8s}"
          f"{_fmt(summary['mape_pub_pct']):>8s} {r_pub:>14s}")


def print_aggregate_summary() -> None:
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def col(name):
        out = []
        for row in rows:
            v = row.get(name)
            if v in (None, "", "None"):
                continue
            out.append(float(v))
        return np.array(out, dtype=float)

    n_subjects = len(rows)
    mae_pub = col("mae_pub")
    rmse_pub = col("rmse_pub")
    coverage = col("coverage")

    print(f"\n=== Накопленный итог по {n_subjects} испытуемым (ubfc_results.csv) ===")
    print("  (усреднение по ИСПЫТУЕМЫМ, не по окнам — иначе длинные записи получают больший вес)")
    if len(mae_pub):
        print(f"  MAE  (publishable): {mae_pub.mean():.2f} ± {mae_pub.std(ddof=1) if len(mae_pub) > 1 else 0.0:.2f}")
    if len(rmse_pub):
        print(f"  RMSE (publishable): {rmse_pub.mean():.2f} ± {rmse_pub.std(ddof=1) if len(rmse_pub) > 1 else 0.0:.2f}")
    if len(coverage):
        print(f"  Среднее покрытие: {coverage.mean() * 100:.1f}%")

    if n_subjects > 3 and len(mae_pub) == n_subjects:
        rng = np.random.default_rng(42)
        n_boot = 10000
        boot_means = np.array([
            rng.choice(mae_pub, size=n_subjects, replace=True).mean() for _ in range(n_boot)
        ])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"  Бутстрэп 95% ДИ для среднего MAE (n_boot={n_boot}): [{lo:.2f}, {hi:.2f}]")


def write_run_meta(path: Path, preset: str, two_pass: bool) -> None:
    """Метаданные прогона: пресет, полный конфиг, коммит, версии. Прогон
    идёт по одной записи за вызов — файл перезаписывается каждым вызовом;
    пресет КАЖДОЙ строки сохраняется ещё и в колонке preset ubfc_results*.csv."""
    import json
    import platform
    import subprocess
    from datetime import datetime, timezone

    from rppg.config_io import config_to_dict

    def _git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip() or None
        except OSError:
            return None

    meta = {
        "preset": preset,
        "two_pass": two_pass,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        "reference_main": "full10s-v1 (benchmark/reference_full10s.py) -> bpm_ref_full10s, mae_pub_full10s",
        "reference_paper": "Welch 8 s (reference_bpm_from_wave, method frozen) -> bpm_ref_wave, mae_pub",
        "pipeline_config": config_to_dict(make_pipeline_config(preset)),
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-dir", type=str, required=True)
    parser.add_argument(
        "--two-pass", action="store_true",
        help="Штатный офлайн-режим (см. rppg.pipeline.apply_second_pass): окна, которым не хватило "
             "истории в причинном проходе, пересматриваются по итоговой истории всей записи. "
             "Результаты пишутся в ubfc_results_twopass.csv / ubfc_windows_twopass/ — ОТДЕЛЬНО от "
             "однопроходных, чтобы обе методики можно было сравнить, а не подменить одну другой.",
    )
    parser.add_argument(
        "--preset", default=DEFAULT_PRESET, choices=sorted(PRESET_OVERRIDES),
        help="Пресет конфигурации пайплайна. legacy_welch8 — конфигурация, на которой получены "
             "результаты статей (git-тег papers-v1).",
    )
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT),
        help="Каталог для ubfc_results*.csv, ubfc_windows*/ и метаданных прогона "
             "(по умолчанию — корень репозитория, как раньше).",
    )
    args = parser.parse_args()

    global RESULTS_CSV, WINDOWS_DIR
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_twopass" if args.two_pass else ""
    RESULTS_CSV = out_dir / f"ubfc_results{suffix}.csv"
    WINDOWS_DIR = out_dir / f"ubfc_windows{suffix}"

    subject_dir = Path(args.subject_dir)
    if not subject_dir.is_dir():
        print(f"ОШИБКА: папка испытуемого не найдена: {subject_dir}", file=sys.stderr)
        sys.exit(1)

    subject = resolve_subject_id(subject_dir)

    try:
        video_path = find_video(subject_dir)
        gt_path = find_ground_truth(subject_dir)
        t_ref, ref_wave, hr_device = load_reference_waveform(gt_path)
    except SubjectDataError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        print("Ничего не записано в ubfc_results.csv.", file=sys.stderr)
        sys.exit(1)

    print(f"Испытуемый {subject}: видео={video_path.name}, эталон={gt_path.name}")
    # Тот же конфиг, что и у пайплайна — им препроцессируется эталонная волна.
    # assumed_fps ставится по РЕАЛЬНОМУ видео: от него считается целевая частота
    # среза детренда, которую надо перенести на fs контактной волны (см.
    # reference_bpm_from_wave); при умолчании 30.0 вместо фактических ~28.7
    # цель сместилась бы на 4%.
    eval_config = make_pipeline_config(args.preset)
    _cap = cv2.VideoCapture(str(video_path))
    _fps = _cap.get(cv2.CAP_PROP_FPS)
    _cap.release()
    if _fps and np.isfinite(_fps) and _fps > 0:
        eval_config.window.assumed_fps = float(_fps)
    rows, window_seconds = run_pipeline(video_path, two_pass=args.two_pass, preset=args.preset)
    if not rows:
        print(f"ОШИБКА: пайплайн не вернул ни одного окна для {video_path}", file=sys.stderr)
        print("Ничего не записано в ubfc_results.csv.", file=sys.stderr)
        sys.exit(1)

    records = build_window_records(rows, window_seconds, t_ref, ref_wave, hr_device, eval_config)
    summary = compute_subject_summary(subject, records, preset=args.preset)
    write_run_meta(out_dir / f"ubfc_run_meta{suffix}.json", args.preset, args.two_pass)

    windows_path = write_window_csv(subject, records)
    upsert_results_csv(summary)

    print_subject_summary(summary)
    print(f"  пооконный лог: {windows_path}")
    print(f"  сводная строка обновлена в {RESULTS_CSV}")

    print_aggregate_summary()


if __name__ == "__main__":
    main()
