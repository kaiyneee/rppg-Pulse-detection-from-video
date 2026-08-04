"""
Задача 3б: почти бесплатный референс "на себе" — контактный PPG с пальца
(см. scripts/finger_ppg.py) рядом с лицевым rPPG.

ЧЕСТНАЯ ОГОВОРКА про синхронизацию: веб-камера (лицо) и телефон (палец,
прижатый к объективу+вспышке) — ДВА независимых устройства с независимыми
часами. Нет способа синхронизировать их программно из одного скрипта на
одном устройстве. Вместо этого используется РУЧНАЯ синхронизация по
обратному отсчёту: команда `record` показывает "3... 2... 1... СТАРТ" на
экране И подаёт звуковой сигнал ровно в момент "СТАРТ" — начните запись на
телефоне В ЭТОТ МОМЕНТ. Погрешность порядка нескольких сотен миллисекунд
(человеческая реакция) — для оценок BPM по 10-секундным окнам это
пренебрежимо мало; если нужно точнее, `compare` принимает --offset-seconds
для ручной подстройки по общему событию на обеих записях (хлопок в ладоши
хорошо виден на лицевом видео и слышен/виден как скачок яркости от
вспышки на видео пальца, если направить телефон в сторону веб-камеры на
долю секунды перед началом записи пальца).

Два подкоманды:

    # 1. Запись лица (веб-камера) — покажет обратный отсчёт, затем
    #    запишет лицевой пайплайн live в JSONL до Ctrl+C/ESC или --duration.
    #    Одновременно (по сигналу "СТАРТ") начните запись пальца на телефоне.
    python scripts/record_with_reference.py record --log face.jsonl --duration 60

    # 2. После того как видео с пальца скопировано с телефона на диск:
    python scripts/record_with_reference.py compare \\
        --face-log face.jsonl --finger-video finger.mp4 --out comparison.png

Критерий приёмки (задача 3б): `compare` выдаёт ОДИН график с двумя кривыми
BPM (лицо и палец) и число MAE между ними.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from rppg.config import PipelineConfig, ExtractionMethod, FrequencyMethod
from rppg.pipeline import RPPGPipeline
from finger_ppg import process_finger_video
from benchmark.evaluate import mae as mae_metric


# --------------------------------------------------------------------------- #
# record: веб-камера с обратным отсчётом для ручной синхронизации
# --------------------------------------------------------------------------- #

def _draw_countdown(frame, seconds_left: float) -> None:
    h, w = frame.shape[:2]
    if seconds_left > 0:
        text = str(int(np.ceil(seconds_left)))
        color = (0, 165, 255)
    else:
        text = "START"
        color = (0, 220, 0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6)
    cv2.putText(frame, "Начните запись пальца на телефоне СЕЙЧАС", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def cmd_record(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    if args.method is not None:
        config.method = ExtractionMethod(args.method)
    if args.freq is not None:
        config.frequency_method = FrequencyMethod(args.freq)
    config.face.model_asset_path = args.model

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {args.camera}")

    print(f"Обратный отсчёт {args.countdown_seconds:.0f}с — приготовьте телефон для записи пальца "
          "(прижмите к объективу+включите фонарик), начните запись РОВНО на 'START'.")

    countdown_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        elapsed = time.time() - countdown_start
        seconds_left = args.countdown_seconds - elapsed
        _draw_countdown(frame, seconds_left)
        cv2.imshow("record_with_reference (ESC to abort)", frame)
        if seconds_left <= 0:
            sys.stdout.write("\a")  # звуковой сигнал терминала на "START"
            sys.stdout.flush()
            break
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Отменено.")
            return

    record_start_ms = time.time() * 1000
    print(f"СТАРТ. Пишу {args.duration:.0f}с в {args.log} (ESC — закончить раньше).")

    with RPPGPipeline(config, log_path=args.log) as pipeline:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(time.time() * 1000 - record_start_ms)
            if timestamp_ms > args.duration * 1000:
                break
            result = pipeline.process_frame(frame, timestamp_ms)
            cv2.putText(frame, f"REC {timestamp_ms/1000:.1f}/{args.duration:.0f}s", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if result is not None:
                cv2.putText(frame, f"BPM: {result.bpm:.0f} (publishable={result.publishable})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0) if result.publishable else (0, 165, 255), 2)
            cv2.imshow("record_with_reference (ESC to abort)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Готово. Лицевой лог: {args.log}")
    print("Теперь скопируйте видео с пальца с телефона на диск и запустите:")
    print(f"  python scripts/record_with_reference.py compare --face-log {args.log} --finger-video <finger.mp4>")


# --------------------------------------------------------------------------- #
# compare: две кривые BPM на одном графике + MAE (критерий приёмки задачи 3б)
# --------------------------------------------------------------------------- #

def _load_face_bpm(face_log: str | None, face_video: str | None, model_path: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Возвращает (timestamp_s, bpm, publishable) из JSONL-лога (см.
    structured_log.py) или, если лога нет, из свежего прогона RPPGPipeline
    по face_video."""
    import pandas as pd

    if face_log is not None:
        df = pd.read_json(face_log, lines=True)
        t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
        bpm = df["bpm"].to_numpy(dtype=float)
        publishable = df["publishable"].to_numpy(dtype=bool)
        return t, bpm, publishable

    config = PipelineConfig()
    config.face.model_asset_path = model_path
    cap = cv2.VideoCapture(face_video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть {face_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.window.assumed_fps
    config.window.assumed_fps = fps

    ts, bpm_list, pub_list = [], [], []
    with RPPGPipeline(config) as pipeline:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx / fps * 1000)
            result = pipeline.process_frame(frame, timestamp_ms)
            if result is not None and not np.isnan(result.bpm):
                ts.append(timestamp_ms / 1000.0)
                bpm_list.append(result.bpm)
                pub_list.append(result.publishable)
            frame_idx += 1
    cap.release()
    return np.array(ts), np.array(bpm_list), np.array(pub_list, dtype=bool)


def cmd_compare(args: argparse.Namespace) -> None:
    if args.face_log is None and args.face_video is None:
        raise SystemExit("Нужен либо --face-log (JSONL), либо --face-video (сырое видео)")

    print("Обрабатываю лицевую запись...")
    face_t, face_bpm, face_publishable = _load_face_bpm(args.face_log, args.face_video, args.model)
    if len(face_t) == 0:
        raise SystemExit("Лицевая запись не дала ни одной оценки BPM")

    print("Обрабатываю видео пальца (референс)...")
    finger_samples = process_finger_video(
        args.finger_video, method_name=args.finger_method, frequency_method=args.freq,
        window_seconds=args.window_seconds, step_seconds=args.step_seconds,
    )
    if not finger_samples:
        raise SystemExit(f"Видео пальца короче {args.window_seconds}с — нет ни одной оценки")
    finger_t = np.array([s.timestamp_ms / 1000.0 for s in finger_samples]) + args.offset_seconds
    finger_bpm = np.array([s.bpm for s in finger_samples])

    # Интерполируем референс (палец) на моменты времени лицевых оценок —
    # только там, где обе записи реально пересекаются во времени (иначе
    # np.interp молча экстраполирует константой на краях, что дало бы
    # обманчиво "согласующиеся" точки за пределами общей записи).
    overlap_mask = (face_t >= finger_t.min()) & (face_t <= finger_t.max())
    if not overlap_mask.any():
        raise SystemExit(
            "Записи не пересекаются во времени — проверьте --offset-seconds "
            f"(лицо: [{face_t.min():.1f}, {face_t.max():.1f}]с, "
            f"палец: [{finger_t.min():.1f}, {finger_t.max():.1f}]с)"
        )
    finger_bpm_at_face_t = np.interp(face_t, finger_t, finger_bpm)

    mae_all = mae_metric(finger_bpm_at_face_t[overlap_mask], face_bpm[overlap_mask])
    pub_overlap = overlap_mask & face_publishable
    mae_published = (
        mae_metric(finger_bpm_at_face_t[pub_overlap], face_bpm[pub_overlap])
        if pub_overlap.any() else float("nan")
    )

    print(f"\nПересекающихся окон лица: {int(overlap_mask.sum())}/{len(face_t)} "
          f"(из них publishable=True: {int(pub_overlap.sum())})")
    print(f"MAE (лицо vs палец), ВСЕ пересекающиеся окна:          {mae_all:.2f} BPM")
    print(f"MAE (лицо vs палец), только publishable=True окна лица: {mae_published:.2f} BPM")

    _plot_comparison(face_t, face_bpm, face_publishable, finger_t, finger_bpm, mae_all, mae_published, args.out)
    print(f"\nГрафик сохранён в {args.out}")


def _plot_comparison(face_t, face_bpm, face_publishable, finger_t, finger_bpm, mae_all, mae_published, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Палитра — см. dataviz skill / scripts/analyze_log.py: статус-цвета
    # фиксированы и не переиспользуются как идентичность серии.
    COLOR_GOOD = "#0ca30c"
    COLOR_CRITICAL = "#d03b3b"
    COLOR_REF = "#2a78d6"
    INK_PRIMARY = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    GRIDLINE = "#e1e0d9"

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#fcfcfb")
    ax.plot(finger_t, finger_bpm, color=COLOR_REF, linewidth=2.0,
            label="референс: палец (контактный PPG)", zorder=2)
    ax.scatter(face_t[face_publishable], face_bpm[face_publishable], color=COLOR_GOOD, s=18, zorder=3,
               label="лицо, publishable=True")
    ax.scatter(face_t[~face_publishable], face_bpm[~face_publishable], color=COLOR_CRITICAL, s=18, zorder=3,
               label="лицо, publishable=False")

    ax.set_xlabel("время, с", color=INK_SECONDARY)
    ax.set_ylabel("BPM", color=INK_SECONDARY)
    ax.set_title(
        f"Лицо vs референс (палец): MAE={mae_all:.2f} BPM (все), "
        f"{mae_published:.2f} BPM (только publishable=True)",
        color=INK_PRIMARY, fontsize=11,
    )
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="best", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Записать лицевую веб-камеру с обратным отсчётом для ручной синхронизации")
    p_record.add_argument("--camera", type=int, default=0)
    p_record.add_argument("--model", type=str, default="models/face_landmarker.task")
    p_record.add_argument("--method", type=str, default=None, choices=[m.value for m in ExtractionMethod])
    p_record.add_argument("--freq", type=str, default=None, choices=[m.value for m in FrequencyMethod])
    p_record.add_argument("--log", type=str, required=True, help="Куда писать JSONL лицевого пайплайна")
    p_record.add_argument("--duration", type=float, default=120.0, help="Длительность записи, с")
    p_record.add_argument("--countdown-seconds", type=float, default=5.0)
    p_record.set_defaults(func=cmd_record)

    p_compare = sub.add_parser("compare", help="Сравнить лицевой BPM с референсом (пальцем) — критерий приёмки задачи 3б")
    p_compare.add_argument("--face-log", type=str, default=None, help="JSONL из record (или run_webcam.py --log)")
    p_compare.add_argument("--face-video", type=str, default=None, help="Альтернатива --face-log: сырое видео лица")
    p_compare.add_argument("--model", type=str, default="models/face_landmarker.task")
    p_compare.add_argument("--finger-video", type=str, required=True)
    p_compare.add_argument("--finger-method", type=str, default="green",
                            choices=[m.value for m in ExtractionMethod if m.value not in ("head_motion", "auto")])
    p_compare.add_argument("--freq", type=str, default="welch", choices=[m.value for m in FrequencyMethod])
    p_compare.add_argument("--window-seconds", type=float, default=10.0)
    p_compare.add_argument("--step-seconds", type=float, default=1.0)
    p_compare.add_argument("--offset-seconds", type=float, default=0.0,
                            help="Сдвиг таймлайна пальца относительно лица (с) — ручная подстройка синхронизации")
    p_compare.add_argument("--out", type=str, default="comparison.png")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
