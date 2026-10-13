"""
Кэш покадровых сигналов с лица и быстрое воспроизведение пайплайна.

Зачем. ~95% времени прогона записи уходит на MediaPipe и извлечение ROI,
а все рычаги точности (выбор метода, спектральная оценка, выбор пика,
трекинг, пороги качества) стоят ПОСЛЕ этого. Кэш хранит ровно то, что
RPPGPipeline кладёт в свой буфер на каждом кадре, — результат
extract_rois до проверки стабильности, флаг обнаружения лица и
отложенное предупреждение о маске кожи, — и позволяет прогнать
сигнальную часть пайплайна заново без видео за секунды.

Гарантия, на которой держится всё остальное: воспроизведение из кэша
обязано совпасть с полным прогоном ПОБИТОВО (verify). Если нет — кэш
непригоден, и любые эксперименты на нём недействительны.

    PYTHONPATH=src venv/bin/python3 benchmark/trace_cache.py build            # все записи
    PYTHONPATH=src venv/bin/python3 benchmark/trace_cache.py verify           # сверка с полным прогоном
"""

from __future__ import annotations

import argparse
import copy
import csv
import pickle
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import rppg.pipeline as pipeline_mod  # noqa: E402
from rppg.config import PipelineConfig  # noqa: E402
from rppg.pipeline import RPPGPipeline, apply_second_pass  # noqa: E402

import eval_ubfc_subject as ev  # noqa: E402

CACHE_DIR = REPO_ROOT / "cache" / "traces"
DUMMY_FRAME = np.zeros((2, 2, 3), dtype=np.uint8)


@dataclass
class FrameTrace:
    detected: bool
    roi: object | None = None          # ROIExtractionResult без landmarks_px
    skin_warning: str | None = None    # _pending_skin_mask_warning, выставленный на этом кадре


@dataclass
class RecordTrace:
    record_id: str
    fps: float
    timestamps_ms: list[int]
    frames: list[FrameTrace]
    t_ref: np.ndarray
    wave: np.ndarray
    hr_device: np.ndarray
    subset: int
    live: list[dict] = field(default_factory=list)  # итог полного прогона, для сверки


def cache_path(record_id: str) -> Path:
    return CACHE_DIR / f"{record_id}.pkl"


class _Unpickler(pickle.Unpickler):
    """Кэш, записанный при запуске этого файла как скрипта, ссылается на
    классы как на __main__.RecordTrace/__main__.FrameTrace; при загрузке
    из другого модуля их надо искать здесь."""

    def find_class(self, module, name):
        if module == "__main__" and name in ("RecordTrace", "FrameTrace"):
            return globals()[name]
        return super().find_class(module, name)


def load_trace(record_id: str) -> RecordTrace:
    with cache_path(record_id).open("rb") as fh:
        return _Unpickler(fh).load()


def make_config(fps: float) -> PipelineConfig:
    """Ровно тот конфиг, что использует scripts/eval_ubfc_subject.py."""
    cfg = PipelineConfig()
    cfg.face.model_asset_path = str(ev.MODEL_PATH)
    cfg.window.assumed_fps = float(fps)
    return cfg


def _row(r) -> dict:
    return {"timestamp_ms": r.timestamp_ms, "bpm": float(r.bpm), "publishable": bool(r.publishable),
            "sqi_score": float(r.sqi_score)}


# --------------------------------------------------------------------------
# Построение кэша: полный прогон с записью промежуточных величин
# --------------------------------------------------------------------------
def build_trace(record_id: str, subject_dir: Path) -> RecordTrace:
    video_path = ev.find_video(subject_dir)
    gt_path = ev.find_ground_truth(subject_dir)
    t_ref, wave, hr_device = ev.load_reference_waveform(gt_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg = make_config(fps)

    frames: list[FrameTrace] = []
    timestamps: list[int] = []
    holder: dict = {}

    original_extract = pipeline_mod.extract_rois

    def recording_extract(*args, **kwargs):
        res = original_extract(*args, **kwargs)
        # Копия ДО того, как _check_valid_pixel_count_stability изменит
        # valid_by_roi на месте; landmarks_px не нужен никому, кроме
        # отладочного оверлея, и занимает основной объём — не храним.
        holder["roi"] = replace(copy.deepcopy(res), landmarks_px=np.zeros((0, 2)))
        return res

    pipeline_mod.extract_rois = recording_extract
    rows = []
    try:
        with RPPGPipeline(cfg, debug=True) as pipe:
            original_detect = pipe._landmarker.detect
            original_refresh = pipe._refresh_skin_mask_cache

            def recording_detect(frame, ts):
                face = original_detect(frame, ts)
                holder["detected"] = bool(face.detected)
                return face

            def recording_refresh(frame, ts):
                before = pipe._pending_skin_mask_warning
                out = original_refresh(frame, ts)
                after = pipe._pending_skin_mask_warning
                holder["warn"] = after if (after is not None and after is not before) else None
                return out

            pipe._landmarker.detect = recording_detect
            pipe._refresh_skin_mask_cache = recording_refresh

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = int(idx / fps * 1000)
                holder.clear()
                res = pipe.process_frame(frame, ts)
                frames.append(FrameTrace(
                    detected=holder.get("detected", False),
                    roi=holder.get("roi"),
                    skin_warning=holder.get("warn"),
                ))
                timestamps.append(ts)
                if res is not None:
                    rows.append(res)
                idx += 1
            apply_second_pass(rows, pipe._trusted_bpm_history, cfg.frequency.history_min_windows)
    finally:
        pipeline_mod.extract_rois = original_extract
        cap.release()

    return RecordTrace(
        record_id=record_id, fps=float(fps), timestamps_ms=timestamps, frames=frames,
        t_ref=t_ref, wave=wave, hr_device=hr_device,
        subset=1 if gt_path.name == "gtdump.xmp" else 2,
        live=[_row(r) for r in rows],
    )


# --------------------------------------------------------------------------
# Воспроизведение: сигнальная часть пайплайна без видео и без MediaPipe
# --------------------------------------------------------------------------
class _StubFace:
    __slots__ = ("detected", "landmarks_norm", "head_pose", "face_presence_ok")

    def __init__(self, detected: bool):
        self.detected = detected
        self.landmarks_norm = None
        self.head_pose = None
        self.face_presence_ok = detected


class _StubLandmarker:
    def __init__(self, *args, **kwargs):
        self.frames: list[FrameTrace] = []
        self.idx = -1

    def detect(self, frame, ts):
        self.idx += 1
        return _StubFace(self.frames[self.idx].detected)

    def close(self):
        pass


def replay(trace: RecordTrace, cfg: PipelineConfig | None = None, two_pass: bool = True,
           debug: bool = True, setup=None, on_result=None) -> tuple[list, RPPGPipeline]:
    """Прогон сигнальной части по кэшу. Возвращает (окна, пайплайн).

    cfg — для экспериментов с параметрами ПОСЛЕ извлечения ROI. Параметры,
    влияющие на сами ROI (маска кожи, полигоны, shrink_factor), из кэша
    не меняются — для них кэш надо перестроить.
    """
    cfg = cfg or make_config(trace.fps)
    original_cls = pipeline_mod.FaceLandmarkerWrapper
    original_extract = pipeline_mod.extract_rois
    pipeline_mod.FaceLandmarkerWrapper = _StubLandmarker
    state = {"lm": None}

    def replay_extract(*args, **kwargs):
        ft = trace.frames[state["lm"].idx]
        return copy.deepcopy(ft.roi)

    pipeline_mod.extract_rois = replay_extract
    rows = []
    try:
        pipe = RPPGPipeline(cfg, debug=debug)
        lm = pipe._landmarker
        lm.frames = trace.frames
        state["lm"] = lm

        def replay_refresh(frame, ts):
            warn = trace.frames[lm.idx].skin_warning
            if warn is not None:
                pipe._pending_skin_mask_warning = warn
            return None

        pipe._refresh_skin_mask_cache = replay_refresh
        if setup is not None:
            setup(pipe)  # точка подключения диагностических обёрток
        for ts in trace.timestamps_ms:
            res = pipe.process_frame(DUMMY_FRAME, ts)
            if res is not None:
                rows.append(res)
                if on_result is not None:
                    on_result(pipe, res)
        if two_pass:
            apply_second_pass(rows, pipe._trusted_bpm_history, cfg.frequency.history_min_windows)
    finally:
        pipeline_mod.FaceLandmarkerWrapper = original_cls
        pipeline_mod.extract_rois = original_extract
    return rows, pipe


# --------------------------------------------------------------------------
def _discover():
    import degenerate as dg
    return dg.discover_records(Path("~/Downloads").expanduser())


def cmd_build(args) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = _discover()
    if args.records:
        wanted = set(args.records.split(","))
        records = [(r, d) for r, d in records if r in wanted]
    for record_id, subject_dir in records:
        if cache_path(record_id).exists() and not args.force:
            print(f"  {record_id:16s} уже в кэше", flush=True)
            continue
        t0 = time.time()
        trace = build_trace(record_id, subject_dir)
        with cache_path(record_id).open("wb") as fh:
            pickle.dump(trace, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size = cache_path(record_id).stat().st_size / 1e6
        print(f"  {record_id:16s} кадров {len(trace.frames):5d}, окон {len(trace.live):3d}, "
              f"{size:5.1f} МБ, {time.time() - t0:5.1f}с", flush=True)
    print("BUILD_DONE", flush=True)


def cmd_verify(args) -> None:
    """Сверка воспроизведения с полным прогоном — и с кэшированным итогом,
    и с независимой выгрузкой прогона первой статьи после правки частоты."""
    after_dir = REPO_ROOT / "results" / "fs_fix" / "after" / "ubfc_windows_twopass"
    ok_all = True
    n_rec = n_win = 0
    t_total = 0.0
    for path in sorted(CACHE_DIR.glob("*.pkl")):
        trace = load_trace(path.stem)
        t0 = time.time()
        rows, _ = replay(trace)
        t_total += time.time() - t0
        got = [_row(r) for r in rows]
        same_live = got == trace.live
        same_paper = None
        paper = after_dir / f"{trace.record_id}.csv"
        if paper.exists():
            with paper.open(newline="", encoding="utf-8") as fh:
                pr = list(csv.DictReader(fh))
            same_paper = len(pr) == len(got) and all(
                abs(float(p["second"]) * 1000 - g["timestamp_ms"]) < 0.5
                and ((p["bpm_pred"] in ("", "None") and not np.isfinite(g["bpm"]))
                     or (p["bpm_pred"] not in ("", "None") and float(p["bpm_pred"]) == g["bpm"]))
                for p, g in zip(pr, got)
            )
        ok = same_live and same_paper is not False
        ok_all &= ok
        n_rec += 1
        n_win += len(got)
        if not ok or args.verbose:
            print(f"  {trace.record_id:16s} окон {len(got):3d}  с полным прогоном: "
                  f"{'да' if same_live else 'НЕТ'}  с выгрузкой статьи: "
                  f"{'—' if same_paper is None else ('да' if same_paper else 'НЕТ')}", flush=True)
    print(f"записей {n_rec}, окон {n_win}, воспроизведение {t_total:.1f}с суммарно "
          f"({t_total / max(n_rec, 1):.2f}с на запись)")
    print("ВЕРДИКТ:", "ВОСПРОИЗВЕДЕНИЕ ПОБИТОВО СОВПАДАЕТ" if ok_all else "ЕСТЬ РАСХОЖДЕНИЯ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--records", default="")
    b.add_argument("--force", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    {"build": cmd_build, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
