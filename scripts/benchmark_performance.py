"""
Замеры производительности (п.32 требований) — на КОНКРЕТНОМ железе, где
запущен скрипт (см. поле "hardware" в выводе), а не абстрактные цифры "из
статьи". Меряет:

  1. Разбивку по стадиям: MediaPipe (детекция+landmarks), ROI-экстракция,
     каждый из 6 методов извлечения сигнала, каждый из 3 методов оценки
     частоты, SQI (quality.assess_quality, включая новые компоненты п.19-22).
  2. Numba: "холодная" JIT-компиляция при первом вызове (@njit(cache=True) в
     accel/fast_ops.py) в СВЕЖЕМ процессе против прогретого вызова в том же
     процессе И против попадания в ДИСКОВЫЙ кэш (cache=True) в новом
     процессе — три разных числа, которые легко перепутать. Плюс сравнение
     установившейся (steady-state) скорости POS с Numba и без (чистый numpy
     fallback, methods.PosMethod(use_numba=False)).
  3. CPU vs GPU delegate MediaPipe (честно — если GPU-делегат недоступен на
     этой платформе, так и написано, а не фабрикуются цифры).
  4. Задержка первой оценки: полный RPPGPipeline.process_frame на потоке
     кадров, где лицо "детектируется" каждый кадр (см. ОГОВОРКУ ниже) —
     явно видно, приходится ли JIT-компиляция POS на первый BPM-шаг и
     насколько он дольше остальных.
  5. Сквозной FPS: "пол" (лицо никогда не найдено — чистая стоимость
     инференса MediaPipe на кадр) и оценка "при найденном лице" (сумма
     стадий 1-2 на каждый кадр + стадии 3-4 раз в WindowConfig.step_seconds).

ОГОВОРКА (честно, а не по умолчанию): для замера стадий 2-5 при "найденном
лице" в этой среде нет доступного тестового фото реального лица (скачивать
биометрику лица по неофициальным каналам без явного согласия исследователь
не должен — см. обсуждение доступа к rPPG-датасетам в п.25/26/31), поэтому
FaceLandmarkerWrapper.detect подменяется (monkeypatch) на фиксированный
СИНТЕТИЧЕСКИЙ результат детекции (see make_synthetic_landmarks) — реальная
геометрия точек лица не нужна для измерения ВРЕМЕНИ выполнения
downstream-стадий (ROI-маски/метод/частота/SQI), только сам факт, что
лицо "найдено" и downstream-код реально исполняется. Стадия 1 (само
MediaPipe-детектирование) измеряется ЧЕСТНО и НЕЗАВИСИМО, на реальных
кадрах, без подмены — инференс модели стоит одинаково независимо от того,
найдено лицо или нет (совпадает по постановке с "полом" FPS, см. пункт 5).

Запуск: PYTHONPATH=src python3 scripts/benchmark_performance.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_PATH = str(REPO_ROOT / "models" / "face_landmarker.task")
NUMBA_CACHE_DIR = REPO_ROOT / "src" / "rppg" / "accel" / "__pycache__"


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #

def timeit_ms(fn, n_iters: int = 50, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    times = np.empty(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "std_ms": float(times.std()),
        "p95_ms": float(np.percentile(times, 95)),
        "n": n_iters,
    }


def make_synthetic_landmarks(n_points: int = 478, seed: int = 0) -> np.ndarray:
    """478 точек в грубом эллипсе (нормализованные [0,1] координаты) —
    ТОЛЬКО для того, чтобы ROI-полигоны/hull'ы строились на невырожденных
    наборах точек внутри кадра. Не имитирует реальную геометрию лица — см.
    ОГОВОРКУ в докстринге модуля о том, почему для замера ВРЕМЕНИ это не
    нужно."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    radius = np.sqrt(rng.uniform(0, 1, n_points))
    x = 0.5 + 0.18 * radius * np.cos(theta)
    y = 0.5 + 0.25 * radius * np.sin(theta)
    z = rng.normal(0, 0.01, n_points)
    return np.stack([x, y, z], axis=1)


def make_skin_colored_frame(h: int, w: int) -> np.ndarray:
    """BGR=(40,60,100) подобран перебором так, чтобы после cv2.BGR2YCrCb
    попадать в диапазон face.roi.build_skin_mask (133<=Cr<=173, 77<=Cb<=127)
    — иначе roi.roi_mean_rgb отбраковывал бы ВСЕ пиксели как "не кожа" и
    downstream-стадии (extract/preprocess) не исполнялись бы по-настоящему,
    искажая замер в сторону "слишком быстро"."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (40, 60, 100)
    return frame


# --------------------------------------------------------------------------- #
# 1. MediaPipe: CPU vs GPU delegate
# --------------------------------------------------------------------------- #

def bench_mediapipe(delegate: str, frame: np.ndarray, n_iters: int = 40, warmup: int = 5):
    from rppg.face.landmarker import FaceLandmarkerWrapper

    try:
        wrapper = FaceLandmarkerWrapper(model_asset_path=MODEL_PATH, delegate=delegate)
    except Exception as exc:  # noqa: BLE001 - честно репортим недоступность делегата
        return None, f"{type(exc).__name__}: {exc}"

    ts = [0]

    def step():
        ts[0] += 33
        wrapper.detect(frame, ts[0])

    try:
        result = timeit_ms(step, n_iters=n_iters, warmup=warmup)
    except Exception as exc:  # noqa: BLE001
        wrapper.close()
        return None, f"{type(exc).__name__}: {exc}"

    wrapper.close()
    return result, None


_MEDIAPIPE_DELEGATE_PROBE_SCRIPT = """
import sys, json
sys.path.insert(0, "src")
import numpy as np
from scripts.benchmark_performance import bench_mediapipe, MODEL_PATH
frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
result, err = bench_mediapipe("{delegate}", frame)
print(json.dumps({{"result": result, "error": err}}))
"""


def bench_mediapipe_isolated(delegate: str) -> tuple[dict | None, str | None]:
    """MediaPipe с GPU-делегатом на некоторых платформах (подтверждено на
    macOS/Metal в этой среде) падает с нативным `CHECK failure` внутри
    C++-рантайма mediapipe — это process-level abort (SIGABRT), а не
    Python-исключение, поймать его в текущем процессе НЕВОЗМОЖНО (try/except
    здесь бесполезен). Поэтому делегат тестируется в ИЗОЛИРОВАННОМ
    подпроцессе: если он падает, ловим только ненулевой код возврата
    родительского процесса, остальной бенчмарк продолжает работать."""
    script = _MEDIAPIPE_DELEGATE_PROBE_SCRIPT.format(delegate=delegate)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        reason = tail[-1] if tail else f"процесс завершился с кодом {proc.returncode}"
        return None, f"подпроцесс упал (returncode={proc.returncode}): {reason}"

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return None, f"не удалось разобрать вывод подпроцесса: {exc}"

    return payload["result"], payload["error"]


# --------------------------------------------------------------------------- #
# 2. ROI-экстракция
# --------------------------------------------------------------------------- #

def bench_roi_extraction(frame: np.ndarray, landmarks_norm: np.ndarray, n_iters: int = 200):
    from rppg.face.roi import extract_rois

    def step():
        extract_rois(frame, landmarks_norm, shrink_factor=0.9, min_valid_fraction=0.6)

    return timeit_ms(step, n_iters=n_iters, warmup=5)


# --------------------------------------------------------------------------- #
# 3. Методы извлечения сигнала (6 методов) + POS с Numba vs без
# --------------------------------------------------------------------------- #

def bench_extraction_methods(fps: float = 30.0, window_seconds: float = 10.0, n_iters: int = 100):
    from rppg.signal.methods import get_method, SignalWindow

    n = int(fps * window_seconds)
    rng = np.random.default_rng(0)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.stack(
        [110.0 + 0.6 * pulse + rng.normal(0, 0.3, n) for _ in range(3)], axis=1
    )
    window = SignalWindow(rgb_traces={"forehead": rgb}, fps=fps, hr_band_hz=(0.7, 4.0))

    results = {}
    for name in ["green", "chrom", "pos", "pca", "ica"]:
        method = get_method(name)

        def step(m=method):
            m.extract(window, "forehead")

        results[name] = timeit_ms(step, n_iters=n_iters, warmup=5)

    pos_no_numba = get_method("pos", use_numba=False)

    def step_pos_no_numba():
        pos_no_numba.extract(window, "forehead")

    results["pos_no_numba"] = timeit_ms(step_pos_no_numba, n_iters=n_iters, warmup=5)

    traj = rng.normal(100.0, 1.0, (n, 40, 2))
    window_motion = SignalWindow(rgb_traces={}, fps=fps, landmark_trajectories=traj, hr_band_hz=(0.7, 4.0))
    head_motion = get_method("head_motion")

    def step_head_motion():
        head_motion.extract(window_motion, None)

    results["head_motion"] = timeit_ms(step_head_motion, n_iters=n_iters, warmup=5)
    return results


# --------------------------------------------------------------------------- #
# 4. Методы оценки частоты (FFT / Welch / Lomb-Scargle)
# --------------------------------------------------------------------------- #

def bench_frequency_methods(fps: float = 30.0, window_seconds: float = 10.0, n_iters: int = 200):
    from rppg.signal.frequency import estimate_hr

    n = int(fps * window_seconds)
    t = np.arange(n) / fps
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.default_rng(0).normal(0, 1, n)
    band = (0.7, 4.0)

    results = {}
    for method in ["fft", "welch"]:
        def step(m=method):
            estimate_hr(signal, fps, band, method=m)

        results[method] = timeit_ms(step, n_iters=n_iters, warmup=5)

    def step_ls():
        estimate_hr(signal, fps, band, method="lomb_scargle", timestamps_sec=t)

    results["lomb_scargle"] = timeit_ms(step_ls, n_iters=n_iters, warmup=5)
    return results


# --------------------------------------------------------------------------- #
# 5. SQI (assess_quality) — включая новые компоненты п.19-22
# --------------------------------------------------------------------------- #

def bench_sqi(fps: float = 30.0, window_seconds: float = 10.0, n_iters: int = 100):
    from rppg.signal.quality import assess_quality, dominant_frequency_and_snr, SQIInputs
    from rppg.config import QualityConfig

    n = int(fps * window_seconds)
    t = np.arange(n) / fps
    rng = np.random.default_rng(0)
    band = (0.7, 4.0)

    processed = np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.normal(0, 1, n)
    background = 0.1 * rng.normal(0, 1, n)
    landmark_traj = rng.normal(100.0, 1.0, (n, 40, 2))
    ipd = np.full(n, 80.0)
    peak_freq, snr_db = dominant_frequency_and_snr(processed, fps, band)
    qcfg = QualityConfig()

    def step():
        assess_quality(
            SQIInputs(
                spectral_snr_db=snr_db,
                peak_freq_hz=peak_freq,
                fps=fps,
                band_hz=band,
                bpm_by_roi={"forehead": 72.0, "left_cheek": 73.0, "right_cheek": 71.0},
                landmark_trajectories=landmark_traj,
                interocular_distances_px=ipd,
                recent_bpm_history=[72.0, 73.0, 71.0, 72.0, 73.0, 72.0],
                harmonic_check_signal=processed,
                background_signal=background,
            ),
            qcfg,
        )

    return timeit_ms(step, n_iters=n_iters, warmup=5)


# --------------------------------------------------------------------------- #
# 6. Numba: cold vs disk-cache-hit vs in-memory-warm, в ОТДЕЛЬНЫХ процессах
# --------------------------------------------------------------------------- #

_NUMBA_PROBE_SCRIPT = """
import sys, time
sys.path.insert(0, "src")
import numpy as np
from rppg.accel.fast_ops import pos_overlap_add_numba
rgb = np.random.default_rng(0).normal(100, 5, (300, 3))
proj = np.array([[0., 1., -1.], [-2., 1., 1.]])
t0 = time.perf_counter(); pos_overlap_add_numba(rgb, 48, proj); t1 = time.perf_counter()
t2 = time.perf_counter(); pos_overlap_add_numba(rgb, 48, proj); t3 = time.perf_counter()
print(f"{(t1 - t0) * 1000:.4f} {(t3 - t2) * 1000:.4f}")
"""


def bench_numba_cold_warm() -> dict:
    for f in NUMBA_CACHE_DIR.glob("fast_ops.*.nb*") if NUMBA_CACHE_DIR.exists() else []:
        f.unlink()

    cold = subprocess.run(
        [sys.executable, "-c", _NUMBA_PROBE_SCRIPT], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    cold_first_ms, cold_second_ms = map(float, cold.stdout.strip().split())

    warm = subprocess.run(
        [sys.executable, "-c", _NUMBA_PROBE_SCRIPT], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    cache_hit_first_ms, cache_hit_second_ms = map(float, warm.stdout.strip().split())

    return {
        # Полная LLVM-компиляция: ни in-memory, ни disk-кэша нет.
        "cold_compile_first_call_ms": cold_first_ms,
        "cold_compile_process_second_call_ms": cold_second_ms,
        # Новый процесс, НО disk-кэш (@njit(cache=True)) уже существует.
        "disk_cache_hit_first_call_ms": cache_hit_first_ms,
        "disk_cache_hit_process_second_call_ms": cache_hit_second_ms,
    }


# --------------------------------------------------------------------------- #
# 7. Сквозной пайплайн: "пол" FPS (лицо не найдено) и латентность первой оценки
# --------------------------------------------------------------------------- #

def bench_pipeline_floor_fps(n_frames: int = 150, resolution: tuple[int, int] = (480, 640)) -> dict:
    from rppg.pipeline import RPPGPipeline
    from rppg.config import PipelineConfig

    frame = np.zeros((*resolution, 3), dtype=np.uint8)
    pipe = RPPGPipeline(PipelineConfig())
    for i in range(5):
        pipe.process_frame(frame, i * 33)

    t0 = time.perf_counter()
    for i in range(n_frames):
        pipe.process_frame(frame, (i + 5) * 33)
    total_ms = (time.perf_counter() - t0) * 1000.0
    pipe.close()

    return {
        "n_frames": n_frames,
        "total_ms": total_ms,
        "ms_per_frame": total_ms / n_frames,
        "fps": n_frames / (total_ms / 1000.0),
    }


def bench_pipeline_with_synthetic_face(
    n_frames: int = 300,
    resolution: tuple[int, int] = (480, 640),
    clear_numba_cache: bool = False,
) -> dict:
    """См. ОГОВОРКУ в докстринге модуля: FaceLandmarkerWrapper.detect
    подменяется на фиксированный синтетический результат, чтобы измерить
    ВСЕ downstream-стадии (ROI/метод/частота/SQI) реально исполняющимися,
    а не падающими в ранний "лицо не найдено" выход."""
    from rppg.pipeline import RPPGPipeline
    from rppg.config import PipelineConfig
    from rppg.face.landmarker import FaceFrameResult, HeadPose

    if clear_numba_cache and NUMBA_CACHE_DIR.exists():
        for f in NUMBA_CACHE_DIR.glob("fast_ops.*.nb*"):
            f.unlink()

    h, w = resolution
    frame = make_skin_colored_frame(h, w)
    landmarks = make_synthetic_landmarks()
    synthetic_result = FaceFrameResult(
        detected=True, landmarks_norm=landmarks, head_pose=HeadPose(0.0, 0.0, 0.0), face_presence_ok=True
    )

    pipe = RPPGPipeline(PipelineConfig())
    pipe._landmarker.detect = lambda frame_bgr, ts: synthetic_result  # monkeypatch только для бенчмарка

    per_frame_ms = np.empty(n_frames)
    first_estimate_frame_idx = None
    first_estimate_wall_ms = None
    t_start = time.perf_counter()
    for i in range(n_frames):
        ts = int(i / 30.0 * 1000)
        t0 = time.perf_counter()
        result = pipe.process_frame(frame, ts)
        per_frame_ms[i] = (time.perf_counter() - t0) * 1000.0
        if result is not None and first_estimate_frame_idx is None:
            first_estimate_frame_idx = i
            first_estimate_wall_ms = (time.perf_counter() - t_start) * 1000.0
    pipe.close()

    return {
        "n_frames": n_frames,
        "mean_ms_per_frame": float(per_frame_ms.mean()),
        "median_ms_per_frame": float(np.median(per_frame_ms)),
        "p95_ms_per_frame": float(np.percentile(per_frame_ms, 95)),
        "max_ms_per_frame": float(per_frame_ms.max()),
        "max_ms_frame_idx": int(np.argmax(per_frame_ms)),
        "fps_mean_excl_first_estimate_window": 1000.0 / float(np.median(per_frame_ms)),
        "first_estimate_frame_idx": first_estimate_frame_idx,
        "first_estimate_wall_ms": first_estimate_wall_ms,
    }


_PIPELINE_WITH_FACE_PROBE_SCRIPT = """
import sys, json
sys.path.insert(0, "src")
from scripts.benchmark_performance import bench_pipeline_with_synthetic_face
result = bench_pipeline_with_synthetic_face(n_frames={n_frames}, clear_numba_cache={clear_numba_cache})
print(json.dumps(result))
"""


def bench_pipeline_with_synthetic_face_isolated(clear_numba_cache: bool, n_frames: int = 300) -> dict:
    """ВАЖНО: bench_pipeline_with_synthetic_face() ДОЛЖЕН запускаться в
    СВЕЖЕМ процессе, где POS ещё ни разу не вызывался — иначе, если в этом
    же Python-процессе POS/numba уже были прогреты РАНЕЕ (например, стадией
    3 этого же скрипта, bench_extraction_methods), очистка ДИСКОВОГО кэша
    numba перед вызовом ничего не даст: @njit держит скомпилированный
    диспетчер В ПАМЯТИ процесса, и удаление файла кэша с диска эту
    in-memory копию не инвалидирует. Это реальная ошибка, которую я сначала
    допустил в этом скрипте (первый прогон с 'очищенным кэшем' в общем
    процессе НЕ показал ожидаемого скачка на первом BPM-окне именно по этой
    причине) — поэтому тест на холодный старт вынесен в изолированный
    подпроцесс, где POS гарантированно не тронут ничем до самого теста."""
    script = _PIPELINE_WITH_FACE_PROBE_SCRIPT.format(n_frames=n_frames, clear_numba_cache=clear_numba_cache)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"подпроцесс упал (returncode={proc.returncode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #

def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
    }
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        from numba import __version__ as numba_version
        info["numba_version"] = numba_version
    except ImportError:
        info["numba_version"] = None
    return info


def main() -> None:
    report: dict = {"hardware": hardware_info()}
    print(f"Железо: {report['hardware']}\n")

    print("--- 1. MediaPipe (CPU delegate) ---")
    frame_640x480 = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    mp_cpu, mp_cpu_err = bench_mediapipe("CPU", frame_640x480)
    report["mediapipe_cpu"] = mp_cpu or {"error": mp_cpu_err}
    print(f"  CPU: {mp_cpu}" if mp_cpu else f"  CPU: НЕДОСТУПНО ({mp_cpu_err})")

    print("--- 1b. MediaPipe (GPU delegate, в изолированном подпроцессе — см. docstring) ---")
    mp_gpu, mp_gpu_err = bench_mediapipe_isolated("GPU")
    report["mediapipe_gpu"] = mp_gpu or {"error": mp_gpu_err}
    print(f"  GPU: {mp_gpu}" if mp_gpu else f"  GPU: НЕДОСТУПНО на этой платформе ({mp_gpu_err})")

    print("\n--- 2. ROI-экстракция ---")
    landmarks = make_synthetic_landmarks()
    skin_frame = make_skin_colored_frame(480, 640)
    roi_result = bench_roi_extraction(skin_frame, landmarks)
    report["roi_extraction"] = roi_result
    print(f"  {roi_result}")

    print("\n--- 3. Методы извлечения сигнала (окно 10с @ 30fps = 300 отсчётов) ---")
    methods_result = bench_extraction_methods()
    report["extraction_methods"] = methods_result
    for name, r in methods_result.items():
        print(f"  {name:15s} mean={r['mean_ms']:.4f} ms  median={r['median_ms']:.4f} ms  p95={r['p95_ms']:.4f} ms")
    speedup = methods_result["pos_no_numba"]["median_ms"] / methods_result["pos"]["median_ms"]
    print(f"  -> POS: Numba даёт {speedup:.1f}x ускорение относительно чистого numpy (по median)")
    report["pos_numba_speedup_x"] = speedup

    print("\n--- 4. Методы оценки частоты ---")
    freq_result = bench_frequency_methods()
    report["frequency_methods"] = freq_result
    for name, r in freq_result.items():
        print(f"  {name:15s} mean={r['mean_ms']:.4f} ms  median={r['median_ms']:.4f} ms")

    print("\n--- 5. SQI (assess_quality, включая harmonic/flicker из п.19-22) ---")
    sqi_result = bench_sqi()
    report["sqi"] = sqi_result
    print(f"  {sqi_result}")

    print("\n--- 6. Numba: cold-compile vs disk-cache-hit vs in-memory-warm (отдельные процессы) ---")
    numba_result = bench_numba_cold_warm()
    report["numba_cold_warm"] = numba_result
    for k, v in numba_result.items():
        print(f"  {k}: {v:.3f} ms")

    print("\n--- 7a. Сквозной пайплайн, 'пол' FPS (лицо никогда не найдено, 640x480) ---")
    floor_result = bench_pipeline_floor_fps()
    report["pipeline_floor_fps"] = floor_result
    print(f"  {floor_result}")

    print("\n--- 7b. Сквозной пайплайн с синтетическим 'лицом' (см. оговорку в докстринге) ---")
    print("      Оба варианта запускаются в ИЗОЛИРОВАННЫХ свежих подпроцессах (см. docstring")
    print("      bench_pipeline_with_synthetic_face_isolated) — иначе более ранние стадии этого же")
    print("      скрипта (стадия 3) уже прогревают POS/numba В ПАМЯТИ, и очистка ТОЛЬКО дискового")
    print("      кэша перестаёт что-либо демонстрировать.")
    print("      (a) numba-кэш ОЧИЩЕН перед стартом -> первый BPM-шаг должен поймать холодную JIT-компиляцию")
    with_face_cold = bench_pipeline_with_synthetic_face_isolated(clear_numba_cache=True)
    report["pipeline_with_face_cold_numba"] = with_face_cold
    print(f"  {with_face_cold}")

    print("\n      (b) СВЕЖИЙ процесс, но disk-кэш numba уже тёплый (остался от (a))")
    with_face_warm = bench_pipeline_with_synthetic_face_isolated(clear_numba_cache=False)
    report["pipeline_with_face_warm_numba"] = with_face_warm
    print(f"  {with_face_warm}")

    # Оценка сквозного FPS "при найденном лице": каждый КАДР стоит
    # MediaPipe + ROI-экстракция; стадии метод/частота/SQI выполняются раз
    # в WindowConfig.step_seconds (по умолчанию 1с), т.е. раз в ~30 кадров
    # при 30fps, и по умолчанию считаются для 3 ROI (forehead/left_cheek/right_cheek).
    if mp_cpu is not None:
        step_frames = 30  # window.step_seconds=1.0 * assumed_fps=30.0 по умолчанию
        per_step_ms = 3 * (methods_result["pos"]["median_ms"] + freq_result["welch"]["median_ms"]) + sqi_result["median_ms"]
        est_ms_per_frame = mp_cpu["median_ms"] + roi_result["median_ms"] + per_step_ms / step_frames
        report["estimated_steady_state_fps_with_face"] = {
            "formula": "mediapipe_median_ms + roi_median_ms + (3*(pos_median_ms+welch_median_ms)+sqi_median_ms)/step_frames",
            "est_ms_per_frame": est_ms_per_frame,
            "est_fps": 1000.0 / est_ms_per_frame,
        }
        print(f"\n--- Оценка установившегося FPS при найденном лице (формула из стадий 1-5) ---")
        print(f"  ~{est_ms_per_frame:.2f} ms/кадр -> ~{1000.0/est_ms_per_frame:.1f} FPS (устойчивое состояние, без JIT-компиляции)")

    out_path = REPO_ROOT / "benchmark" / "performance_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nПолный отчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
