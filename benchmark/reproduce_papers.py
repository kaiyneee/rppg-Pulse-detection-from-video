"""
Воспроизведение чисел статей одной командой — на пресете legacy_welch8
(конфигурация, на которой получены обе статьи; git-тег papers-v1).

    # статья 1: UBFC-rPPG, однопроходный и двухпроходный режимы (~1 ч)
    PYTHONPATH=src venv/bin/python3 benchmark/reproduce_papers.py --paper 1

    # статья 2: вырожденные входы, 11 условий x 50 записей (~7 ч)
    PYTHONPATH=src venv/bin/python3 benchmark/reproduce_papers.py --paper 2

Результаты пишутся в results/reproduce/paper<N>/ вместе с run_meta.json,
где записано имя пресета. --records ограничивает набор записей (для
быстрой проверки), --conditions — набор условий статьи 2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_BIN = str(REPO_ROOT / "venv" / "bin" / "python3")
PRESET = "legacy_welch8"


def _env():
    return dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))


def _git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return out.stdout.strip() or None


def _git_dirty() -> bool:
    """Есть ли незакоммиченные изменения отслеживаемых файлов: тогда хеш
    коммита не описывает код, которым получен результат, полностью."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    return bool(out.stdout.strip())


def paper1(out: Path, data_dir: Path, records: list[str] | None) -> None:
    sys.path.insert(0, str(REPO_ROOT / "benchmark"))
    import degenerate as dg  # поиск записей — тот же, что во всех скриптах

    found = dg.discover_records(data_dir)
    if records:
        found = [(r, d) for r, d in found if r in set(records)]
    for record_id, subject_dir in found:
        for extra in ([], ["--two-pass"]):
            cmd = [PY_BIN, str(REPO_ROOT / "scripts" / "eval_ubfc_subject.py"), "--subject-dir", str(subject_dir),
                   "--preset", PRESET, "--out-dir", str(out), *extra]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=REPO_ROOT)
            if proc.returncode != 0:
                raise SystemExit(f"{record_id}: eval_ubfc_subject упал:\n{proc.stderr[-800:]}")
        print(f"  {record_id}", flush=True)
    for script, args, name in (
        ("report_ubfc_split.py", ["--results", str(out / "ubfc_results_twopass.csv")], "report.txt"),
        ("analyze_ubfc_windows.py", ["--dir", str(out / "ubfc_windows_twopass")], "analysis.txt"),
    ):
        proc = subprocess.run([PY_BIN, str(REPO_ROOT / "scripts" / script), *args],
                              capture_output=True, text=True, env=_env(), cwd=REPO_ROOT)
        (out / name).write_text(proc.stdout, encoding="utf-8")
    meta = {"paper": 1, "preset": PRESET, "git_commit": _git_commit(), "git_dirty": _git_dirty(),
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": [r for r, _ in found], "outputs": ["ubfc_results*.csv", "ubfc_windows*/", "report.txt", "analysis.txt"]}
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def paper2(out: Path, data_dir: Path, records: list[str] | None, conditions: str | None) -> None:
    cmd = [PY_BIN, "-u", str(REPO_ROOT / "benchmark" / "degenerate.py"), "--preset", PRESET,
           "--out-dir", str(out), "--data-dir", str(data_dir)]
    if records:
        cmd += ["--records", ",".join(records)]
    if conditions:
        cmd += ["--conditions", conditions]
    proc = subprocess.run(cmd, env=_env(), cwd=REPO_ROOT)  # run_meta.json с пресетом пишет degenerate.py
    if proc.returncode != 0:
        raise SystemExit("degenerate.py завершился с ошибкой")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", type=int, choices=[1, 2], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default="~/Downloads")
    ap.add_argument("--records", default="")
    ap.add_argument("--conditions", default=None)
    args = ap.parse_args()
    out = Path(args.out or REPO_ROOT / "results" / "reproduce" / f"paper{args.paper}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    records = [r.strip() for r in args.records.split(",") if r.strip()] or None
    data_dir = Path(args.data_dir).expanduser()
    if args.paper == 1:
        paper1(out, data_dir, records)
    else:
        paper2(out, data_dir, records, args.conditions)
    print(f"готово: {out}")


if __name__ == "__main__":
    main()
