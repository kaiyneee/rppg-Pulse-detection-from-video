"""
Критерий приёмки 5: повторный прогон даёт побитово те же строки.

Полный прогон идёт около 7 часов, поэтому повторяется ПОДВЫБОРКА пар
(запись, условие), а сравнение ведётся построчно с соответствующими
строками основного windows.csv. Подвыборка подобрана так, чтобы задеть
все нетривиальные пути: G0 (подлинное видео), A2 (генератор шума —
проверяет фиксацию seed), B1f (маска лица и мерцание), D1 (подмена
выбора ROI на фон).

    PYTHONPATH=src venv/bin/python3 benchmark/degenerate_reproducibility.py --repeat-dir /path/to/repeat

Результат пишется в results/degenerate/reproducibility_check.txt, откуда
его читает benchmark/degenerate_acceptance.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path, pairs: set[tuple[str, str]] | None):
    rows, found = [], set()
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_rec, i_cond = header.index("record_id"), header.index("condition")
        for row in reader:
            key = (row[i_rec], row[i_cond])
            if pairs is None or key in pairs:
                rows.append(",".join(row))
                found.add(key)
    return rows, found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "degenerate"))
    ap.add_argument("--repeat-dir", required=True, help="каталог повторного прогона (его windows.csv)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    repeat_dir = Path(args.repeat_dir).expanduser()
    repeat_csv = repeat_dir / "windows.csv"
    if not repeat_csv.exists():
        repeat_csv = repeat_dir / "windows_partial.csv"

    repeat_rows, pairs = read_rows(repeat_csv, None)
    main_rows, _ = read_rows(out_dir / "windows.csv", pairs)

    h_main = hashlib.sha256("\n".join(sorted(main_rows)).encode()).hexdigest()
    h_repeat = hashlib.sha256("\n".join(sorted(repeat_rows)).encode()).hexdigest()
    same = h_main == h_repeat and len(main_rows) == len(repeat_rows)

    verdict = "СОВПАДАЕТ" if same else "РАЗЛИЧАЕТСЯ"
    lines = [
        f"{verdict} побитово на подвыборке: {len(pairs)} пар (запись, условие), {len(repeat_rows)} строк",
        f"пары: {', '.join(f'{r}/{c}' for r, c in sorted(pairs))}",
        f"sha256 основного прогона : {h_main}",
        f"sha256 повторного прогона: {h_repeat}",
        f"источник повторного прогона: {repeat_csv}",
    ]
    if not same:
        for a, b in zip(sorted(main_rows), sorted(repeat_rows)):
            if a != b:
                lines += [f"первое расхождение:", f"  основной : {a}", f"  повторный: {b}"]
                break
    text = "\n".join(lines)
    (out_dir / "reproducibility_check.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
