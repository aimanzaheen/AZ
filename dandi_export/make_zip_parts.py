"""Split a folder into self-contained zip parts, each under a size limit, so it
can be sent through channels that cap file size (e.g. 30 MB per file).

Every part is an ordinary zip: unzip all parts into the same place and you get
the original folder back. Files too large for one part are split by rows
(CSV, CSV.GZ and Parquet become several CSV files named <stem>_part01of03.csv);
other oversized files are split into raw byte chunks (<name>.part01of03) with a
note in README_PARTS.txt on how to join them.

Usage:
    python dandi_export/make_zip_parts.py "Dandi Data sets" OUT_DIR [--limit-mb 29]
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd


def compressed_size(data: bytes) -> int:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("x", data)
    return buf.tell()


def read_table(p: Path) -> pd.DataFrame | None:
    name = p.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(p, low_memory=False, keep_default_na=False)
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    return None


def split_table(p: Path, rel: Path, limit: int) -> list[tuple[str, bytes]]:
    df = read_table(p)
    stem = p.name
    for suf in (".csv.gz", ".csv", ".parquet"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    full = compressed_size(df.to_csv(index=False).encode())
    n = max(2, math.ceil(full / (limit * 0.8)))
    while True:
        rows = math.ceil(len(df) / n)
        chunks = [df.iloc[i * rows:(i + 1) * rows].to_csv(index=False).encode() for i in range(n)]
        if all(compressed_size(c) < limit for c in chunks):
            break
        n += 1
    return [(str(rel.parent / f"{stem}_part{i + 1:02d}of{n:02d}.csv"), c) for i, c in enumerate(chunks)]


def split_bytes(p: Path, rel: Path, limit: int) -> list[tuple[str, bytes]]:
    data = p.read_bytes()
    size = int(limit * 0.95)
    n = math.ceil(len(data) / size)
    return [(f"{rel}.part{i + 1:02d}of{n:02d}", data[i * size:(i + 1) * size]) for i in range(n)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--limit-mb", type=float, default=29)
    ap.add_argument("--prefix", default=None)
    a = ap.parse_args(argv)
    limit = int(a.limit_mb * 1024 * 1024) - 64 * 1024  # leave room for zip headers
    root = a.folder.parent
    prefix = a.prefix or a.folder.name
    a.out_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, bytes, int]] = []  # (arcname, data, compressed size)
    split_notes = []
    for p in sorted(a.folder.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        data = p.read_bytes()
        c = compressed_size(data)
        if c < limit:
            items.append((str(rel), data, c))
            continue
        parts = split_table(p, rel, limit) if read_table_ok(p) else split_bytes(p, rel, limit)
        split_notes.append(f"{rel} -> {len(parts)} pieces ({'CSV row chunks' if read_table_ok(p) else 'byte chunks'})")
        items += [(n, d, compressed_size(d)) for n, d in parts]

    # first-fit decreasing bin packing by compressed size
    bins: list[list] = []
    for it in sorted(items, key=lambda x: -x[2]):
        for b in bins:
            if sum(x[2] for x in b) + it[2] < limit:
                b.append(it)
                break
        else:
            bins.append([it])
    bins.sort(key=lambda b: min(x[0] for x in b))
    total = len(bins)
    readme = [f"{prefix}: {total} zip parts. Unzip ALL parts into the same location (e.g. your Desktop);",
              f"together they recreate the folder '{a.folder.name}'.", ""]
    if split_notes:
        readme += ["Files too large for one part were split:", *[f"  {s}" for s in split_notes], "",
                   "CSV row chunks: each piece is a normal CSV with the header row; open any piece, or",
                   "combine them in Python with pandas.concat([pd.read_csv(f) for f in sorted(glob('..._part*.csv'))]).",
                   "Byte chunks (.partNNofMM): join in order. macOS/Linux: cat name.part* > name ;",
                   "Windows: copy /b name.part01ofMM+name.part02ofMM+... name", ""]
    for i, b in enumerate(bins, 1):
        zp = a.out_dir / f"{prefix} - part {i:02d} of {total:02d}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            if i == 1:
                z.writestr(f"{a.folder.name}/README_PARTS.txt", "\n".join(readme))
            for name, data, _ in sorted(b):
                z.writestr(name, data)
        print(f"{zp.name}: {zp.stat().st_size / 1e6:.1f} MB, {len(b)} files")
    for s in split_notes:
        print("split:", s)
    return 0


def read_table_ok(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".parquet"))


if __name__ == "__main__":
    sys.exit(main())
