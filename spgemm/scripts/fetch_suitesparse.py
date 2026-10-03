#!/usr/bin/env python3
"""Download the SuiteSparse matrices listed in scripts/suitesparse_matrices.txt into matrices/.

Usage:
    scripts/fetch_suitesparse.py              # every matrix in the list
    scripts/fetch_suitesparse.py cant SiH4    # only these (by name)

Each matrix is fetched as <group>/<name>.tar.gz from https://sparse.tamu.edu/MM and only <name>.mtx is
kept, as matrices/<name>.mtx. Matrices already present are skipped. matrices/ is git-ignored.
"""
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "scripts" / "suitesparse_matrices.txt"
OUT = ROOT / "matrices"
URL = "https://sparse.tamu.edu/MM/{}.tar.gz"


def listed_matrices():
    for line in LIST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


def fetch(entry):
    name = entry.split("/")[-1]
    target = OUT / f"{name}.mtx"
    if target.exists():
        print(f"  {name}: already present")
        return
    print(f"  {name}: downloading {URL.format(entry)}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.tar.gz"
        urllib.request.urlretrieve(URL.format(entry), archive)
        with tarfile.open(archive) as tar:
            member = tar.getmember(f"{name}/{name}.mtx")
            with tar.extractfile(member) as src, open(target.with_suffix(".part"), "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    target.with_suffix(".part").rename(target)  # only complete files get the .mtx name
    print(f"  {name}: {target.stat().st_size / 1e6:.1f} MB")


def main():
    OUT.mkdir(exist_ok=True)
    wanted = set(sys.argv[1:])
    entries = [e for e in listed_matrices() if not wanted or e.split("/")[-1] in wanted]
    unknown = wanted - {e.split("/")[-1] for e in entries}
    if unknown:
        print(f"Not in {LIST.name}: {', '.join(sorted(unknown))}")
        return 1
    for entry in entries:
        fetch(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
