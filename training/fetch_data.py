"""Download the NLBSE'24 CSVs at the pinned upstream commit and verify their SHA-256.

Usage: .venv/bin/python training/fetch_data.py [--dest data/raw]
Never run in CI. Each file is downloaded to a temporary name in the destination directory,
verified, then renamed into place atomically. A file that is already present and verified is
skipped. A file that is present but fails verification is left untouched and reported, so local
data is never silently replaced.
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from nlbse_data import RAW_URL, SHA256, sha256_file

TIMEOUT_S = 60


def fetch(dest, files=None, opener=urllib.request.urlopen, timeout=TIMEOUT_S) -> dict:
    """Ensure each `files` entry ({name: sha256}) exists in `dest` with the right hash.

    Returns {name: "present" | "downloaded"}. Raises ValueError on a hash mismatch.
    """
    files = SHA256 if files is None else files
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, expected in files.items():
        target = dest / name
        if target.exists():
            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(f"{target} exists but sha256 is {actual}, expected {expected}; "
                                 "move it away and rerun")
            status[name] = "present"
            continue
        fd, tmp = tempfile.mkstemp(dir=dest, prefix=f".{name}.", suffix=".part")
        try:
            h = hashlib.sha256()
            with os.fdopen(fd, "wb") as out, opener(RAW_URL.format(name=name), timeout=timeout) as resp:
                for chunk in iter(lambda: resp.read(1 << 16), b""):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f"{name}: downloaded sha256 {h.hexdigest()}, expected {expected}")
            os.chmod(tmp, 0o644)  # mkstemp creates 0600
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        status[name] = "downloaded"
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args(argv)
    for name, state in fetch(args.dest).items():
        print(f"{state:10} {args.dest / name}  sha256 {SHA256[name]}")


if __name__ == "__main__":
    sys.exit(main())
