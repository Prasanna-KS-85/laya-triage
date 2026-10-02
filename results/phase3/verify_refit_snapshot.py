"""Verify the refit revision (R1b) against R1 and the local refit config (Phase 3c-B Stop A3).

Run from anywhere: .venv/bin/python results/phase3/verify_refit_snapshot.py [--revision SHA]
Offline: reads only the local HF cache (both snapshots, blobs, cached remote file listings) and
results/phase3/refit/rl_agent_config.json. Prints a PASS/FAIL table and exits 1 on any FAIL. No issue data.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from verify_snapshot import git_blob_sha1, repo_dir, sha256_file

HERE = Path(__file__).resolve().parent
REPO = "Prasanna85/laya-issue-triage"
R1 = "a704b3eadd185f1fa028576cfa50605b6eada4a4"
R1_WEIGHTS_SHA256 = "5cd2cc8f3e5d56130fcef9d4cd98aea29734716f4c7e1165b9e353ecae0fd4e9"
UNCHANGED = ("run_meta.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
             ".gitattributes")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--revision", default="76ece1fb0eb8b32bd5d8c509293c1692a2534805")
    args = ap.parse_args(argv)
    new, old = (repo_dir(REPO) / "snapshots" / rev for rev in (args.revision, R1))
    if not new.is_dir():
        raise SystemExit(f"snapshot not in the local HF cache: {new}")
    rows = []

    def check(name, ok, detail=""):
        rows.append((name, bool(ok), str(detail)))

    files_new = {str(p.relative_to(new)) for p in new.rglob("*") if p.is_file()}
    files_old = {str(p.relative_to(old)) for p in old.rglob("*") if p.is_file()}
    check("file list == R1's", files_new == files_old, sorted(files_new))
    listing = json.loads((repo_dir(REPO) / "trees" / f"{args.revision}.json").read_text())["files"]
    check("local files == cached remote listing", set(listing) == files_new, f"{len(listing)} remote files")
    check("sizes == remote listing", all((new / p).stat().st_size == listing[p]["size"] for p in files_new))
    weights_sha = sha256_file(new / "model.safetensors")
    check("model.safetensors sha256 == R1's", weights_sha == R1_WEIGHTS_SHA256, weights_sha[:16] + "…")
    check("model.safetensors remote LFS sha256 == R1's", listing["model.safetensors"].get("lfs_sha256") == R1_WEIGHTS_SHA256)
    same_blob = os.path.realpath(new / "model.safetensors") == os.path.realpath(old / "model.safetensors")
    check("model.safetensors is the same cache blob as R1's", same_blob,
          Path(os.path.realpath(new / "model.safetensors")).name[:16] + "…")
    small = [p for p in files_new if "lfs_sha256" not in listing[p]]
    check("other files' git blob ids == remote listing",
          all(git_blob_sha1(new / p) == listing[p]["blob_id"] for p in small), f"{len(small)} files")
    refit = HERE / "refit" / "rl_agent_config.json"
    check("rl_agent_config.json byte-identical to refit/rl_agent_config.json",
          (new / "rl_agent_config.json").read_bytes() == refit.read_bytes())
    cfg_new = json.loads((new / "rl_agent_config.json").read_text())
    cfg_old = json.loads((old / "rl_agent_config.json").read_text())
    check("config differs from R1's only in temperature[0]",
          {k: v for k, v in cfg_new.items() if k != "temperature"} == {k: v for k, v in cfg_old.items() if k != "temperature"}
          and cfg_new["temperature"][1:] == cfg_old["temperature"][1:] and cfg_new["temperature"][0] == 2.6968,
          f"{cfg_old['temperature'][0]} -> {cfg_new['temperature'][0]}")
    for p in UNCHANGED:
        check(f"{p} byte-identical to R1's", (new / p).read_bytes() == (old / p).read_bytes())
    meta = json.loads((new / "run_meta.json").read_text())

    width = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\nnote: run_meta.json (unchanged from R1) still records the notebook fit: temperature "
          f"{meta['temperature']}, train.fitted_temperatures {meta['train']['fitted_temperatures']}")
    n_fail = sum(not ok for _, ok, _ in rows)
    print(f"\n{len(rows) - n_fail} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
