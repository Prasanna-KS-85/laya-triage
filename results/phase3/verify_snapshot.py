"""Verify a fine-tuned snapshot in the local HF cache against what the Kaggle notebook promised.

Run from anywhere: .venv/bin/python results/phase3/verify_snapshot.py [--repo R] [--revision SHA] [--eps 0.0]
Offline: reads only the HF cache (snapshot, blobs and the cached remote file listing
trees/<revision>.json), results/phase3/items_sha256.json and the installed package versions. Prints
a PASS/FAIL table and the run_meta.json fields it used, and exits 1 on any FAIL. No issue data.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_REPO, BASE_REV = "convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
EXPECTED_FILES = {"encoder/config.json", "model.safetensors", "rl_agent_config.json", "run_meta.json",
                  "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
OPTIONAL_FILES = {".gitattributes"}
PIN_PACKAGES = ("laya", "transformers", "tokenizers", "huggingface_hub", "safetensors")
CONSTANTS = {"max_len": 1024, "head_max_len": 256, "seed": 42, "epochs": 4, "grad_accum": 4}
PLAN = {"items": 1196, "calib_items": 119, "optimizer_steps": 68, "effective_batch": 64}
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def repo_dir(repo):
    return hub_cache() / ("models--" + repo.replace("/", "--"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def safetensors_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return n, header


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="Prasanna85/laya-issue-triage")
    ap.add_argument("--revision", default="a704b3eadd185f1fa028576cfa50605b6eada4a4")
    ap.add_argument("--eps", default="0.0", help="label smoothing of the run (key in items_sha256.json)")
    args = ap.parse_args(argv)

    snap = repo_dir(args.repo) / "snapshots" / args.revision
    base = repo_dir(BASE_REPO) / "snapshots" / BASE_REV
    if not snap.is_dir():
        raise SystemExit(f"snapshot not in the local HF cache: {snap}")
    rows = []

    def check(name, ok, detail=""):
        rows.append((name, bool(ok), str(detail)))

    # Files
    local = {str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file()}
    check("file list == expected (+ .gitattributes)", EXPECTED_FILES <= local <= EXPECTED_FILES | OPTIONAL_FILES,
          sorted(local))
    check("no checkpoint directory", not any("checkpoint" in p for p in local))
    listing = json.loads((repo_dir(args.repo) / "trees" / f"{args.revision}.json").read_text())["files"]
    check("local files == cached remote listing", set(listing) == local, f"{len(listing)} remote files")
    sizes = {p: (snap / p).stat().st_size for p in local}
    check("sizes == remote listing", all(sizes[p] == listing[p]["size"] for p in local),
          f"total {sum(sizes.values()):,} B")
    lfs = listing["model.safetensors"].get("lfs_sha256")
    weights_sha = sha256_file(snap / "model.safetensors")
    check("model.safetensors sha256 == remote LFS sha256", weights_sha == lfs, weights_sha[:16] + "…")
    small = [p for p in local if "lfs_sha256" not in listing[p]]
    check("other files' git blob ids == remote listing",
          all(git_blob_sha1(snap / p) == listing[p]["blob_id"] for p in small), f"{len(small)} files")

    # Weights
    n_header, header = safetensors_header(snap / "model.safetensors")
    data_end = max(t["data_offsets"][1] for t in header.values())
    check("safetensors header + data == file size", 8 + n_header + data_end == sizes["model.safetensors"])
    _, base_header = safetensors_header(base / "model.safetensors")
    names_shapes = {k: v["shape"] for k, v in header.items()}
    check("tensor names and shapes == base", names_shapes == {k: v["shape"] for k, v in base_header.items()},
          f"{len(header)} tensors")
    dtypes = sorted({v["dtype"] for v in header.values()})
    base_dtypes = {}
    for v in base_header.values():
        base_dtypes[v["dtype"]] = base_dtypes.get(v["dtype"], 0) + 1
    check("all tensors F16", dtypes == ["F16"], f"{dtypes}; base: {base_dtypes}")

    # run_meta.json
    meta = json.loads((snap / "run_meta.json").read_text())
    exp = json.loads((HERE / "items_sha256.json").read_text())
    import importlib
    local_pins = {p: importlib.import_module(p).__version__ for p in PIN_PACKAGES}
    temps = meta["train"]["fitted_temperatures"]
    t_choice = temps[0]
    check("smoke == false", meta["smoke"] is False)
    check(f"items sha256 == items_sha256.json[{args.eps}]", meta["items"]["sha256"] == exp["items"][args.eps]["sha256"],
          meta["items"]["sha256"][:16] + "…")
    check("plan items/calib/steps/effective batch", all(meta["plan"][k] == v for k, v in PLAN.items()),
          {k: meta["plan"][k] for k in PLAN})
    check("train optimizer_steps == scheduler_t_max == 68",
          meta["train"]["optimizer_steps"] == meta["train"]["scheduler_t_max"] == 68)
    consts = {**CONSTANTS, "label_smoothing": float(args.eps)}
    check("constants", all(meta["constants"][k] == v for k, v in consts.items()),
          {k: meta["constants"][k] for k in consts})
    check("versions == local pins", all(meta["versions"][p] == v for p, v in local_pins.items()), local_pins)
    check("data hashes == pinned", meta["data"] == exp["files"])
    check("base == pinned base", meta["base"] == {"repo": BASE_REPO, "revision": BASE_REV})
    check("train_seconds present and > 0", meta["train"].get("train_seconds", 0) > 0, meta["train"].get("train_seconds"))
    check("choice T in (0.5, 5.0), != 1.0, != 1.2",
          TEMP_MIN < t_choice < TEMP_MAX and t_choice not in (1.0, 1.2), round(t_choice, 4))
    check("run_meta temperature == train fitted_temperatures", meta["temperature"] == temps)

    # rl_agent_config.json
    cfg = json.loads((snap / "rl_agent_config.json").read_text())
    check("config max_len 1024 / head_max_len 256", (cfg["max_len"], cfg["head_max_len"]) == (1024, 256))
    check("config fine_tuned true", cfg.get("fine_tuned") is True)
    check("config model_name laya-issue-triage", cfg.get("model_name") == "laya-issue-triage")
    check("config has no temperature_by_options", "temperature_by_options" not in cfg)
    check("config temperature == run_meta", cfg["temperature"] == temps)

    width = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    used = {"smoke": meta["smoke"], "items.sha256": meta["items"]["sha256"], "plan": meta["plan"],
            "train": meta["train"], "constants": meta["constants"], "versions": meta["versions"],
            "data": meta["data"], "base": meta["base"], "temperature": meta["temperature"]}
    print("\nrun_meta.json fields used:\n" + json.dumps(used, indent=1))
    n_fail = sum(not ok for _, ok, _ in rows)
    print(f"\n{len(rows) - n_fail} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
