"""NFR-1 latency probe for the GitHub runner (PROJECT_SPEC.md NFR-1, §17 Q4): CPU p50 / p95 per issue, model preloaded.

  python latency_probe.py ISSUES_JSON --default-config TRIAGE_DEFAULT_YML [--repeats 3]

Run with the Action's venv python (~/.cache/laya-triage/venv/bin/python) after the Action step, so the model is in
the Hugging Face cache. Loads the pinned model once on CPU fp32, makes 3 warm-up calls, then times one
`Classifier.classify` (one `predict`) per fixture that is not skipped, `--repeats` times. Prints and appends to
$GITHUB_STEP_SUMMARY: runner CPU model, logical CPUs, RAM, torch threads, load time, p50 / p95 / mean / max.
No issue text is printed.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.pop("LAYA_CPU_AMP", None)


def runner_info():
    cpu, ram = platform.processor() or platform.machine(), None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram = round(int(line.split()[1]) / 1024)
                break
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_mib": ram, "system": platform.platform()}


def percentile(values, q):
    import numpy as np

    return float(np.percentile(values, q))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("issues")
    ap.add_argument("--default-config", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)

    import torch

    from laya_triage.classifier import Classifier
    from laya_triage.config import load_config
    from laya_triage.event_loader import issue_from_api, should_skip
    from laya_triage.preprocess import preprocess

    cfg = load_config(args.default_config)
    states = []
    for n, f in enumerate(json.loads(Path(args.issues).read_text(encoding="utf-8"))["fixtures"], 1):
        issue = issue_from_api({"number": n, "title": f["title"], "body": f["body"], "user": {"login": "probe"},
                                "labels": [{"name": x} for x in f["labels_at_creation"]]}, "opened")
        if not should_skip(issue, cfg):
            states.append((n, preprocess(issue.title, issue.body, cfg.preprocess)))
    t = time.perf_counter()
    clf = Classifier(cfg.model.repo, cfg.model.revision, cfg.model.max_len)
    load_s = time.perf_counter() - t
    for n, s in states[:3]:
        clf.classify(s, n)
    ms = [clf.classify(s, n).latency_ms for _ in range(args.repeats) for n, s in states]
    out = {**runner_info(), "torch_threads": torch.get_num_threads(), "load_s": round(load_s, 2), "n": len(ms),
           "p50_ms": round(percentile(ms, 50), 1), "p95_ms": round(percentile(ms, 95), 1),
           "mean_ms": round(sum(ms) / len(ms), 1), "max_ms": round(max(ms), 1),
           "nfr1_p95_le_1000ms": percentile(ms, 95) <= 1000}
    print(json.dumps(out, indent=1))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Latency probe (NFR-1)\n\n| " + " | ".join(out) + " |\n|" + "---|" * len(out) + "\n| "
                    + " | ".join(str(v) for v in out.values()) + " |\n\n")


if __name__ == "__main__":
    main()
