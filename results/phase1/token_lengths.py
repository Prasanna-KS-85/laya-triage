"""Token lengths of the cleaned states in data/processed/{train,val,test}.jsonl (DATA_CARD.md).

Run from anywhere: .venv/bin/python results/phase1/token_lengths.py
Offline; needs the Laya checkpoint in the local HF cache. Counts state tokens the way laya does
(serialize_state + tokenizer, no special tokens) and compares them with the room the frozen
question leaves at 512/192 and 1024/256 (laya.common.state_room). A cross-check on a few val rows
compares these counts with laya's own `usage` from a real predict(). Writes token_lengths.json
next to this file. Reads issue text but writes only aggregates.
"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("LAYA_CPU_AMP", None)

import laya
from huggingface_hub import snapshot_download
from laya.agent import Agent, _load_tokenizer
from laya.common import encode_text, serialize_state, state_room

from laya_triage.questions import ISSUE_TYPE_QUESTION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO, REV = "convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
SETTINGS = {"512/192": (512, 192), "1024/256": (1024, 256)}
PCTS = (50, 75, 90, 95, 99)
CROSSCHECK_ROWS = 5


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


snap = Path(snapshot_download(REPO, revision=REV, allow_patterns=["rl_agent_config.json", "tokenizer/*"]))
cfg = json.loads((snap / "rl_agent_config.json").read_text())
tok = _load_tokenizer(str(snap / "tokenizer"), cfg)
q = Agent._to_internal(ISSUE_TYPE_QUESTION["issue_type"])
room = {name: state_room(tok, q, ml, hml) for name, (ml, hml) in SETTINGS.items()}


def state_tokens(state):
    text = serialize_state(state).replace(tok.mask_token, " ")
    return len(encode_text(tok, text, add_special_tokens=False)["input_ids"])


report = {"laya": laya.__version__, "revision": REV, "state_room": room, "splits": {}}
states = {}
for split in ("train", "val", "test"):
    with open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    states[split] = [r["state"] for r in rows]
    n = [state_tokens(s) for s in states[split]]
    report["splits"][split] = {
        "rows": len(n),
        "state_tokens_percentiles": {f"p{p}": round(percentile(n, p), 1) for p in PCTS},
        "state_tokens_max": max(n),
        "share_state_tokens_gt_512": round(sum(x > 512 for x in n) / len(n), 4),
        "share_state_tokens_gt_1024": round(sum(x > 1024 for x in n) / len(n), 4),
        **{f"share_truncated_at_{k}": round(sum(x > room[k] for x in n) / len(n), 4) for k in SETTINGS},
    }

# Cross-check against laya's own accounting on the first val rows.
agent = laya.load(REPO, revision=REV, device="cpu")
checks = []
for s in states["val"][:CROSSCHECK_ROWS]:
    mine = state_tokens(s)
    for k, (ml, hml) in SETTINGS.items():
        u = agent.predict(s, ISSUE_TYPE_QUESTION, max_len=ml, head_max_len=hml)["usage"]
        # usage.state_tokens is the full state length (before truncation)
        checks.append(u["state_tokens"] == mine and u["truncated"] == (mine > room[k])
                      and u["state_tokens_dropped"] == max(0, mine - room[k]))
report["crosscheck"] = {"rows": CROSSCHECK_ROWS, "settings": list(SETTINGS), "all_match": all(checks)}

(HERE / "token_lengths.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
sys.exit(0 if all(checks) else 1)
