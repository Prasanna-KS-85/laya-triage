"""Generate training/finetune_kaggle.ipynb (PROJECT_SPEC.md §9.3, Phase 3 task 1).

Usage: .venv/bin/python training/build_notebook.py [--out PATH]
The notebook is generated, never edited by hand. Inputs: training/make_items.py (embedded byte for
byte), results/phase3/items_sha256.json (expected hashes) and TRAIN_DDP below (the adapted upstream
train_ddp.py). tests/test_notebook_sync.py checks that regenerating gives the committed notebook.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training/finetune_kaggle.ipynb"
exp = json.loads((ROOT / "results/phase3/items_sha256.json").read_text(encoding="utf-8"))
make_items_src = (ROOT / "training/make_items.py").read_text(encoding="utf-8")

# Upstream train_ddp.py (Laya notebook at v0.3.23) with the laya-triage changes marked inline.
TRAIN_DDP = r'''# Adapted from the Laya notebook's train_ddp.py (Apache-2.0; see the notebook's first cell).
# Changes are marked "laya-triage:". argv: MODEL_DIR OUTPUT_DIR RUN_CONFIG_JSON
import os, sys, time, json, random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, TEMP_MIN, TEMP_MAX

def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items])
    }

def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # laya-triage: run settings come from the notebook's constants cell instead of literals.
    with open(sys.argv[3]) as f:
        run = json.load(f)
    # laya-triage: seed torch/CUDA per rank (RL noise, dropout). DDP still broadcasts rank 0's
    # weights; GPU kernels stay non-deterministic, so runs are seeded, not bit-reproducible.
    torch.manual_seed(run["seed"] + rank)
    torch.cuda.manual_seed(run["seed"] + rank)
    
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = run["max_len"]            # laya-triage: 1024, same budget the items were built at
    cfg["head_max_len"] = run["head_max_len"]  # laya-triage: 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    all_items = torch.load(run["items_path"], weights_only=False)

    # Hold the calibration slice out of training before sharding. Temperatures fitted on
    # items the run has already trained on measure the fit rather than the calibration: the
    # model is near-certain and near-correct on them, so the optimiser has nothing to soften
    # and returns a degenerate scale. The seed is fixed and rank-independent, so every rank
    # withholds exactly the same items and none of them reaches a training batch.
    CALIB_MAX = 400
    CALIB_SEED = 20260922  # laya-triage: named (upstream literal) so run_meta.json can record it
    order = list(range(len(all_items)))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    # Trim before sharding so every rank gets the same count (drops at most
    # world_size-1 items). Otherwise ranks can run different numbers of
    # micro-batches and DDP deadlocks on the unmatched all-reduce.
    # See https://github.com/NandhaKishorM/laya/issues/678
    train_items = train_items[:len(train_items) // world_size * world_size]
    my_items = train_items[rank::world_size]
    
    EPOCHS = run["epochs"]          # laya-triage: upstream 4; SMOKE 1
    MICRO_BATCH = 8      # 8 sequences per forward pass per GPU
    GRAD_ACCUM = run["grad_accum"]  # laya-triage: upstream 4 (effective batch 8 * 2 * 4 = 64); SMOKE 1
    GROUP_SIZE = 4       # GRPO baseline samples
    LR_ENCODER = 2.5e-5  # Encoder adaptation rate
    LR_HEAD = 1.0e-4     # Head adaptation rate
    SIGMA_START = 0.4    # Exploration noise
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)
    
    # laya-triage: T_max = the optimizer steps the loop below really takes. It also steps on the
    # last, partial accumulation group; upstream's floor division undercounts (64 vs 68 here).
    micro_per_epoch = -(-len(my_items) // MICRO_BATCH)
    total_updates = -(-micro_per_epoch // GRAD_ACCUM) * EPOCHS
    assert total_updates > 0, "no optimizer step would run"
    n_updates = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    if rank == 0:
        print(f"Starting 2xT4 DDP training: {len(train_items)} train items ({len(calib_items)} held out for calibration) | {len(my_items)} per rank | {EPOCHS} epochs")
        print(f"Optimizer steps planned: {total_updates} (scheduler T_max)")
    t0 = time.time()
    
    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        
        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress
        
        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            
            batch = collate_train_batch(chunk, tok.pad_token_id)
            
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device)
                )
            
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            
            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            
            # 2. Evaluate proper scoring reward (w_sph=0.75 for soft target matching)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            
            # 3. Policy gradient loss + full 1.0 soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            
            scaler.scale(loss).backward()
            accum_step += 1
            
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                n_updates += 1
            
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            
            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}")

        if rank == 0:
            print(f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | Avg Loss: {epoch_loss/max(1, n_batches):.4f} ===")

        dist.barrier()

        # Overwrite a single rolling checkpoint after each epoch so a crash,
        # OOM, or Kaggle session timeout doesn't lose all prior training.
        if rank == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "total_epochs": EPOCHS,
                    "avg_loss": epoch_loss / max(1, n_batches)
                }, f, indent=2)
            print(f"  Saved rolling checkpoint (epoch {epoch+1}/{EPOCHS}) to {ckpt_dir}")

    dist.barrier()
    
    # Post-training temperature calibration on rank 0 (micro-batched in chunks of 16 to prevent OOM)
    if rank == 0:
        print("\nFitting post-training calibration temperatures...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        # calib_items was held out above and never entered a training batch
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device)
                    )
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))
        
        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted calibration temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
        except Exception as e:
            print("Temperature fitting fallback:", e)
        os.makedirs(output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-issue-triage"  # laya-triage (not read by laya at load time)
        cfg["temperature"] = fitted_temps
        # This fit is per type; inherited bucket overrides would hide the new values.
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Model successfully saved to {output_dir}!")
        # laya-triage: counts for run_meta.json (no issue text).
        assert n_updates == total_updates, (n_updates, total_updates)
        with open(os.path.join(output_dir, "train_summary.json"), "w") as f:
            json.dump({
                "calib_seed": CALIB_SEED, "calib_items": len(calib_items), "train_items": len(train_items),
                "items_per_rank": len(my_items), "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
                "epochs": EPOCHS, "optimizer_steps": n_updates, "scheduler_t_max": total_updates,
                "train_seconds": round(time.time() - t0, 1), "fitted_temperatures": fitted_temps,
            }, f, indent=2)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''

files = json.dumps(exp["files"], indent=4)
items = json.dumps({k: v["sha256"] for k, v in exp["items"].items()}, indent=4)
smoke = json.dumps({k: v["sha256"] for k, v in exp["smoke"].items()}, indent=4)

MD0 = """# Laya Triage: fine-tune Laya on NLBSE'24 issue types (Kaggle GPU T4 ×2 DDP)

Part of **laya-triage** (PROJECT_SPEC.md §9.3, Phase 3). Fine-tunes `convaiinnovations/laya` at a pinned
revision on our `train.jsonl` (1,196 issues, one `choice` question: `bug` / `feature` / `question`) and pushes
the result to the private HF repo `Prasanna85/laya-issue-triage`.

### Attribution
Adapted from the Laya notebook
[`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/d8a2e59781ca135169a36095056132e273cd9938/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
by NandhaKishorM / Convai Innovations, licensed under the Apache License 2.0, at tag `v0.3.23`
(commit `d8a2e59781ca135169a36095056132e273cd9938`; annotated tag object `ae3222b3fcdf424254a2c726d72161f671a86d95`).
This notebook is a modified version; the changes are listed below.

### Changes from upstream
1. Versions pinned (`laya==0.3.23`, `transformers==5.18.0`) and asserted in the first code cell; upstream ran `pip install -U` with lower bounds.
2. Data: our private Kaggle dataset (`train.jsonl`, `val.jsonl`), found by recursive glob and verified by SHA-256 and row count, replaces `LocalLLaMA/typed-decisions`. Rows hold native JSON objects, so nothing is `json.loads`'ed.
3. Base checkpoint pinned to revision `55cf4c4e…` with only the files `laya.load` needs, downloaded anonymously (`token=False`); upstream downloaded the unpinned full snapshot.
4. Training items are built by `make_items.py` (embedded byte-identical from `training/make_items.py`) at **1024/256**, the budget training and inference use. Upstream built them at the root config's 512/192 while saving 1024/256. Items are checked against a hash computed locally (`results/phase3/items_sha256.json`). Optional label smoothing.
5. `train_ddp.py`: settings read from a run config, torch/CUDA seeded per rank (`SEED + rank`), scheduler `T_max` equals the true optimizer-step count (upstream's floor division undercounts), `model_name = "laya-issue-triage"`, unused imports removed, a `train_summary.json` written. Training loop, loss, calibration split (seed 20260922) and temperature fitting are unchanged.
6. Training is launched with `subprocess` so a failure stops the notebook; post-train checks on the saved config.
7. Upstream's typed-decisions evaluation, metrics, model card and report cells are removed. A 30-row GPU smoke predict on val is optional. Thresholds are fitted later on CPU fp32 (§10.4), not here.
8. Push: `upload_folder` of the model files plus `run_meta.json` to the existing repo, which the notebook never creates; the token comes from Kaggle Secrets `HF_TOKEN` and is never printed.
9. `SMOKE = True` by default: 50 rows, 1 epoch, `GRAD_ACCUM = 1`, no push.

### Kaggle settings
Accelerator **GPU T4 ×2**, Internet **On**, the private dataset with `train.jsonl` and `val.jsonl` attached (never `test.jsonl`),
and the secret `HF_TOKEN` (needed only when `SMOKE = False`). Run the cells top to bottom after every session restart.
"""

C_PINS = """!pip install -q "laya==0.3.23" "transformers==5.18.0"
# 1. Pins (PROJECT_SPEC.md §9.3). Kaggle drops installs when a session restarts: run this cell first, every time.
import sys

import huggingface_hub
import laya
import safetensors
import tokenizers
import torch
import transformers

PINS = {"laya": "0.3.23", "transformers": "5.18.0", "tokenizers": "0.23.2", "huggingface_hub": "1.33.0", "safetensors": "0.8.0"}
assert laya.__version__ == PINS["laya"], laya.__version__
assert transformers.__version__ == PINS["transformers"], transformers.__version__
assert tokenizers.__version__ == PINS["tokenizers"], tokenizers.__version__
assert huggingface_hub.__version__ == PINS["huggingface_hub"], huggingface_hub.__version__
assert safetensors.__version__ == PINS["safetensors"], safetensors.__version__
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda)
print("Pinned versions OK:", PINS)"""

C_GPU = """# 2. GPU check (as upstream)
!nvidia-smi
import os

n_gpu = torch.cuda.device_count()
print(f"CUDA Available: {torch.cuda.is_available()} | Visible GPUs: {n_gpu}")
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory / 1e9:.1f} GB)")
assert n_gpu >= 2, "Expected 2 GPUs: Notebook options -> Accelerator -> GPU T4 x2."
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GPU_NAMES = [torch.cuda.get_device_properties(i).name for i in range(n_gpu)]"""

C_CONST = f"""# 3. Constants. SMOKE = True (default): 50 rows, 1 epoch, GRAD_ACCUM = 1, no push. Set False for a real run.
SMOKE = True

REPO = "Prasanna85/laya-issue-triage"  # exists and is private; never created from here
BASE_REPO = "convaiinnovations/laya"
BASE_REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
UPSTREAM_NOTEBOOK = {{
    "repo": "NandhaKishorM/laya",
    "tag": "v0.3.23",
    "commit": "d8a2e59781ca135169a36095056132e273cd9938",
    "tag_object": "ae3222b3fcdf424254a2c726d72161f671a86d95",
}}
MAX_LEN = 1024
HEAD_MAX_LEN = 256
SEED = 42
EPOCHS = 4
GRAD_ACCUM = 4
LABEL_SMOOTHING = 0.0  # 0.0 or 0.1 (§9.5)
SMOKE_ROWS = 50
MODEL_NAME = "laya-issue-triage"

WORK = "/kaggle/working"
OUTPUT_DIR = f"{{WORK}}/laya_issue_triage"
ITEMS_PATH = f"{{WORK}}/train_items.pt"
RUN_CONFIG = f"{{WORK}}/run_config.json"

# From results/phase3/items_sha256.json (built locally with the same pins and tokenizer).
EXPECTED_FILES = {files}
EXPECTED_ITEMS_SHA256 = {items}
EXPECTED_SMOKE_SHA256 = {smoke}

if SMOKE:
    EPOCHS, GRAD_ACCUM = 1, 1
PUSH = not SMOKE
assert str(LABEL_SMOOTHING) in EXPECTED_ITEMS_SHA256, LABEL_SMOOTHING
print(f"SMOKE={{SMOKE}} PUSH={{PUSH}} EPOCHS={{EPOCHS}} GRAD_ACCUM={{GRAD_ACCUM}} LABEL_SMOOTHING={{LABEL_SMOOTHING}} "
      f"MAX_LEN={{MAX_LEN}} HEAD_MAX_LEN={{HEAD_MAX_LEN}} SEED={{SEED}}")"""

C_DATA = """# 4. Data: exactly one train.jsonl under /kaggle/input, verified by SHA-256, size and row count.
import glob
import hashlib


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


hits = sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
assert len(hits) == 1, f"expected exactly one train.jsonl under /kaggle/input, found {len(hits)}: {hits}"
assert not glob.glob("/kaggle/input/**/test.jsonl", recursive=True), "test.jsonl must never be attached on Kaggle"
DATA_DIR = os.path.dirname(hits[0])
DATA = {}
for name, expected in EXPECTED_FILES.items():
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as f:
        n_rows = sum(1 for line in f if line.strip())
    got = {"sha256": sha256_file(path), "bytes": os.path.getsize(path), "rows": n_rows}
    assert got == expected, f"{name}: {got} != {expected}"
    DATA[name] = path
    print(f"{name}: OK {got}")"""

C_BASE = """# 5. Base checkpoint at the pinned revision, minimal file set, anonymous download (the HF token is only for the push).
import json

from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config

MODEL_DIR = snapshot_download(
    BASE_REPO,
    revision=BASE_REV,
    allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    token=False,
)
_fix_tokenizer_config(MODEL_DIR)  # as upstream; train_ddp.py loads the tokenizer from MODEL_DIR
with open(os.path.join(MODEL_DIR, "rl_agent_config.json")) as f:
    base_cfg = json.load(f)
assert (base_cfg["max_len"], base_cfg["head_max_len"]) == (512, 192), "unexpected base config"
print("Base:", BASE_REPO, "@", BASE_REV, "| native max_len/head_max_len:", base_cfg["max_len"], base_cfg["head_max_len"])"""

C_MAKE_ITEMS = "%%writefile /kaggle/working/make_items.py\n" + make_items_src

C_ITEMS = """# 7. Training items at MAX_LEN / HEAD_MAX_LEN; the hash must equal the local build (items-hash parity).
sys.path.insert(0, WORK)
import make_items

rows = make_items.load_rows(DATA["train.jsonl"])
if SMOKE:
    rows = make_items.smoke_subset(rows, SMOKE_ROWS)  # spread over the file, not the first rows
item_tok = make_items.load_tokenizer(os.path.join(MODEL_DIR, "tokenizer"))
items = make_items.build_items(rows, item_tok, MAX_LEN, HEAD_MAX_LEN, LABEL_SMOOTHING)
ITEMS_SHA256 = make_items.items_sha256(items)
expected = (EXPECTED_SMOKE_SHA256 if SMOKE else EXPECTED_ITEMS_SHA256)[str(LABEL_SMOOTHING)]
assert ITEMS_SHA256 == expected, f"items hash {ITEMS_SHA256} != expected {expected}: parity with the local build broken"
ITEMS_SUMMARY = make_items.summarize(items)
print(json.dumps({"rows": len(rows), **ITEMS_SUMMARY, "sha256": ITEMS_SHA256}, indent=1))
torch.save(items, ITEMS_PATH)
print("Saved", len(items), "items to", ITEMS_PATH)"""

C_PLAN = """# 8. Training plan (same arithmetic as train_ddp.py).
WORLD_SIZE, MICRO_BATCH, CALIB_MAX = 2, 8, 400
n_calib = min(CALIB_MAX, len(items) // 10)
n_train = (len(items) - n_calib) // WORLD_SIZE * WORLD_SIZE
per_rank = n_train // WORLD_SIZE
micro_per_epoch = -(-per_rank // MICRO_BATCH)
steps_per_epoch = -(-micro_per_epoch // GRAD_ACCUM)
PLAN = {
    "items": len(items), "calib_items": n_calib, "train_items": n_train, "items_per_rank": per_rank,
    "micro_batches_per_epoch": micro_per_epoch, "optimizer_steps_per_epoch": steps_per_epoch,
    "epochs": EPOCHS, "optimizer_steps": steps_per_epoch * EPOCHS,
    "effective_batch": MICRO_BATCH * WORLD_SIZE * GRAD_ACCUM,
}
assert PLAN["optimizer_steps"] > 0, "total_updates would be 0"
print("Training plan:", json.dumps(PLAN, indent=1))"""

C_TRAIN_DDP = "%%writefile /kaggle/working/train_ddp.py\n" + TRAIN_DDP

C_LAUNCH = '''# 10. Launch torchrun on both GPUs; a non-zero exit stops the notebook.
import subprocess

with open(RUN_CONFIG, "w") as f:
    json.dump({"seed": SEED, "epochs": EPOCHS, "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN,
               "head_max_len": HEAD_MAX_LEN, "items_path": ITEMS_PATH}, f, indent=1)
cmd = ["torchrun", "--standalone", "--nproc_per_node=2", f"{WORK}/train_ddp.py", MODEL_DIR, OUTPUT_DIR, RUN_CONFIG]
print("Executing DDP training:", " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
assert proc.wait() == 0, f"torchrun exited with {proc.returncode}"'''

C_CHECKS = """# 11. Post-train checks on the saved model.
from laya.common import TEMP_MAX, TEMP_MIN
from safetensors import safe_open

with open(os.path.join(OUTPUT_DIR, "rl_agent_config.json")) as f:
    out_cfg = json.load(f)
with open(os.path.join(OUTPUT_DIR, "train_summary.json")) as f:
    TRAIN_SUMMARY = json.load(f)

for rel in ("model.safetensors", "rl_agent_config.json", "encoder/config.json"):
    assert os.path.isfile(os.path.join(OUTPUT_DIR, rel)), rel
assert os.listdir(os.path.join(OUTPUT_DIR, "tokenizer")), "tokenizer/ is empty"
assert (out_cfg["max_len"], out_cfg["head_max_len"]) == (MAX_LEN, HEAD_MAX_LEN), (out_cfg["max_len"], out_cfg["head_max_len"])
assert out_cfg["fine_tuned"] is True
assert out_cfg["model_name"] == MODEL_NAME, out_cfg["model_name"]
assert "temperature_by_options" not in out_cfg
temps = out_cfg["temperature"]
assert len(temps) == 3 and all(TEMP_MIN <= t <= TEMP_MAX for t in temps), temps
assert TRAIN_SUMMARY["optimizer_steps"] == PLAN["optimizer_steps"], (TRAIN_SUMMARY["optimizer_steps"], PLAN)
assert TRAIN_SUMMARY["calib_items"] == PLAN["calib_items"], (TRAIN_SUMMARY["calib_items"], PLAN)

t_choice = temps[0]  # order: choice, score, noul; only choice is fitted here (score/noul keep 1.2)
if PLAN["calib_items"] < 10:
    assert t_choice == 1.0, t_choice  # fit_one_temp returns 1.0 below 10 calibration items (SMOKE)
else:
    assert t_choice != 1.2, "choice temperature is the unfitted fallback 1.2"
if abs(t_choice - TEMP_MIN) < 1e-6 or abs(t_choice - TEMP_MAX) < 1e-6:
    print(f"WARNING: fitted choice temperature {t_choice} sits at the clamp bound [{TEMP_MIN}, {TEMP_MAX}]")


def tensor_names(path):
    with safe_open(path, "pt") as f:
        return set(f.keys())


assert tensor_names(os.path.join(OUTPUT_DIR, "model.safetensors")) == tensor_names(os.path.join(MODEL_DIR, "model.safetensors"))
print("Post-train checks OK | temperatures (choice, score, noul):", temps, "| optimizer steps:", TRAIN_SUMMARY["optimizer_steps"])"""

C_PREDICT = """# 12. Optional GPU smoke predict on 30 val rows. GPU numerics: a sanity check only, never for thresholds (§10.4).
from collections import Counter

RUN_GPU_PREDICT = True
GPU_PREDICT = None
if RUN_GPU_PREDICT:
    with open(DATA["val.jsonl"], encoding="utf-8") as f:  # §8.6 eval rows: "expected" label, no "gold"
        val_rows = make_items.smoke_subset([json.loads(line) for line in f if line.strip()], 30)
    agent = laya.Agent(OUTPUT_DIR, device="cuda")
    assert agent.cfg["max_len"] == MAX_LEN
    res = agent.predict_batch([r["state"] for r in val_rows], make_items.ISSUE_TYPE_QUESTION)
    preds = [r["answers"]["issue_type"]["choice"] for r in res]
    golds = [r["expected"]["issue_type"] for r in val_rows]
    GPU_PREDICT = {
        "rows": len(val_rows),
        "accuracy": round(sum(p == g for p, g in zip(preds, golds)) / len(val_rows), 4),
        "pred_counts": dict(Counter(preds)),
        "gold_counts": dict(Counter(golds)),
    }
    print("GPU smoke predict:", GPU_PREDICT)
    del agent
    torch.cuda.empty_cache()"""

C_META = """# 13. run_meta.json: versions, constants, hashes, temperatures. No issue text.
import datetime
import platform

RUN_META = {
    "project": "laya-triage",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "smoke": SMOKE,
    "upstream_notebook": UPSTREAM_NOTEBOOK,
    "base": {"repo": BASE_REPO, "revision": BASE_REV},
    "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, **PINS},
    "gpus": GPU_NAMES,
    "constants": {
        "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN, "seed": SEED, "epochs": EPOCHS,
        "grad_accum": GRAD_ACCUM, "label_smoothing": LABEL_SMOOTHING, "model_name": MODEL_NAME,
    },
    "data": EXPECTED_FILES,
    "items": {"sha256": ITEMS_SHA256, **ITEMS_SUMMARY},
    "plan": PLAN,
    "train": TRAIN_SUMMARY,
    "temperature": out_cfg["temperature"],
    "gpu_smoke_predict": GPU_PREDICT,
}
with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
    json.dump(RUN_META, f, indent=1)
print(json.dumps(RUN_META, indent=1))"""

C_PUSH = """# 14. Push to the existing HF repo (skipped when SMOKE); the repo is never created here and the token never printed.
import fnmatch

UPLOAD_PATTERNS = ["model.safetensors", "rl_agent_config.json", "tokenizer/*", "encoder/*", "run_meta.json"]
if not PUSH:
    print("SMOKE run: push skipped.")
else:
    from huggingface_hub import HfApi
    from kaggle_secrets import UserSecretsClient

    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
    info = api.repo_info(REPO, repo_type="model")
    print(f"{REPO} visibility: {'private' if info.private else 'PUBLIC'}")
    local = sorted(os.path.relpath(os.path.join(d, n), OUTPUT_DIR) for d, _, names in os.walk(OUTPUT_DIR) for n in names)
    print("Uploading:", [p for p in local if any(fnmatch.fnmatch(p, pat) for pat in UPLOAD_PATTERNS)])
    commit = api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=REPO,
        repo_type="model",
        allow_patterns=UPLOAD_PATTERNS,
        commit_message=f"laya-triage fine-tune: items {ITEMS_SHA256[:12]}, eps {LABEL_SMOOTHING}, epochs {EPOCHS}, seed {SEED}",
    )
    print("HF commit oid (record in results/experiments.md):", commit.oid)"""


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = [md(MD0), code(C_PINS), code(C_GPU), code(C_CONST), code(C_DATA),
         code(C_BASE), code(C_MAKE_ITEMS), code(C_ITEMS), code(C_PLAN), code(C_TRAIN_DDP), code(C_LAUNCH),
         code(C_CHECKS), code(C_PREDICT), code(C_META), code(C_PUSH)]
nb = {
    "cells": cells,
    "metadata": {
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def render():
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate training/finetune_kaggle.ipynb.")
    ap.add_argument("--out", type=Path, default=NOTEBOOK)
    out = ap.parse_args(argv).out
    out.write_text(render(), encoding="utf-8")
    print("wrote", out, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
