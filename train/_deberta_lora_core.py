"""Shared LoRA fine-tuning and inference utilities for DeBERTa-v3-large.

Training entry points import this module instead of exposing it directly. The
workflow trains two adapters:
- label: three-way detection, label in {0, 1, 2}
- family: source attribution for machine text, family in {0..7}
"""

from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model, DebertaV2PreTrainedModel
from transformers import (
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
)


DEFAULT_MODEL_DIR = ROOT / "model"
DEFAULT_TRAIN = ROOT / "data" / "train.jsonl"
DEFAULT_TEST = ROOT / "data" / "test_a_release.jsonl"
DEFAULT_OUTPUT = ROOT / "ckpt"
DEFAULT_SUBMISSION_DIR = ROOT / "result" / "submissions"
CSV_DIR = ROOT / "analysis" / "csv"

LABEL_NAMES = {0: "human", 1: "machine", 2: "hybrid"}
FAMILY_NAMES = {
    0: "openai",
    1: "alibaba",
    2: "deepseek",
    3: "bytedance",
    4: "moonshot",
    5: "google",
    6: "anthropic",
    7: "xai",
}
HEAD_PARAM_KEYWORDS = ("classifier",)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def maybe_sample(rows: list[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    if not sample or sample >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample)


def task_rows(rows: list[dict[str, Any]], task: str) -> tuple[list[str], list[int], dict[int, str]]:
    if task == "label":
        return [r["text"] for r in rows], [int(r["label"]) for r in rows], LABEL_NAMES
    if task == "family":
        machine_rows = [r for r in rows if int(r["label"]) == 1]
        return [r["text"] for r in machine_rows], [int(r["family"]) for r in machine_rows], FAMILY_NAMES
    raise ValueError(f"Unknown task: {task}")


class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer,
        max_length: int,
        max_chars: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chars = max_chars

    def __len__(self) -> int:
        return len(self.texts)

    def _trim_text(self, text: str) -> str:
        if self.max_chars <= 0 or len(text) <= self.max_chars:
            return text
        half = self.max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self._trim_text(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_tokenizer(model_dir: Path):
    return DebertaV2Tokenizer.from_pretrained(model_dir)


class StylePoolingClassificationHead(nn.Module):
    """Pooling head for authorship/source attribution style signals."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        hidden_size_out: int,
        dropout: float,
        multi_sample_dropout: int,
    ) -> None:
        super().__init__()
        self.attention_pool = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size * 4)
        self.fc = nn.Linear(hidden_size * 4, hidden_size_out)
        self.out_proj = nn.Linear(hidden_size_out, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropout) for _ in range(max(1, multi_sample_dropout))
        )
        self.activation = nn.GELU()
        self.multi_sample_dropout = max(1, multi_sample_dropout)

    def _masked_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        token_count = mask.sum(dim=1).clamp(min=1.0)

        cls_pool = hidden_states[:, 0]
        mean_pool = (hidden_states * mask).sum(dim=1) / token_count

        # Use finite sentinels that are safe under CUDA autocast/float16.
        hidden_mask_value = -1e4 if hidden_states.dtype == torch.float16 else torch.finfo(hidden_states.dtype).min
        max_pool = hidden_states.masked_fill(mask == 0, hidden_mask_value).max(dim=1).values

        attn_scores = self.attention_pool(hidden_states).squeeze(-1)
        attn_mask_value = -1e4 if attn_scores.dtype == torch.float16 else torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(attention_mask == 0, attn_mask_value)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(dtype=hidden_states.dtype)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)

        return torch.cat([cls_pool, mean_pool, max_pool, attn_pool], dim=-1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self._masked_pool(hidden_states, attention_mask)
        features = self.activation(self.fc(self.norm(pooled)))
        if self.training and self.multi_sample_dropout > 1:
            logits = [self.out_proj(dropout(features)) for dropout in self.dropouts]
            return torch.stack(logits, dim=0).mean(dim=0)
        return self.out_proj(self.dropout(features))


class DebertaV2ForStyleClassification(DebertaV2PreTrainedModel):
    def __init__(
        self,
        config,
        head_hidden_size: int = 0,
        head_dropout: float = 0.2,
        multi_sample_dropout: int = 5,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        hidden_size_out = head_hidden_size if head_hidden_size > 0 else config.hidden_size
        self.classifier = StylePoolingClassificationHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            hidden_size_out=hidden_size_out,
            dropout=head_dropout,
            multi_sample_dropout=multi_sample_dropout,
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if attention_mask is None:
            attention_mask = torch.ones(sequence_output.shape[:2], device=sequence_output.device, dtype=torch.long)

        logits = self.classifier(sequence_output, attention_mask)
        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1)) if labels is not None else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def make_base_model(
    model_dir: Path,
    num_labels: int,
    device_type: str,
    args: argparse.Namespace | None = None,
):
    return DebertaV2ForStyleClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        dtype=torch.float16 if device_type == "cuda" else torch.float32,
        head_hidden_size=args.head_hidden_size if args is not None else 0,
        head_dropout=args.head_dropout if args is not None else 0.2,
        multi_sample_dropout=args.multi_sample_dropout if args is not None else 5,
    )


def make_lora_model(model_dir: Path, num_labels: int, args: argparse.Namespace, print_trainable: bool = True):
    model = make_base_model(
        model_dir,
        num_labels,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        args=args,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_proj", "key_proj", "value_proj"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    if print_trainable:
        model.print_trainable_parameters()
    return model


def log_stage(message: str, rank: int = 0, main_only: bool = True) -> None:
    if main_only and rank != 0:
        return
    timestamp = datetime.now().isoformat(timespec="seconds")
    tqdm.write(f"[{timestamp}][rank={rank}] {message}")


def json_safe_config(args: argparse.Namespace, extra: dict[str, Any]) -> dict[str, Any]:
    config = vars(args).copy()
    config.update(extra)
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def init_wandb(args: argparse.Namespace, task: str, rank: int, extra_config: dict[str, Any]):
    if not args.wandb or rank != 0:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb is enabled but not installed. Install with: pip install wandb") from exc

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=json_safe_config(args, {"task": task, **extra_config}),
    )
    return run


def wandb_log(run, data: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(data, step=step)


def setup_runtime(args: argparse.Namespace, distributed: bool) -> tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.cpu:
        if distributed:
            raise RuntimeError("Distributed training requires CUDA; remove --cpu.")
        return torch.device("cpu"), False, 0, 1, 0

    if not distributed:
        if world_size > 1:
            raise RuntimeError("Single-GPU entry was launched with torchrun. Use train_deberta_lora_ddp.py instead.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False, 0, 1, 0

    if distributed:
        if world_size <= 1:
            raise RuntimeError("DDP entry requires torchrun, for example: torchrun --nproc_per_node 4 ...")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun was used but CUDA is not available.")
        log_stage(
            f"init distributed backend={getattr(args, 'dist_backend', 'nccl')} "
            f"world_size={world_size} local_rank={local_rank}",
            rank,
            main_only=False,
        )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=getattr(args, "dist_backend", "nccl"))
        log_stage(f"distributed ready device=cuda:{local_rank}", rank, main_only=False)
        return torch.device("cuda", local_rank), True, rank, world_size, local_rank

    raise RuntimeError("Unreachable runtime state.")


def cleanup_runtime(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_optimizer(model, args: argparse.Namespace, rank: int) -> torch.optim.Optimizer:
    lora_params = []
    head_params = []
    lora_count = 0
    head_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in HEAD_PARAM_KEYWORDS):
            head_params.append(param)
            head_count += param.numel()
        else:
            lora_params.append(param)
            lora_count += param.numel()

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.head_weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    log_stage(
        f"optimizer lora_lr={args.lr} head_lr={args.head_lr} "
        f"lora_params={lora_count:,} head_params={head_count:,}",
        rank,
    )
    return torch.optim.AdamW(param_groups)


@torch.no_grad()
def predict_logits(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits_list: list[np.ndarray] = []
    for batch in tqdm(loader, desc="predict", dynamic_ncols=True, mininterval=1.0, leave=True):
        batch = move_batch(batch, device)
        batch.pop("labels", None)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(**batch)
        logits_list.append(outputs.logits.detach().float().cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def evaluate(model, loader: DataLoader, device: torch.device, labels: list[int], num_labels: int) -> dict[str, float]:
    logits = predict_logits(model, loader, device)
    pred = logits.argmax(axis=1).astype(int).tolist()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, labels=list(range(num_labels)), average="macro", zero_division=0)),
    }


def save_metadata(out_dir: Path, args: argparse.Namespace, metrics: dict[str, float], class_names: dict[int, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "class_names.json").write_text(
        json.dumps({str(k): v for k, v in class_names.items()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_task(task: str, args: argparse.Namespace, distributed: bool = False) -> None:
    device, distributed, rank, world_size, local_rank = setup_runtime(args, distributed)
    is_main = rank == 0
    log_stage(f"start task={task} device={device} distributed={distributed} world_size={world_size}", rank)
    seed_everything(args.seed + rank)
    log_stage(f"read train data from {args.train}", rank)
    rows = maybe_sample(read_jsonl(args.train), args.sample, args.seed)
    texts, labels, class_names = task_rows(rows, task)
    num_labels = len(class_names)
    log_stage(f"task_rows={len(texts)} num_labels={num_labels} sample={args.sample}", rank)

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts,
        labels,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=labels,
    )
    log_stage(f"split train_rows={len(train_texts)} valid_rows={len(valid_texts)}", rank)

    log_stage(f"load tokenizer from {args.model_dir}", rank)
    tokenizer = make_tokenizer(args.model_dir)
    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_length, args.max_chars)
    valid_ds = TextDataset(valid_texts, valid_labels, tokenizer, args.max_length, args.max_chars)
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    loader_options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_options,
    )
    log_stage(
        f"dataloader train_batches_per_rank={len(train_loader)} valid_batches={len(valid_loader)} "
        f"batch_size={args.batch_size} eval_batch_size={args.eval_batch_size}",
        rank,
    )

    wandb_run = init_wandb(
        args,
        task,
        rank,
        {
            "world_size": world_size,
            "train_rows_total": len(texts),
            "train_rows": len(train_texts),
            "valid_rows": len(valid_texts),
            "num_labels": num_labels,
        },
    )

    log_stage(f"load model from {args.model_dir}", rank)
    model = make_lora_model(args.model_dir, num_labels, args, print_trainable=is_main)
    log_stage("move model to device", rank)
    model.to(device)
    if distributed:
        log_stage("wrap model with DDP", rank)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=getattr(args, "ddp_find_unused_parameters", False),
        )
        log_stage("DDP model ready", rank)
    optimizer = make_optimizer(model, args, rank)
    if wandb_run is not None:
        import wandb

        wandb.watch(unwrap_model(model), log="gradients", log_freq=100)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    out_dir = args.output_dir / task
    best_f1 = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, args.epochs + 1):
            log_stage(f"start epoch {epoch}/{args.epochs}", rank)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            progress = tqdm(
                train_loader,
                desc=f"{task} epoch {epoch}/{args.epochs}",
                disable=not is_main,
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )
            for step, batch in enumerate(progress, start=1):
                batch = move_batch(batch, device)
                should_sync = step % args.grad_accum == 0 or step == len(train_loader)
                sync_context = model.no_sync() if distributed and not should_sync else nullcontext()
                with sync_context:
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        loss = model(**batch).loss / args.grad_accum
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")
                    scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * args.grad_accum

                if should_sync:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scale_before_step = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= scale_before_step:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if is_main:
                        current_loss = running_loss / step
                        current_lr = scheduler.get_last_lr()[0]
                        progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")
                        wandb_log(
                            wandb_run,
                            {
                                f"{task}/train_loss": current_loss,
                                f"{task}/lr": current_lr,
                                f"{task}/epoch": epoch,
                            },
                            step=global_step,
                        )

            if is_main:
                eval_model = unwrap_model(model)
                metrics = evaluate(eval_model, valid_loader, device, valid_labels, num_labels)
                metrics.update(
                    {
                        "task": task,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_rows": len(train_texts),
                        "valid_rows": len(valid_texts),
                        "max_length": args.max_length,
                        "lora_r": args.lora_r,
                        "world_size": world_size,
                    }
                )
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
                wandb_log(
                    wandb_run,
                    {
                        f"{task}/valid_accuracy": metrics["accuracy"],
                        f"{task}/valid_macro_f1": metrics["macro_f1"],
                        f"{task}/best_macro_f1": max(best_f1, metrics["macro_f1"]),
                    },
                    step=global_step,
                )

                if metrics["macro_f1"] > best_f1:
                    best_f1 = metrics["macro_f1"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    eval_model.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    save_metadata(out_dir, args, metrics, class_names)
                    print(f"Saved best {task} adapter to {out_dir}")
            barrier(distributed)
    finally:
        if "wandb_run" in locals() and wandb_run is not None:
            wandb_run.finish()
        cleanup_runtime(distributed)


def load_adapter_model(base_dir: Path, adapter_dir: Path, num_labels: int, device: torch.device):
    model = make_base_model(
        base_dir,
        num_labels,
        device_type=device.type,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()
    return model


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_test(args: argparse.Namespace) -> None:
    device, _, _, _, _ = setup_runtime(args, distributed=False)
    rows = read_jsonl(args.test)
    if args.predict_limit and args.predict_limit > 0:
        rows = rows[: args.predict_limit]
    texts = [r["text"] for r in rows]
    ids = [r["id"] for r in rows]

    tokenizer = make_tokenizer(args.output_dir / "label")
    label_model = load_adapter_model(args.model_dir, args.output_dir / "label", len(LABEL_NAMES), device)
    test_ds = TextDataset(texts, None, tokenizer, args.max_length, args.max_chars)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    label_logits = predict_logits(label_model, test_loader, device)
    label_prob = softmax(label_logits)
    label_pred = label_prob.argmax(axis=1).astype(int)

    family_pred = np.full(len(rows), -1, dtype=int)
    machine_idx = np.where(label_pred == 1)[0]
    family_prob_full = np.zeros((len(rows), len(FAMILY_NAMES)), dtype=np.float32)
    if len(machine_idx) > 0:
        family_model = load_adapter_model(args.model_dir, args.output_dir / "family", len(FAMILY_NAMES), device)
        family_texts = [texts[i] for i in machine_idx]
        family_ds = TextDataset(family_texts, None, tokenizer, args.max_length, args.max_chars)
        family_loader = DataLoader(family_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        family_logits = predict_logits(family_model, family_loader, device)
        family_prob = softmax(family_logits)
        family_pred[machine_idx] = family_prob.argmax(axis=1).astype(int)
        family_prob_full[machine_idx] = family_prob

    out_rows = [
        {"id": sample_id, "label": int(label), "family": int(family)}
        for sample_id, label, family in zip(ids, label_pred, family_pred)
    ]

    args.submission_dir.mkdir(parents=True, exist_ok=True)
    submit_path = args.submission_dir / "submit_deberta_lora.jsonl"
    write_jsonl(submit_path, out_rows)

    pred_df = pd.DataFrame({"id": ids, "label": label_pred, "family": family_pred})
    pred_df["label_name"] = pred_df["label"].map(LABEL_NAMES)
    pred_df["family_name"] = pred_df["family"].map(FAMILY_NAMES).fillna("none")
    for i, name in LABEL_NAMES.items():
        pred_df[f"prob_label_{name}"] = label_prob[:, i]
    for i, name in FAMILY_NAMES.items():
        pred_df[f"prob_family_{name}"] = family_prob_full[:, i]
    csv_path = args.submission_dir / "test_a_deberta_lora_predictions.csv"
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {submit_path}")
    print(f"Wrote {csv_path}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser, include_ddp_args: bool = False) -> None:
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=0, help="Optional row sample for smoke tests.")
    parser.add_argument("--predict-limit", type=int, default=0, help="Optional test row limit for prediction smoke tests.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--head-hidden-size", type=int, default=0, help="0 uses the backbone hidden size.")
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--multi-sample-dropout", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="ccks2026")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    if include_ddp_args:
        parser.add_argument("--dist-backend", default="nccl")
        parser.add_argument("--ddp-find-unused-parameters", action="store_true")


def run_cli(distributed: bool = False, include_predict: bool = True) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = ["train-label", "train-family"]
    if include_predict:
        commands.append("predict")
    for command in commands:
        sub = subparsers.add_parser(command)
        add_common_args(sub, include_ddp_args=distributed)
    args = parser.parse_args()

    if args.command == "train-label":
        train_task("label", args, distributed=distributed)
    elif args.command == "train-family":
        train_task("family", args, distributed=distributed)
    elif args.command == "predict":
        predict_test(args)
    else:
        raise ValueError(args.command)
