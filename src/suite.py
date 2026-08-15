from __future__ import annotations

import inspect
import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.result_analysis import (
    prediction_diagnostics,
    require_matching_run_signature,
    summarize_run_frame,
    validate_run_frame_integrity,
    public_run_frame,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment_config.json"
_default_cache_root = ROOT / "cache"
if "::" in str(_default_cache_root):
    # fsspec interprets ``::`` in a local path as a chained protocol. This
    # repository is sometimes checked out under a URL-shaped directory name.
    cache_slug = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:12]
    _default_cache_root = Path(tempfile.gettempdir()) / f"domain_finetuning_cache_{cache_slug}"
DATA_CACHE_ROOT = Path(os.environ.get("DOMAIN_FINETUNING_DATA_CACHE", _default_cache_root)).expanduser().resolve()
METHODS = ("full_ft", "lora", "adapter", "ia3", "bitfit")


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_event(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now_iso(), **payload}, ensure_ascii=False) + "\n")
        handle.flush()


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def runtime_info():
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    accelerator = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    return {
        "time": now_iso(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_available,
        "accelerator": accelerator,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Apple MPS" if mps_available else "CPU",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3) if torch.cuda.is_available() else 0,
        "machine": platform.machine(),
    }


def precheck(require_cuda=True, output_path=None):
    import datasets
    import peft
    import transformers

    info = runtime_info() | {
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
    }
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 감지되지 않았습니다. Python (ai_lab_first) 커널인지 확인하세요.")
    atomic_json(Path(output_path) if output_path else ROOT / "results" / "environment.json", info)
    return info


@dataclass(frozen=True)
class TaskSpec:
    key: str
    path: str
    subset: str | None
    text_col: str
    label_col: str
    num_labels: int
    source_split: str | None = None
    label_threshold: float | None = None
    direct: str | None = None


TASKS = {
    "measuring_hate_speech": TaskSpec("measuring_hate_speech", "ucberkeley-dlab/measuring-hate-speech", None, "comment", "hatespeech", 2, "train", 1.0),
    "tweet_sentiment": TaskSpec("tweet_sentiment", "cardiffnlp/tweet_eval", "sentiment", "text", "label", 3),
    "finance_sentiment": TaskSpec("finance_sentiment", "lmassaron/FinancialPhraseBank", None, "sentence", "label", 3),
    "movie_reviews": TaskSpec("movie_reviews", "stanfordnlp/imdb", None, "text", "label", 2),
    "product_reviews": TaskSpec("product_reviews", "SetFit/amazon_reviews_multi_en", None, "text", "label", 5),
    "tweet_emotion": TaskSpec("tweet_emotion", "cardiffnlp/tweet_eval", "emotion", "text", "label", 4),
    "tweet_hate": TaskSpec("tweet_hate", "cardiffnlp/tweet_eval", "hate", "text", "label", 2),
    "tweet_offensive": TaskSpec("tweet_offensive", "cardiffnlp/tweet_eval", "offensive", "text", "label", 2),
    "tweet_irony": TaskSpec("tweet_irony", "cardiffnlp/tweet_eval", "irony", "text", "label", 2),
    "news_topic": TaskSpec("news_topic", "fancyzhx/ag_news", None, "text", "label", 4),
    "news_ynat": TaskSpec("news_ynat", "klue", "ynat", "title", "label", 7),
    "movie_nsmc": TaskSpec("movie_nsmc", "csv", None, "document", "label", 2, direct="nsmc"),
    "comment_kmhas_binary": TaskSpec("comment_kmhas_binary", "csv", None, "text", "label", 2, direct="kmhas"),
}


def _raw_dataset(spec: TaskSpec):
    if spec.direct == "nsmc":
        return load_dataset("csv", data_files={
            "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
            "test": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
        }, delimiter="\t")
    if spec.direct == "kmhas":
        return load_dataset("csv", data_files={
            "train": "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_train.txt",
            "validation": "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_valid.txt",
            "test": "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_test.txt",
        }, delimiter="\t", column_names=["text", "label"], skiprows=1)
    return load_dataset(spec.path, spec.subset) if spec.subset else load_dataset(spec.path)


def _kmhas_binary(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [int(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (int, np.integer)):
        value = [int(value)]
    return 0 if list(value) == [8] else 1


def _standardize(raw, spec: TaskSpec):
    if spec.source_split:
        raw = DatasetDict(all=raw[spec.source_split])
    standardized = {}
    for split_name, split in raw.items():
        texts, labels, ids = [], [], []
        columns = set(split.column_names)
        text_col = spec.text_col if spec.text_col in columns else next((x for x in ("text", "sentence", "comment", "document", "title") if x in columns), None)
        label_col = spec.label_col if spec.label_col in columns else next((x for x in ("label", "labels", "hatespeech", "hate_speech_score") if x in columns), None)
        if text_col is None or label_col is None:
            raise RuntimeError(f"{spec.key}: text/label 컬럼 확인 실패: {split.column_names}")
        for i, row in enumerate(split):
            text = row.get(text_col)
            raw_label = row.get(label_col)
            if text is None or raw_label is None:
                continue
            if spec.key == "comment_kmhas_binary":
                label = _kmhas_binary(raw_label)
            elif spec.label_threshold is not None:
                try:
                    label = int(float(raw_label) >= spec.label_threshold)
                except (TypeError, ValueError):
                    continue
            else:
                label = int(raw_label)
            if not 0 <= label < spec.num_labels:
                continue
            texts.append(str(text)); labels.append(label); ids.append(f"{spec.key}:{split_name}:{i}")
        standardized[split_name] = Dataset.from_dict({"sample_id": ids, "text": texts, "labels": labels})
    return DatasetDict(standardized)


def _split_indices(labels, test_size, seed):
    idx = np.arange(len(labels))
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=np.asarray(labels))
    except ValueError:
        return train_test_split(idx, test_size=test_size, random_state=seed)


def _ensure_three_splits(ds: DatasetDict, seed=42):
    if "all" in ds:
        train_idx, hold_idx = _split_indices(ds["all"]["labels"], 0.2, seed)
        hold = ds["all"].select(sorted(hold_idx.tolist()))
        val_idx, test_idx = _split_indices(hold["labels"], 0.5, seed)
        return DatasetDict(train=ds["all"].select(sorted(train_idx.tolist())), validation=hold.select(sorted(val_idx.tolist())), test=hold.select(sorted(test_idx.tolist())))
    if all(x in ds for x in ("train", "validation", "test")):
        return DatasetDict({x: ds[x] for x in ("train", "validation", "test")})
    if "test" in ds:
        train_idx, val_idx = _split_indices(ds["train"]["labels"], 0.1, seed)
        return DatasetDict(train=ds["train"].select(sorted(train_idx.tolist())), validation=ds["train"].select(sorted(val_idx.tolist())), test=ds["test"])
    if "validation" in ds:
        train_idx, val_idx = _split_indices(ds["train"]["labels"], 0.1, seed)
        return DatasetDict(train=ds["train"].select(sorted(train_idx.tolist())), validation=ds["train"].select(sorted(val_idx.tolist())), test=ds["validation"])
    train_idx, hold_idx = _split_indices(ds["train"]["labels"], 0.2, seed)
    hold = ds["train"].select(sorted(hold_idx.tolist()))
    val_idx, test_idx = _split_indices(hold["labels"], 0.5, seed)
    return DatasetDict(train=ds["train"].select(sorted(train_idx.tolist())), validation=hold.select(sorted(val_idx.tolist())), test=hold.select(sorted(test_idx.tolist())))


def _balanced_limit(ds: Dataset, limit: int | None, seed=42):
    if not limit or len(ds) <= limit:
        return ds
    labels = np.asarray(ds["labels"]); idx = np.arange(len(ds))
    chosen, _ = train_test_split(idx, train_size=limit, random_state=seed, stratify=labels)
    return ds.select(sorted(chosen.tolist()))


def load_task(task_key, run_mode, limits=None):
    if run_mode == "SMOKE":
        cache_tag = "smoke"
    elif limits:
        cache_tag = "paper_" + "_".join(f"{k}{int(v)}" for k, v in sorted(limits.items()))
    else:
        cache_tag = "paper_full"
    cache = DATA_CACHE_ROOT / task_key / cache_tag
    cache_complete = (cache / "dataset_dict.json").exists() and all(
        (cache / split / "state.json").exists() and (cache / split / "dataset_info.json").exists()
        for split in ("train", "validation", "test")
    )
    if cache_complete:
        return load_from_disk(str(cache))
    if cache.exists():
        broken = cache.with_name(cache.name + ".incomplete_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        cache.rename(broken)
    spec = TASKS[task_key]
    ds = _ensure_three_splits(_standardize(_raw_dataset(spec), spec))
    cfg = load_config()
    if run_mode == "SMOKE":
        limits = cfg["smoke_limits"]
    if limits:
        ds = DatasetDict({name: _balanced_limit(split, int(limits[name]), 42) for name, split in ds.items()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp_cache = cache.with_name(cache.name + ".building")
    if temp_cache.exists():
        shutil.rmtree(temp_cache)
    ds.save_to_disk(str(temp_cache))
    temp_cache.rename(cache)
    atomic_json(cache.parent / f"{cache_tag}_manifest.json", {
        "task": task_key, "run_mode": run_mode, "source": spec.path, "subset": spec.subset,
        "limits": limits, "rows": {k: len(v) for k, v in ds.items()}, "split_seed": 42,
        "fingerprints": {k: getattr(v, "_fingerprint", "UNKNOWN") for k, v in ds.items()},
    })
    return ds


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        return hidden_states + self.up(self.dropout(self.activation(self.down(hidden_states))))


class OutputWithAdapter(nn.Module):
    def __init__(self, original, hidden_size, bottleneck, dropout):
        super().__init__()
        self.dense = original.dense
        self.LayerNorm = original.LayerNorm
        self.dropout = original.dropout
        self.adapter = BottleneckAdapter(hidden_size, bottleneck, dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


def _encoder_layers(model):
    for attr in ("roberta", "bert", "deberta", "electra"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            return base.encoder.layer
    raise RuntimeError(f"Adapter 미지원 모델 구조: {model.__class__.__name__}")


def _unfreeze_head(model):
    for name, param in model.named_parameters():
        if any(token in name for token in ("classifier", "score")):
            param.requires_grad = True


def build_model(model_name, num_labels, method, cfg, revision=None):
    pretrained_kwargs = {"revision": revision} if revision else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
        **pretrained_kwargs,
    )
    if method == "full_ft":
        return model
    if method == "bitfit":
        for param in model.parameters(): param.requires_grad = False
        for name, param in model.named_parameters():
            if name.endswith(".bias"): param.requires_grad = True
        _unfreeze_head(model)
        return model
    if method == "adapter":
        for param in model.parameters(): param.requires_grad = False
        acfg = cfg["adapter"]
        for layer in _encoder_layers(model):
            layer.output = OutputWithAdapter(layer.output, model.config.hidden_size, acfg["bottleneck"], acfg["dropout"])
        _unfreeze_head(model)
        return model
    from peft import IA3Config, LoraConfig, TaskType, get_peft_model
    if method == "lora":
        lcfg = cfg["lora"]
        peft_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"], target_modules=["query", "value"], modules_to_save=["classifier"], bias="none")
    elif method == "ia3":
        peft_cfg = IA3Config(task_type=TaskType.SEQ_CLS, target_modules=["key", "value", "intermediate.dense"], feedforward_modules=["intermediate.dense"], modules_to_save=["classifier"])
    else:
        raise ValueError(method)
    return get_peft_model(model, peft_cfg)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_params": trainable, "total_params": total, "trainable_parameter_ratio": trainable / total}


def compute_metrics(prediction):
    labels = prediction.label_ids
    preds = np.argmax(prediction.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1, "macro_precision": precision, "macro_recall": recall}


class AtomicEpochCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            try:
                self.rows = pd.read_csv(self.path).to_dict("records")
            except Exception:
                self.rows = []
        else:
            self.rows = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        pd.DataFrame(self.rows).to_csv(temp, index=False, encoding="utf-8-sig")
        os.replace(temp, self.path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.rows.append({"time": now_iso(), "event": "log", "epoch": state.epoch, "global_step": state.global_step, **logs})
            self._save()

    def on_save(self, args, state, control, **kwargs):
        self.rows.append({"time": now_iso(), "event": "checkpoint", "epoch": state.epoch, "global_step": state.global_step})
        self._save()


def training_args(run_dir, method, seed, epochs, run_mode, cfg):
    return _training_args(run_dir, method, seed, epochs, run_mode, cfg, overrides=None)


def _training_args(run_dir, method, seed, epochs, run_mode, cfg, overrides=None):
    overrides = overrides or {}
    requested_precision = overrides.get("precision", cfg["precision"])
    use_fp16 = requested_precision == "fp16" and torch.cuda.is_available()
    kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=overrides.get("learning_rate", cfg["learning_rates"][method]),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=1 if run_mode == "SMOKE" else overrides.get("epochs", epochs),
        weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
        logging_strategy="steps", logging_steps=20,
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, report_to="none",
        seed=seed, data_seed=42, fp16=use_fp16,
        dataloader_num_workers=cfg["dataloader_num_workers"],
        optim=cfg.get("optimizer", "adamw_torch"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
    )
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def _class_weights(labels, num_labels, strategy):
    if strategy in (None, "none"):
        return None
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_labels).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"class weighting requires every class in train split; counts={counts.tolist()}")
    if strategy == "inverse_frequency":
        weights = counts.sum() / (num_labels * counts)
    elif strategy == "inverse_sqrt_frequency":
        weights = 1.0 / np.sqrt(counts)
        weights /= weights.mean()
    else:
        raise ValueError(f"unknown class-weighting strategy: {strategy}")
    return weights.astype(np.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=self.class_weights.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def _latest_checkpoint(run_dir):
    root = run_dir / "checkpoints"
    candidates = sorted(
        (p for p in root.glob("checkpoint-*") if (p / "trainer_state.json").exists()),
        key=lambda p: int(p.name.split("-")[-1]),
    ) if root.exists() else []
    return str(candidates[-1]) if candidates else None


def run_one(
    study,
    task_key,
    model_name,
    method,
    seed,
    run_mode,
    epochs,
    limits=None,
    *,
    experiment_id=None,
    variant=None,
    training_overrides=None,
    class_weighting="none",
    keep_checkpoint=None,
    force=False,
):
    cfg = load_config()
    if method not in METHODS: raise ValueError(method)
    model_slug = model_name.replace("/", "__")
    if experiment_id:
        run_dir = ROOT / "results" / "followup" / experiment_id / (variant or "default") / task_key / model_slug / method / f"seed_{seed}"
    else:
        run_dir = ROOT / "results" / study / run_mode / task_key / model_slug / method / f"seed_{seed}"
    signature_payload = {
        "study": study,
        "task": task_key,
        "model": model_name,
        "method": method,
        "seed": seed,
        "run_mode": run_mode,
        "epochs": epochs,
        "limits": limits,
        "experiment_id": experiment_id,
        "variant": variant,
        "training_overrides": training_overrides or {},
        "class_weighting": class_weighting,
        "keep_checkpoint": keep_checkpoint,
        "protocol": cfg,
        "suite_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("status") == "COMPLETE" and not force:
                require_matching_run_signature(existing, run_signature, metrics_path)
                return existing
        except (json.JSONDecodeError, OSError):
            pass
    if force and run_dir.exists():
        archived = run_dir.with_name(run_dir.name + ".superseded_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.rename(archived)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_at_start = _latest_checkpoint(run_dir)
    previous_status = {}
    if (run_dir / "status.json").exists():
        try: previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        except Exception: previous_status = {}
    if checkpoint_at_start:
        require_matching_run_signature(
            previous_status,
            run_signature,
            run_dir / "status.json",
        )
    status = {
        "status": "RUNNING", "started_at": previous_status.get("started_at", now_iso()),
        "resumed_at": now_iso() if checkpoint_at_start else None,
        "resume_count": int(previous_status.get("resume_count", 0)) + (1 if checkpoint_at_start else 0),
        "resumed_from_checkpoint": checkpoint_at_start,
        "study": study, "task": task_key, "model": model_name, "method": method,
        "seed": seed, "run_mode": run_mode, "experiment_id": experiment_id,
        "variant": variant, "run_signature": run_signature,
    }
    atomic_json(run_dir / "status.json", status)
    append_event(run_dir / "events.jsonl", {"event": "RUN_STARTED", **status})
    try:
        set_seed(seed)
        ds = load_task(task_key, run_mode, limits)
        spec = TASKS[task_key]
        revision = cfg.get("model_revisions", {}).get(model_name)
        pretrained_kwargs = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False if "bertweet" in model_name.lower() else True,
            **pretrained_kwargs,
        )
        def tokenize(batch): return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])
        tokenized = ds.map(tokenize, batched=True, remove_columns=["sample_id", "text"])
        model = build_model(model_name, spec.num_labels, method, cfg, revision=revision)
        counts = parameter_counts(model)
        class_weights = _class_weights(ds["train"]["labels"], spec.num_labels, class_weighting)
        atomic_json(run_dir / "run_config.json", {
            **status, "epochs": epochs, "hyperparameters": cfg, "runtime": runtime_info(),
            "model_commit": getattr(model.config, "_commit_hash", None), **counts,
            "training_overrides": training_overrides or {}, "class_weighting": class_weighting,
            "class_weights": class_weights.tolist() if class_weights is not None else None,
            "train_label_counts": np.bincount(np.asarray(ds["train"]["labels"]), minlength=spec.num_labels).astype(int).tolist(),
        })
        callback = AtomicEpochCallback(run_dir / "epoch_metrics.csv")
        trainer_kwargs = dict(
            model=model, args=_training_args(run_dir, method, seed, epochs, run_mode, cfg, training_overrides),
            train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
            data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"]), callback],
        )
        if "processing_class" in inspect.signature(Trainer.__init__).parameters: trainer_kwargs["processing_class"] = tokenizer
        else: trainer_kwargs["tokenizer"] = tokenizer
        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs) if class_weights is not None else Trainer(**trainer_kwargs)
        trainer_device = str(trainer.args.device)
        checkpoint = _latest_checkpoint(run_dir)
        started = time.perf_counter()
        trainer.train(resume_from_checkpoint=checkpoint)
        train_seconds = time.perf_counter() - started
        test = trainer.predict(tokenized["test"], metric_key_prefix="test")
        predictions = np.argmax(test.predictions, axis=-1)
        prediction_profile = prediction_diagnostics(test.label_ids, predictions, spec.num_labels)
        pd.DataFrame({"sample_id": ds["test"]["sample_id"], "label": test.label_ids, "prediction": predictions}).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        history = pd.DataFrame(trainer.state.log_history)
        history.to_csv(run_dir / "trainer_history.csv", index=False, encoding="utf-8-sig")
        result = {
            "status": "COMPLETE", "completed_at": now_iso(), "study": study, "task": task_key,
            "model": model_name, "method": method, "seed": seed, "run_mode": run_mode,
            "experiment_id": experiment_id, "variant": variant, "run_signature": run_signature,
            "train_rows": len(ds["train"]), "validation_rows": len(ds["validation"]), "test_rows": len(ds["test"]),
            "epochs_requested": 1 if run_mode == "SMOKE" else (training_overrides or {}).get("epochs", epochs),
            "epochs_completed": float(trainer.state.epoch or 0.0),
            "global_step": int(trainer.state.global_step),
            "best_validation_macro_f1": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
            "learning_rate": (training_overrides or {}).get("learning_rate", cfg["learning_rates"][method]),
            "class_weighting": class_weighting,
            "trainer_device": trainer_device,
            "train_seconds": train_seconds, **counts, **test.metrics, "runtime": runtime_info(),
            "best_checkpoint": (
                Path(trainer.state.best_model_checkpoint).name
                if trainer.state.best_model_checkpoint else None
            ),
            **prediction_profile,
        }
        atomic_json(metrics_path, result)
        atomic_json(run_dir / "status.json", result)
        append_event(run_dir / "events.jsonl", {
            "event": "RUN_COMPLETE", "test_macro_f1": result.get("test_macro_f1"),
            "constant_prediction_collapse": result["constant_prediction_collapse"],
        })
        should_keep_checkpoint = cfg["keep_best_checkpoint"] if keep_checkpoint is None else keep_checkpoint
        if not should_keep_checkpoint:
            shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        del trainer, model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result
    except Exception as exc:
        failed = {**status, "status": "FAILED", "failed_at": now_iso(), "error_type": type(exc).__name__, "error": str(exc)}
        atomic_json(run_dir / "status.json", failed)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        append_event(run_dir / "events.jsonl", {"event": "RUN_FAILED", "error": str(exc)})
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        raise


def build_jobs(study):
    cfg = load_config(); section = cfg[study]
    mode_limits = section.get("limits")
    jobs = []
    for task in section["tasks"]:
        for model in section["models"]:
            for method in cfg["methods"]:
                for seed in cfg["seeds"]:
                    jobs.append({"study": study, "task_key": task, "model_name": model, "method": method, "seed": seed, "epochs": section["epochs"], "limits": mode_limits})
    return jobs


def run_study(study, run_mode="SMOKE", max_jobs=None, continue_on_error=None):
    if run_mode not in {"SMOKE", "PAPER"}: raise ValueError("run_mode은 SMOKE 또는 PAPER만 가능합니다.")
    precheck(require_cuda=True)
    cfg = load_config(); jobs = build_jobs(study)
    if max_jobs is not None: jobs = jobs[:max_jobs]
    continue_on_error = cfg["continue_on_error"] if continue_on_error is None else continue_on_error
    progress_path = ROOT / "results" / study / run_mode / "progress.json"
    events_path = ROOT / "results" / study / run_mode / "events.jsonl"
    completed, failed = 0, 0
    results = []
    for index, job in enumerate(jobs, 1):
        current = {"study": study, "run_mode": run_mode, "total": len(jobs), "index": index, "completed": completed, "failed": failed, "current": job, "updated_at": now_iso()}
        atomic_json(progress_path, current); append_event(events_path, {"event": "JOB_DISPATCH", "index": index, **job})
        try:
            result = run_one(run_mode=run_mode, **job)
            results.append(result); completed += 1
        except Exception:
            failed += 1
            if not continue_on_error:
                atomic_json(progress_path, {**current, "status": "STOPPED_ON_ERROR", "failed": failed, "updated_at": now_iso()})
                raise
        atomic_json(progress_path, {**current, "status": "RUNNING", "completed": completed, "failed": failed, "updated_at": now_iso()})
    final = {"study": study, "run_mode": run_mode, "status": "COMPLETE" if failed == 0 else "COMPLETE_WITH_ERRORS", "total": len(jobs), "completed": completed, "failed": failed, "updated_at": now_iso()}
    atomic_json(progress_path, final); append_event(events_path, {"event": "STUDY_FINISHED", **final})
    return pd.DataFrame(results)


def aggregate(run_mode="PAPER", *, strict=True):
    """Aggregate broad-benchmark runs after provenance and seed checks.

    ``strict=False`` permits an intentionally partial progress export. Final
    PAPER tables should always use the default strict validation.
    """
    if run_mode not in {"SMOKE", "PAPER"}:
        raise ValueError("run_mode은 SMOKE 또는 PAPER만 가능합니다.")
    cfg = load_config()
    rows = []
    for path in (ROOT / "results").glob(f"study*/{run_mode}/**/final_metrics.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        relative_parts = path.relative_to(ROOT / "results").parts
        if len(relative_parts) != 7:
            raise ValueError(f"unexpected result path layout: {path}")
        path_study, path_mode, path_task, path_model, path_method, path_seed, _ = relative_parts
        expected_path_identity = (
            str(row.get("study")),
            str(row.get("run_mode")),
            str(row.get("task")),
            str(row.get("model", "")).replace("/", "__"),
            str(row.get("method")),
            f"seed_{row.get('seed')}",
        )
        if expected_path_identity != (
            path_study,
            path_mode,
            path_task,
            path_model,
            path_method,
            path_seed,
        ):
            raise ValueError(f"result payload/path identity mismatch: {path}")
        row["source_file"] = str(path.relative_to(ROOT))
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected_keys = None
    if strict:
        expected_keys = {
            (study, task, model, method, int(seed))
            for study in ("study1", "study2", "study3")
            for task in cfg[study]["tasks"]
            for model in cfg[study]["models"]
            for method in cfg["methods"]
            for seed in cfg["seeds"]
        }
    validate_run_frame_integrity(
        frame,
        expected_seeds=cfg["seeds"] if strict else None,
        expected_run_mode=run_mode,
        expected_keys=expected_keys,
    )
    summary = summarize_run_frame(
        frame,
        expected_seeds=cfg["seeds"] if strict else None,
    )
    if strict and not bool(summary["seed_coverage_ok"].all()):
        raise ValueError("aggregate contains a group with invalid seed coverage")
    out = ROOT / "results" / "aggregate"; out.mkdir(parents=True, exist_ok=True)
    public_run_frame(frame).to_csv(
        out / f"all_runs_{run_mode.lower()}.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )
    summary.to_csv(
        out / f"summary_{run_mode.lower()}.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )
    return frame
