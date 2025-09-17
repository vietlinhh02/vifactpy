from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import inspect

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


@dataclass
class RationaleConfig:
    model_name: str = "xlm-roberta-base"  # multilingual, works for Vietnamese
    max_length: int = 256


class PairDataset(Dataset):
    def __init__(self, tokenizer, pairs, max_length: int):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ex = self.pairs[idx]
        enc = self.tokenizer(
            ex.text_a,
            ex.text_b,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = int(ex.label)
        return enc


class RationaleClassifier:
    def __init__(self, cfg: RationaleConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    def get_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        output_dir: str,
        *,
        lr: float = 2e-5,
        epochs: int = 2,
        batch_size: int = 16,
        wd: float = 0.01,
        fp16: bool = True,
        eval_steps: int = 200,
        save_steps: int = 200,
        logging_steps: int = 50,
        seed: int = 42,
        num_workers: int = 2,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: Optional[bool] = None,
        compute_metrics: Optional[Callable] = None,
        early_stopping_patience: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> Trainer:
        eval_strategy = "steps" if eval_dataset is not None else "no"
        args_kwargs = dict(
            output_dir=output_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=wd,
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,
            seed=seed,
            dataloader_num_workers=num_workers,
        )
        init_params = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in init_params:
            args_kwargs["eval_strategy"] = eval_strategy
        elif "evaluation_strategy" in init_params:
            args_kwargs["evaluation_strategy"] = eval_strategy
        args = TrainingArguments(**args_kwargs)
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        if early_stopping_patience is not None and eval_dataset is not None:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        return trainer
