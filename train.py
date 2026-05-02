#    Copyright 2026 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer,
    TrainerCallback
)

from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =========================
# Paths (for reproducibility)
# =========================
DATA_PATH = "./data/fusion_data_sequences_train.jsonl"
MODEL_PATH = "./pretrained_model/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./outputs"
CM_DIR = "./outputs/confusion_matrix"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# =========================
# Tokenization function
# =========================
def process_func(example):
    MAX_LENGTH = 384
    label_map = {"complete response": 1, "non-complete response": 0}
    label = label_map[example['label']]

    # 构造 instruction 文本
    instruction_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: August 2025\nToday Date: 5 Sep 2025\n"
        "You are now playing the role of a nuclear medicine physician, who is an expert in radiomics.<|eot_id|>\n\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + example['instruction'] + example['input'] +
        "<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    tokenized = tokenizer(
        instruction_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
        "labels": torch.tensor(label, dtype=torch.float)
    }

# =========================
# Custom Trainer
# =========================
class MyTrainer(Trainer):
    """Trainer with BCE loss for binary classification."""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if logits.ndim == 2:
        logits = logits.squeeze(-1)

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

# =========================
# Callback: metrics + CM
# =========================
class MetricsCallback(TrainerCallback):
    def __init__(self, fold):
        self.fold = fold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = round(state.epoch)

        print(f"\n[Fold {self.fold} | Epoch {epoch}]")
        print(f"AUC: {metrics['eval_auc']:.4f}")

        cm = metrics["eval_confusion_matrix"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Fold {self.fold} - Epoch {epoch}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.savefig(f"{CM_DIR}/fold_{self.fold}_epoch_{epoch}.png")
        plt.close()

# =========================
# Load tokenizer & dataset
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_json(DATA_PATH, lines=True)
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

# =========================
# Cross validation setup
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
labels = df["label"].map({"complete response": 1, "non-complete response": 0}).values

# =========================
# LoRA configuration
# =========================
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False
)

# =========================
# Training loop
# =========================
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), labels)):

    print(f"\n========== Fold {fold + 1} ==========")

    train_data = tokenized_dataset.select(train_idx)
    val_data = tokenized_dataset.select(val_idx)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    # Training arguments
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/fold_{fold+1}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=100,
        learning_rate=1e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch"
    )

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback(fold+1)]
    )

    trainer.train()
