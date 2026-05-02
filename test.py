#    Copyright 2026 Hangyu, Wang
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

# Copyright 2026 Hangyu Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.

import torch
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# =========================
# Paths (for reproducibility)
# =========================
BASE_MODEL_PATH = "./pretrained_model/Meta-Llama-3-8B-Instruct"
CHECKPOINT_PATH = "./checkpoints/best_lora"

TEST_DATA_PATH = "./data/fusion_data_sequences_test.jsonl"

# =========================
# Load model & tokenizer
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# Data preprocessing
# =========================
def process_func(example):
    """Convert raw sample into model input format."""
    MAX_LENGTH = 384

    label_map = {
        "complete response": 1,
        "non-complete response": 0
    }

    label = label_map[example["label"]]

    text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: August 2025\nToday Date: 5 Sep 2025\n"
        "You are now playing the role of a nuclear medicine physician, who is an expert in radiomics.<|eot_id|>\n\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + example['instruction'] + example['input'] +
        "<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    tokenized = tokenizer(
        text,
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
# Load test dataset
# =========================
df_test = pd.read_json(TEST_DATA_PATH, lines=True)
dataset_test = Dataset.from_pandas(df_test)
tokenized_test = dataset_test.map(process_func, remove_columns=dataset_test.column_names)

# =========================
# DataLoader
# =========================
dataloader = torch.utils.data.DataLoader(
    tokenized_test,
    batch_size=1,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer)
)

# =========================
# Inference
# =========================
all_probs = []
all_labels = []

device = model.device

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits.squeeze(-1).detach().cpu().float()
        probs = torch.sigmoid(logits).numpy()

        all_probs.extend(probs)
        all_labels.extend(labels)

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# =========================
# Threshold (You should define best threshold from validation set)
# =========================
best_threshold = 0.5

all_preds = (all_probs >= best_threshold).astype(int)

# =========================
# Evaluation metrics
# =========================
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

# =========================
# Results
# =========================
print("\n=== Test Set Performance ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))
