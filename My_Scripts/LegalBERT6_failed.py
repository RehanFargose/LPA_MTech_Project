#!/usr/bin/env python3
"""
Multi-task LegalBERT: Verdict (binary) + IPC (multi-label) training script
Optimized for GTX 1650 (4GB): small batch, max_length=256, AMP, safe defaults.

Based on user's uploaded script and sample row. See:
- original script: /mnt/data/LegalBERT5.py (uploaded). :contentReference[oaicite:2]{index=2}
- sample row: /mnt/data/sample_parquet_row.json (uploaded). :contentReference[oaicite:3]{index=3}
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight




# -------------------------
# Config / Hyperparameters
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = "nlpaueb/legal-bert-base-uncased"   # change if you prefer another LegalBERT
MODEL_NAME = "prajjwal1/bert-medium"
# MODEL_NAME = "prajjwal1/bert-small"

# change to your parquet. If not present, will try JSON fallback.
DATA_PATH = "D:/LPA_MTech_Project/Enriched_Datasets/SupremeCourt_Combined_1990_2009_enriched.parquet"        
MAX_LENGTH = 160            # safe for 4GB GPU
BATCH_SIZE = 2               # fits GTX 1650
VAL_BATCH_SIZE = 4
NUM_EPOCHS = 5
LR = 3e-5
WEIGHT_IPC = 0.5             # IPC loss weight relative to verdict loss
ACCUM_STEPS = 1              # keep =1 for simplicity on small GPU; you can bump to 2/4 if you want effective larger batch
OUTPUT_DIR = "D:/LPA_MTech_Project/My_Models/LegalBERT_Models/small-bert-3"
SEED = 42
ipc_loss_fn = None
NUM_IPC_LABELS = 0


os.makedirs(OUTPUT_DIR, exist_ok=True)




# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

set_seed()



# -------------------------
# Load dataset (parquet preferred)
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            print(f"Loaded parquet: {path} -> {len(df)} rows")
            return df
        except Exception as e:
            print("Failed to read parquet:", e)

    # fallback: try JSON sample file or a single-json-lines file
    json_fallback = "sample_parquet_row.json"
    if os.path.exists(json_fallback):
        print("Parquet not found. Loading JSON fallback:", json_fallback)
        with open(json_fallback, "r", encoding="utf-8") as f:
            row = json.load(f)
            df = pd.DataFrame([row])
            return df

    raise FileNotFoundError(f"No dataset found at {path} and no fallback JSON at {json_fallback}.")


df = load_data(DATA_PATH)

# Keep only the columns we need. The sample row used 'text' and 'verdict_label'.
# We also support 'ipc_sections' which should be a list (e.g. [420,302]) or a comma-separated string.
if "text" not in df.columns:
    raise KeyError("Dataset must contain a 'text' column. Auto-detection disabled to avoid incorrect columns like raw_html.")



# Normalize verdict label column names
if "verdict_label" in df.columns:

    # Force everything to string and clean
    df["verdict_label"] = df["verdict_label"].astype(str).str.strip().str.lower()

    # Map allowed/allowed/allowed → 1, dismiss/dismissed → 0
    df["verdict_label"] = df["verdict_label"].replace({
        "allow": 1, "allowed": 1, "allowed.": 1, "allowed,": 1,
        "dismiss": 0, "dismissed": 0, "dismissed.": 0, "dismissed,": 0,
    })

    # Convert numeric-like strings
    df["verdict_label"] = pd.to_numeric(df["verdict_label"], errors="coerce")

    # Drop invalids
    df = df.dropna(subset=["verdict_label"]).reset_index(drop=True)

    df["label"] = df["verdict_label"].astype(int)
elif "label" in df.columns:
    df["label"] = df["label"].astype(int)
else:
    raise KeyError("No verdict label column found: expected 'verdict_label' or 'label'.")



# Ensure labels are binary 0/1
unique_labels = sorted(df["label"].unique().tolist())
print("Unique verdict labels found:", unique_labels)
if not set(unique_labels).issubset({0, 1}):
    # Attempt to map common text labels
    mapping = {}
    for v in unique_labels:
        if isinstance(v, str):
            vlow = v.lower()
            if "dismiss" in vlow:
                mapping[v] = 0
            elif "allow" in vlow or "allowed" in vlow or "grant" in vlow:
                mapping[v] = 1
    if mapping:
        df["label"] = df["label"].map(lambda x: mapping.get(x, x)).astype(int)
        print("Mapped string labels -> numeric using heuristic mapping.")
    else:
        print("Warning: labels not binary 0/1. Found:", unique_labels)
        # try to force-binarize by taking > median as 1
        med = np.median(df["label"].astype(float))
        df["label"] = (df["label"].astype(float) > med).astype(int)
        print("Binarized labels by median split (unsafe).")


# Recompute class weights AFTER cleaning labels
# weights = compute_class_weight("balanced", classes=[0,1], y=df["label"])
# weights = compute_class_weight("balanced", classes=np.array([0,1]), y=df["label"])
weights = torch.tensor([1.0, 1.5], dtype=torch.float32).to(DEVICE)



# -------------------------
# Build IPC multi-label targets
# -------------------------
# Support multiple input formats:
# - 'ipc_sections' column containing list of ints or list of strings
# - 'ipc' column (comma-separated strings)
# ---- FORCE IPC DISABLED ----
NUM_IPC_LABELS = 0
ipc_loss_fn = None

df["ipc_list"] = [[] for _ in range(len(df))]
df["ipc_multi"] = [[] for _ in range(len(df))]

mlb = MultiLabelBinarizer()
mlb.fit([[]])

# ipc_col = None
# for candidate in ["ipc_sections", "ipc", "ipc_mentions", "ipc_mentions_extra"]:
#     if candidate in df.columns:
#         ipc_col = candidate
#         break

# if ipc_col is None:
#     # If IPC not present, create an empty list per row (training will skip IPC loss effectively)
#     print("No IPC column found. Creating empty IPC lists (model will learn with verdict-only signal).")
#     df["ipc_list"] = [[] for _ in range(len(df))]
# else:
#     def normalize_ipc_cell(v):
#         # Case 1: missing / null
#         if v is None:
#             return []

#         # Case 2: numpy array
#         if isinstance(v, np.ndarray):
#             return [str(x).strip() for x in v.tolist() if x is not None and str(x).strip() != ""]

#         # Case 3: list
#         if isinstance(v, list):
#             return [str(x).strip() for x in v if x is not None and str(x).strip() != ""]

#         # Case 4: string (comma separated or empty)
#         if isinstance(v, str):
#             if v.strip() == "":
#                 return []
#             return [p.strip() for p in v.split(",") if p.strip() != ""]

#         # Case 5: everything else (int, float, etc.)
#         v = str(v).strip()
#         return [] if v == "" else [v]

#     df["ipc_list"] = df[ipc_col].apply(normalize_ipc_cell)



# # Build MultiLabelBinarizer on IPC tokens (strings)
# mlb = MultiLabelBinarizer(sparse_output=False)
# try:
#     ipc_matrix = mlb.fit_transform(df["ipc_list"])
#     NUM_IPC_LABELS = ipc_matrix.shape[1]
#     print("NUM_IPC_LABELS:", NUM_IPC_LABELS)
# except Exception as e:
#     print("Error building IPC matrix:", e)
#     # fallback: no IPC labels
#     df["ipc_list"] = [[] for _ in range(len(df))]
#     mlb = MultiLabelBinarizer()
#     mlb.fit([[]])
#     NUM_IPC_LABELS = 0
#     ipc_matrix = mlb.transform(df["ipc_list"])

# # attach ipc multi vectors into df
# df["ipc_multi"] = ipc_matrix.tolist()







# -------------------------
# Tokenizer, Dataset, DataLoader
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].astype(str).tolist()
        self.verdicts = df["label"].tolist()
        self.ipcs = df["ipc_multi"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        verdict = torch.tensor(self.verdicts[idx], dtype=torch.long)
        ipc = torch.tensor(self.ipcs[idx], dtype=torch.float32)
        return input_ids, attention_mask, verdict, ipc

# Split train/val (stratify by verdict label)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])

train_dataset = MultiTaskDataset(train_df.reset_index(drop=True))
val_dataset = MultiTaskDataset(val_df.reset_index(drop=True))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# -------------------------
# Model Definition
# -------------------------
class MultiTaskLegalBERT(nn.Module):
    def __init__(self, model_name: str, num_verdict_labels: int = 2, num_ipc_labels: int = 0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # verdict head (classification)
        self.verdict_head = nn.Linear(hidden_size, num_verdict_labels)

        # IPC head (multi-label)
        self.ipc_head = nn.Linear(hidden_size, num_ipc_labels) if num_ipc_labels > 0 else None

        # optional dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Mean Pooling
        token_embeddings = out.last_hidden_state                          # (B, L, H)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        pooled = torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        pooled = self.dropout(pooled)

        verdict_logits = self.verdict_head(pooled)
        ipc_logits = self.ipc_head(pooled) if self.ipc_head is not None else None

        return verdict_logits, ipc_logits



# model = MultiTaskLegalBERT(MODEL_NAME, num_verdict_labels=2, num_ipc_labels=NUM_IPC_LABELS).to(DEVICE)
model = MultiTaskLegalBERT(MODEL_NAME, num_verdict_labels=2, num_ipc_labels=0).to(DEVICE)
print("Model loaded. Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()



# -------------------------
# Losses, optimizer, scheduler
# -------------------------
# Verdict class weights from train set to handle imbalance
train_counts = train_df["label"].value_counts().sort_index().to_numpy(dtype=np.float32)
train_counts = np.maximum(train_counts, 1.0)
inv_freq = 1.0 / train_counts
# verdict_weights = torch.tensor(inv_freq, dtype=torch.float32).to(DEVICE)
# verdict_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE))
verdict_loss_fn = nn.CrossEntropyLoss(weight=weights)


# IPC multi-label loss
# ipc_loss_fn = nn.BCEWithLogitsLoss() if NUM_IPC_LABELS > 0 else None

# optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = max(1, int(0.06 * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))



# -------------------------
# Training / Eval helpers
# -------------------------

# def compute_loss(verdict_logits, ipc_logits, verdict_labels, ipc_labels):
#     # verdict_labels: LongTensor (B,)
#     loss_v = verdict_loss_fn(verdict_logits, verdict_labels)
#     if ipc_loss_fn is not None and ipc_logits is not None:
#         loss_ipc = ipc_loss_fn(ipc_logits, ipc_labels)
#         return loss_v + WEIGHT_IPC * loss_ipc, loss_v.detach().item(), loss_ipc.detach().item()
#     else:
#         return loss_v, loss_v.detach().item(), 0.0


def compute_loss(verdict_logits, _, verdict_labels, _2):
    loss_v = verdict_loss_fn(verdict_logits, verdict_labels)
    return loss_v, loss_v.detach().item(), 0.0




def evaluate(model, loader):
    model.eval()
    all_preds_v = []
    all_trues_v = []
    losses = []
    ipc_preds = []
    ipc_trues = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, verdicts, ipcs = [b.to(DEVICE) for b in batch]
            # with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
                v_logits, i_logits = model(input_ids, attention_mask)
                loss, loss_v, loss_ipc = compute_loss(v_logits, i_logits, verdicts, ipcs)
            losses.append(loss.item() if isinstance(loss, torch.Tensor) else float(loss))
            preds = torch.argmax(v_logits, dim=-1).cpu().numpy().tolist()
            all_preds_v.extend(preds)
            all_trues_v.extend(verdicts.cpu().numpy().tolist())
            if i_logits is not None:
                ipc_preds.extend(torch.sigmoid(i_logits).cpu().numpy().tolist())
                ipc_trues.extend(ipcs.cpu().numpy().tolist())

    acc = accuracy_score(all_trues_v, all_preds_v) if len(all_trues_v)>0 else 0.0
    f1 = f1_score(all_trues_v, all_preds_v, average="weighted") if len(all_trues_v)>0 else 0.0
    return {"loss": float(np.mean(losses)) if losses else 0.0, "acc": acc, "f1": f1, "ipc_preds": ipc_preds, "ipc_trues": ipc_trues}

# -------------------------
# Training loop
# -------------------------
best_val_f1 = -1.0
save_every = 1
early_stop_patience = 2
no_improve = 0


for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_v_loss = 0.0
    total_ipc_loss = 0.0
    step = 0

    # for batch in train_loader:
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        input_ids, attention_mask, verdicts, ipcs = [b.to(DEVICE) for b in batch]

        # with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            v_logits, i_logits = model(input_ids, attention_mask)
            loss, loss_v, loss_ipc = compute_loss(v_logits, i_logits, verdicts, ipcs)

            # normalize for accumulation if used
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()


        total_loss += float(loss.item() * ACCUM_STEPS)
        total_v_loss += float(loss_v)
        if NUM_IPC_LABELS > 0:
            total_ipc_loss += float(loss_ipc)

        # <<< INSERT THIS HERE >>>
        if step % 100 == 0:
            tqdm.write(f"[Epoch {epoch} | Step {step}] Loss: {loss_v:.4f}")
        # <<< END INSERT >>>

        step += 1

    # End epoch eval
    val_metrics = evaluate(model, val_loader)
    avg_train_loss = total_loss / max(1, step)
    avg_train_vloss = total_v_loss / max(1, step)
    avg_train_ipc = total_ipc_loss / max(1, step)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} (v:{avg_train_vloss:.4f} ipc:{avg_train_ipc:.4f}) | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    # save
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mlb_classes": mlb.classes_.tolist(),
            "tokenizer_name": MODEL_NAME
        }, ckpt_path)
        print("Saved best model to", ckpt_path)

    
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= early_stop_patience:
            print("Early stopping triggered")
            break


print("Training complete. Best val F1:", best_val_f1)
