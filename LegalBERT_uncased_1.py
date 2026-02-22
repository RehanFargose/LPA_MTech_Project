#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import math
import time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F



# In[2]:


# ---------------------------
# CONFIG (tweak if needed)
# ---------------------------
start = int(input("Enter Start Year: "))
end = int(input("Enter Ending Year: "))
model_num = str(input("Enter a Model number to save: "))

DATA_PATH = f"D:/LPA_MTech_Project/Enriched_Datasets/SupremeCourt_Combined_{start}_{end}_enriched.parquet"
# MODEL_NAME = "D:/LPA_MTech_Project/My_Models/DAPT/dapt_medium_bert/final"

MODEL_NAME = "D:/LPA_MTech_Project/My_Models/DAPT/dapt_legalbert_uncased1/final"
# D:/LPA_MTech_Project/My_Models/DAPT/dapt_legalbert_uncased1/final

OUTPUT_DIR = f"My_Models/LegalBERT_Models/legalbert_uncased_{start}-{end}_medium_{model_num}"
SEED = 42


# In[3]:


# Preprocessing / tokenization
MAX_WORDS = 2000        # keep first ~350 words (~512 tokens or fewer)
MAX_LENGTH = 384       # tokenizer max tokens
BATCH_SIZE = 2         
ACCUM_STEPS = 16   # effective batch = 32
# Max Epochs should be 5 for a combined dataset on local device
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
alpha = 0.25   # weight minority class more
gamma = 2.0    # focus on hard examples


# In[4]:


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# In[5]:


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary/multi-class classification.
    alpha controls class weight.
    gamma controls focus on hard examples.
    """
    ce_loss = F.cross_entropy(logits, labels, reduction='none') 
    pt = torch.exp(-ce_loss)              # probability of correct class
    focal_loss_value = alpha * (1 - pt)**gamma * ce_loss
    return focal_loss_value.mean()


# In[6]:


# ---------------------------
# Utilities / Seed
# ---------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



# In[7]:


# ---------------------------
# 1 Load & preprocess (PANDAS) BEFORE tokenization
# ---------------------------
print("Loading parquet...")
df = pd.read_parquet(DATA_PATH)

# Keep required columns and drop NaNs
df = df[["text", "verdict_label"]].dropna()

# rename label
df = df.rename(columns={"verdict_label": "label"})
df["label"] = df["label"].astype(int)

# simple cleaning function
def clean_text(t):
    s = str(t)
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s

print("Cleaning text (may take a moment)...")
df["text"] = df["text"].map(clean_text)

# Keep only first MAX_WORDS words (very important for speed)
def first_n_words(s, n=MAX_WORDS):
    words = s.split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n])

df["text"] = df["text"].map(lambda s: first_n_words(s, MAX_WORDS))

# Reset index
df.reset_index(drop=True, inplace=True)

print("Dataset size:", len(df))
print(df.head(2))


# In[8]:


# ---------------------------
# 2 Tokenize ALL at once (fast)
# ---------------------------
print("Loading tokenizer & tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Tokenize in batches to avoid memory spikes
texts = df["text"].tolist()
labels = df["label"].to_numpy(dtype=np.int64)

# Use tokenizer.batch_encode_plus (returns lists / tensors)
def encode_head_tail(texts, tokenizer, max_length=256, head_ratio=0.5):
    input_ids_list = []
    attention_masks = []

    head_len = int(max_length * head_ratio)
    tail_len = max_length - head_len

    for t in texts:
        tokens = tokenizer.encode(t, add_special_tokens=False)

        if len(tokens) <= max_length:
            # pad normally
            ids = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        else:
            # head + tail split
            head_tokens = tokens[:head_len]
            tail_tokens = tokens[-tail_len:]
            ids = head_tokens + tail_tokens

        mask = [1 if id != tokenizer.pad_token_id else 0 for id in ids]

        input_ids_list.append(ids)
        attention_masks.append(mask)

    return torch.tensor(input_ids_list), torch.tensor(attention_masks)

print("Running head+tail tokenization...")

# input_ids = encodings["input_ids"]
input_ids, attention_mask = encode_head_tail(
    texts,
    tokenizer,
    max_length=MAX_LENGTH,
    head_ratio=0.5   # 128 head, 128 tail
)
labels_t = torch.tensor(labels, dtype=torch.long)


print("Tokenized shape:", input_ids.shape)


# In[9]:


# ---------------------------
# 3 Build TensorDataset and DataLoaders
# ---------------------------
dataset = TensorDataset(input_ids, attention_mask, labels_t)

# Train/test split (85/15)
n = len(dataset)
train_n = int(0.85 * n)
test_n = n - train_n
train_ds, test_ds = random_split(dataset, [train_n, test_n], generator=torch.Generator().manual_seed(SEED))

print(f"Train size: {len(train_ds)}  Test size: {len(test_ds)}")

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
# test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,                # CPU workers
    pin_memory=True,
    persistent_workers=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)



# In[10]:


# ---------------------------
# 4 Model, optimizer, amp scaler
# ---------------------------
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# # Compile model for speed (works on PyTorch 2.x)
# try:
#     model = torch.compile(model)
#     print("Model compiled successfully with torch.compile()")
# except Exception as e:
#     print("torch.compile() not supported:", e)

    
# optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# Simple LR scheduler (optional)
total_steps = len(train_loader) * EPOCHS
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)  # or None
scheduler = None    # removes per-step overhead, simpler + faster



# In[11]:


# ---------------------------
# 5 Training loop (fast)
# ---------------------------
def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids_b, att_mask_b, labels_b = [b.to(DEVICE) for b in batch]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids_b, attention_mask=att_mask_b)
                logits = outputs.logits
                loss = focal_loss(logits, labels_b)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist())
            trues.extend(labels_b.detach().cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    return {"loss": np.mean(losses), "accuracy": acc, "f1": f1, "preds": preds, "labels": trues}

print("Starting training... (epochs: {}, batch_size: {})".format(EPOCHS, BATCH_SIZE))

best_val_f1 = -1.0
global_step = 0
start_time = time.time()


# In[12]:


for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids_b, att_mask_b, labels_b = [b.to(DEVICE) for b in batch]

       # Zero grad ONLY at the start of each accumulation cycle
        if step % ACCUM_STEPS == 0:
            optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids_b, attention_mask=att_mask_b)
            logits = outputs.logits
            # Apply focal loss (scaled for gradient accumulation)
            loss = focal_loss(logits, labels_b) / ACCUM_STEPS


        # backward on scaled loss
        scaler.scale(loss).backward()

        # Update weights only every ACCUM_STEPS steps
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item() * ACCUM_STEPS   # multiply back for correct logging
        global_step += 1

        pbar.set_postfix({"loss": f"{epoch_loss/(step+1):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    # End of epoch: evaluate on test set
    val = evaluate(model, test_loader)
    epoch_time = time.time() - start_time
    print(f"\nEpoch {epoch+1} finished. Train loss: {epoch_loss/len(train_loader):.4f}  Val loss: {val['loss']:.4f}  Acc: {val['accuracy']:.4f}  F1: {val['f1']:.4f}  Time:{epoch_time:.1f}s\n")

    # Save best model
    if val["f1"] > best_val_f1:
        best_val_f1 = val["f1"]
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        print(f"Saved best model (F1 {best_val_f1:.4f}) to {os.path.join(OUTPUT_DIR,'best_model')}")




# In[13]:


torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scaler_state": scaler.state_dict(),
    "best_val_f1": best_val_f1
}, os.path.join(OUTPUT_DIR, "checkpoint_epoch6.pt"))


# In[14]:


# Final evaluation
final = evaluate(model, test_loader)
print("Final evaluation on test set:")
print(f"Loss: {final['loss']:.4f}  Acc: {final['accuracy']:.4f}  F1: {final['f1']:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(final["labels"], final["preds"]))


# In[15]:


# Save final
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print("Saved final model to", os.path.join(OUTPUT_DIR, "final_model"))


# In[ ]:





# In[16]:


# jupyter nbconvert --to python LegalBERT6.3_DAPT.ipynb


# In[ ]:




