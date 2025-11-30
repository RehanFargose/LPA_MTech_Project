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






# ---------------------------
# CONFIG (tweak if needed)
# ---------------------------
start = int(input("Enter Start Year: "))
end = int(input("Enter Ending Year: "))
DATA_PATH = f"D:/LPA_MTech_Project/Enriched_Datasets/SupremeCourt_Combined_{start}_{end}_enriched.parquet"
# MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MODEL_NAME = "prajjwal1/bert-small"
OUTPUT_DIR = f"My_Models/legalbert_fast_out_{start}-{end}"
SEED = 42





# Preprocessing / tokenization
MAX_WORDS = 120          # keep first ~350 words (~512 tokens or fewer)
MAX_LENGTH = 128        # tokenizer max tokens
BATCH_SIZE = 4          # try 4; reduce to 2 if OOM
# Max Epochs should be 5 for a combined dataset on local device
# 3 works fine too
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01




# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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







# ---------------------------
# 2 Tokenize ALL at once (fast)
# ---------------------------
print("Loading tokenizer & tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Tokenize in batches to avoid memory spikes
texts = df["text"].tolist()
labels = df["label"].to_numpy(dtype=np.int64)

# Use tokenizer.batch_encode_plus (returns lists / tensors)
encodings = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels_t = torch.tensor(labels, dtype=torch.long)

print("Tokenized shape:", input_ids.shape)







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
                outputs = model(input_ids=input_ids_b, attention_mask=att_mask_b, labels=labels_b)
                loss = outputs.loss
                logits = outputs.logits
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









for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids_b, att_mask_b, labels_b = [b.to(DEVICE) for b in batch]

        optimizer.zero_grad()

        # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids_b, attention_mask=att_mask_b, labels=labels_b)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # if scheduler is not None:
        #     scheduler.step()

        epoch_loss += loss.item()
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





# Final evaluation
final = evaluate(model, test_loader)
print("Final evaluation on test set:")
print(f"Loss: {final['loss']:.4f}  Acc: {final['accuracy']:.4f}  F1: {final['f1']:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(final["labels"], final["preds"]))





# Save final
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print("Saved final model to", os.path.join(OUTPUT_DIR, "final_model"))
