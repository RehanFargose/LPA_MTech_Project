#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import math


# In[2]:


# ============================
# Configs
# ============================
PARQUET_PATH = "D:\LPA_MTech_Project\Enriched_Datasets\SupremeCourt_Combined_1990_2025_enriched.parquet"      
TEXT_COLUMN = "text"
MODEL_NAME = "google/bert_uncased_L-8_H-512_A-8"
OUTPUT_DIR = "D:\LPA_MTech_Project\My_Models\DAPT\dapt_medium_bert"
MAX_LEN = 128
BATCH_SIZE = 4
ACCUM_STEPS = 16    # Effective batch = 64
LR = 3e-5
EPOCHS = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


# ============================
# Load parquet
# ============================
print("Loading parquet...")
df = pd.read_parquet(PARQUET_PATH)


# In[4]:


# Keep only non-empty texts
texts = df[TEXT_COLUMN].dropna().astype(str).tolist()
print(f"Total documents loaded: {len(texts)}")


# In[5]:


# ============================
# Dataset
# ============================
class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Remove the extra dimension
        return {k: v.squeeze(0) for k, v in enc.items()}

   


# In[6]:


# ============================
# Tokenizer & Model
# ============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
model.to(device)


# In[7]:


# ============================
# DataLoader
# ============================
dataset = MLMDataset(texts, tokenizer, MAX_LEN)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)


# In[8]:


# ============================
# Optimizer & Scheduler
# ============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = len(loader) * EPOCHS // ACCUM_STEPS
warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup_steps,
    num_training_steps
)

scaler = torch.cuda.amp.GradScaler()


# In[9]:


# ============================
# Training Loop
# ============================
model.train()
step = 0

print("\nStarting DAPT/MLM Training...\n")

for epoch in range(EPOCHS):
    epoch_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    optimizer.zero_grad()

    for batch in pbar:
        # batch = {k: v.squeeze().to(device) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}


        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        step += 1
        pbar.set_postfix({"loss": epoch_loss / (step+1)})

    ppl = math.exp(epoch_loss / len(loader))
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f} | Perplexity: {ppl:.2f}")

    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")


# In[10]:


print("\nTraining Complete!")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Saved DAPT model successfully.")


# In[ ]:





# In[ ]:





# In[ ]:




