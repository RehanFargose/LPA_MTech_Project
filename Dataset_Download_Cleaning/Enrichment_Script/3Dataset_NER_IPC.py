#!/usr/bin/env python
# Dataset_NER_IPC_HYBRID.py
# Hybrid IPC extractor: Legal-MBERT + Regex + Context Matching
# Loads enriched parquet → extracts IPC → saves updated parquet
# Safe for GTX 1650 (4GB). No HF Dataset issues.

import re
import math
import time
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# --------------------------------------
# Config
# --------------------------------------
MODEL_NAME = "Babelscape/wikineural-multilingual-ner"    # Upgrade from DSLIM → Indian Legal Model
BATCH_SIZE = 8                             # GTX 1650 safe batch size
MAX_CHARS = 3500                           # GPU-safe truncation
DEVICE = 0 if torch.cuda.is_available() else -1

print("Using model:", MODEL_NAME)

# --------------------------------------
# Enhanced IPC Regex Patterns (High Recall)
# --------------------------------------
IPC_REGEXES = [
    r"[Ss]ection\s+(\d{1,3}[A-Za-z]?)",
    r"[Ss]ec\.?\s*(\d{1,3}[A-Za-z]?)",
    r"[Ss]\.?\s*(\d{1,3}[A-Za-z]?)",
    r"[Uu]\s*/\s*[Ss]\.?\s*(\d{1,3}[A-Za-z]?)",
    r"read with\s+[Ss]ection\s+(\d{1,3}[A-Za-z]?)",
    r"[Rr]\s*/\s*[Ww]\s*(\d{1,3}[A-Za-z]?)",
    r"[Rr]\s*[\./]?\s*[Ww]\s*Sec\.?\s*(\d{1,3}[A-Za-z]?)",
    r"[Ii][Pp][Cc]\s*(\d{1,3}[A-Za-z]?)",
    r"(\d{1,3}[A-Za-z]?)\s*[Ii][Pp][Cc]",
]



IPC_NUM_PATTERN = re.compile(r"\b(\d{1,3}[A-Za-z]?)\b")


# --------------------------------------
# Regex extractor
# --------------------------------------
def regex_extract(text):
    results = []
    for pat in IPC_REGEXES:
        matches = re.findall(pat, text)
        for m in matches:
            results.append(m.upper())
    return results


# --------------------------------------
# NER-based extractor (Legal-MBERT tokens)
# --------------------------------------
def ner_extract(preds):
    out = []
    for p in preds:
        token = p.get("word", "")
        m = IPC_NUM_PATTERN.search(token)
        if not m:
            continue
        sec = m.group(1)
        num_match = re.match(r"(\d+)", sec)
        if not num_match:
            continue
        n = int(num_match.group(1))
        if 1 <= n <= 511:
            out.append(sec.upper())
    return out


# --------------------------------------
# Context extractor (detect numbers near “IPC” / “Penal Code” etc.)
# --------------------------------------
def context_extract(text):
    out = []
    words = text.split()
    for i, w in enumerate(words):
        if w.lower() in ["ipc", "code", "penal", "penalcode", "indianpenalcode"]:
            window = words[max(0, i-3): i+4]
            for tok in window:
                m = IPC_NUM_PATTERN.search(tok)
                if m:
                    sec = m.group(1).upper()
                    num = re.match(r"(\d+)", sec)
                    if num:
                        n = int(num.group(1))
                        if 1 <= n <= 511:
                            out.append(sec)
    return out


# --------------------------------------
# Build Legal-MBERT token classifier
# --------------------------------------
def build_pipeline():
    print(f"Loading Legal-MBERT: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )
    return pipe


# --------------------------------------
# BATCHEd inference
# --------------------------------------
def batched_extract(df, pipe):
    n = len(df)
    ipc_out = [None] * n

    print("\n🚀 Running Hybrid IPC Extraction\n")

    for start in tqdm(range(0, n, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n)
        batch_idx = list(range(start, end))

        texts = []
        for idx in batch_idx:
            t = df.iloc[idx]["text"]
            texts.append(t[:MAX_CHARS] if isinstance(t, str) else "")

        # Run NER
        try:
            ner_preds = pipe(texts)
        except RuntimeError:
            print("GPU OOM → fell back to CPU for this batch")
            pipe = pipeline("token-classification", model=model, tokenizer=tokenizer,
                            aggregation_strategy="simple", device=-1, batch_size=1)
            ner_preds = pipe(texts)

        # Merge all extraction methods
        for i, preds in zip(batch_idx, ner_preds):
            text = df.iloc[i]["text"] or ""

            s1 = ner_extract(preds)
            s2 = regex_extract(text)
            s3 = context_extract(text)

            merged = sorted(set(s1 + s2 + s3))
            ipc_out[i] = merged

    return ipc_out


# --------------------------------------
# Main script logic
# --------------------------------------
def main(input_pq, output_pq):

    print("Loading parquet:", input_pq)
    df = pd.read_parquet(input_pq)

    # normalize old IPC
    df["ipc_sections"] = df["ipc_sections"].apply(lambda x: x if isinstance(x, list) else [])

    pipe = build_pipeline()

    start_time = time.time()
    new_sections = batched_extract(df, pipe)
    elapsed = (time.time() - start_time) / 60
    print(f"\n⏱ Finished in {elapsed:.2f} minutes")

    # assign back
    df["ipc_sections"] = new_sections
    df["num_ipc_refs"] = df["ipc_sections"].apply(len)
    df["has_ipc"] = (df["num_ipc_refs"] > 0).astype(int)

    # IPC titles
    df["ipc_titles"] = df["ipc_sections"].apply(lambda lst: [IPC_MAP.get(x, "Unknown") for x in lst])
    df["ipc_mentions"] = df["ipc_sections"]

    print("Saving to:", output_pq)
    df.to_parquet(output_pq, index=False)

    print("\n🎉 DONE — Improved IPC extraction!")
    print("Total cases with IPC:", df["has_ipc"].sum())


# --------------------------------------
# CLI
# --------------------------------------
if __name__ == "__main__":
    start = int(input("Enter Starting Year: "))
    end = int(input("Enter Ending Year: "))
    dataset_num = int(input("Enter Dataset number: "))

    input_file = fr"D:\LPA_MTech_Project\Enriched_Datasets\SupremeCourt_Combined_{start}_{end}_enriched.parquet"
    output_file = fr"D:\LPA_MTech_Project\Enriched_Datasets\SC_{start}_{end}_enriched_IPC_{dataset_num}.parquet"

    main(input_file, output_file)
