#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json
import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# In[10]:


# CONFIG
# MODEL_DIR = r"My_Models/LegalBERT_Models/legalbert_uncased_1990-2025_medium_1/best_model"
MODEL_DIR = r"D:/LPA_MTech_Project/My_Models/LegalBERT_Models/legalbert_uncased_1990-2025_medium_1/best_model"
IPC_JSON_PATH = "ipc_sections.json"
CASE_FILE_PATH = "2020_1_453_483_EN.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIM_THRESHOLD = 0.75   # raised due to stricter grounding
MAX_LENGTH = 256


# In[11]:


# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()


# In[12]:


# Embedding helpers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)



def embed_text(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**inputs)

    return mean_pooling(output, inputs["attention_mask"]).cpu().numpy()



# In[13]:


# LEGAL GATES
def is_criminal_case(text: str) -> bool:
    criminal_signals = [
        "criminal appeal",
        "convicted under section",
        "offence under section",
        "u/s",
        "u/ss",
        "sentenced to",
        "ipc"
    ]
    t = text.lower()
    return any(s in t for s in criminal_signals)

def section_lexically_present(section: str, text: str) -> bool:
    patterns = [
        rf"section\s+{section}\b",
        rf"section\s+{section}\s+ipc",
        rf"u/s\.?\s*{section}\b",
        rf"u/ss\.?\s*{section}\b"
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def extract_relevant_sentences(text: str, section: str):
    sentences = re.split(r"(?<=[.])\s+", text)
    matches = []

    for s in sentences:
        s_l = s.lower()
        if (
            f"section {section}" in s_l
            or f"u/s {section}" in s_l
            or f"u/ss {section}" in s_l
        ):
            matches.append(s.strip())

    return matches


# In[14]:


# IPC TEXT (LEAN, NOT GENERIC)
def ipc_definition(section, title):
    return f"Section {section} IPC: {title}."

# IPC MAPPING LOGIC
def map_ipc_sections(case_text, ipc_map):
    results = []

    # GATE 1: Civil vs Criminal
    if not is_criminal_case(case_text):
        return []

    for sec, title in ipc_map.items():

        # GATE 2: Section must exist in text
        if not section_lexically_present(sec, case_text):
            continue

        # GATE 3: Sentence-level grounding
        candidate_sentences = extract_relevant_sentences(case_text, sec)
        if not candidate_sentences:
            continue

        ipc_emb = embed_text(ipc_definition(sec, title))

        best_score = 0.0
        best_sentence = None

        for sent in candidate_sentences:
            sent_emb = embed_text(sent)
            score = cosine_similarity(sent_emb, ipc_emb)[0][0]

            if score > best_score:
                best_score = score
                best_sentence = sent

        if best_score >= SIM_THRESHOLD:
            results.append({
                "section": sec,
                "title": title,
                "score": round(best_score, 3),
                "evidence": best_sentence
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)


# In[15]:


# MAIN
# =========================
if __name__ == "__main__":

    with open(IPC_JSON_PATH, "r", encoding="utf-8") as f:
        ipc_map = json.load(f)

    with open(CASE_FILE_PATH, "r", encoding="utf-8") as f:
        case_text = f.read()

    results = map_ipc_sections(case_text, ipc_map)

    print("\nMapped IPC Sections:\n")

    if not results:
        print("✅ No IPC sections applicable.")
    else:
        for r in results:
            print(
                f"IPC {r['section']} — {r['title']} "
                f"(score={r['score']})\n"
                f"  ↳ Evidence: {r['evidence']}\n"
            )


# In[ ]:





# In[ ]:





# In[16]:


# jupyter nbconvert --to python IPC_Test1.ipynb


# In[ ]:





# In[ ]:





# 
