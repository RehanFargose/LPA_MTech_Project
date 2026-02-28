#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, re, json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import fitz  # PyMuPDF
from pathlib import Path


# In[ ]:


# jupyter nbconvert --to python text_extraction_improved.ipynb


# In[2]:


# -------------------------------------------------------------------
# Patterns that MUST be preserved for downstream enrichment
# -------------------------------------------------------------------
ANCHOR_PATTERNS = [
    r'^\s*(?:CORAM)\b[:\s-].*',
    r'^\s*(?:BENCH)\b[:\s-].*',
    r'^\s*(?:JUDGMENT|JUDGEMENT)\b.*',
    r'^\s*(?:ORDER)\b.*',
    r'^\s*(?:FACTS|BACKGROUND|BRIEF FACTS)\b.*',
    r'^\s*(?:ISSUES|ARGUMENTS|ANALYSIS|REASONING|CONCLUSION)\b.*',
]

# -------------------------------------------------------------------
# Citation patterns we want to PRESERVE
# -------------------------------------------------------------------
CITATION_INLINE_PATTERNS = [
    r'\[\d{4}\]\s*\d+\s*S\.C\.R\.\s*\d+',
    r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',
    r'\bAIR\s+\d{4}\b',
    r'\bSCC\b',
    r'\bSCR\b',
]

def contains_citation(line):
    return any(re.search(p, line) for p in CITATION_INLINE_PATTERNS)


# -------------------------------------------------------------------
# True page-header patterns (only delete these, not real citations!)
# -------------------------------------------------------------------
PAGE_HEADER_PATTERNS = [
    r'^\s*SUPREME COURT REPORTS\b.*$',
    r'^\s*SUPREME COURT OF INDIA\b.*$',
    r'^\s*ITEM NO\..*$',
    r'^\s*COURT NO\..*$',
    r'^\s*SECTION\s+[A-Z0-9-]+$',
    r'^\s*REPORTABLE\s*$',
    r'^\s*NON-REPORTABLE\s*$',
]

PAGE_NUMBER_PATTERN = re.compile(r'^\s*(?:Page\s+)?\d+\s*$', flags=re.IGNORECASE)


# -------------------------------------------------------------------
# CLEANING FUNCTION — FINAL VERSION (STRUCTURE PRESERVED)
# -------------------------------------------------------------------
def clean_case_text_keep_structure(text):

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')

    cleaned = []

    for ln in lines:
        raw_line = ln.strip()

        if raw_line == "":
            cleaned.append("")
            continue

        # 1. KEEP all anchor lines
        if any(re.match(p, ln, flags=re.IGNORECASE) for p in ANCHOR_PATTERNS):
            cleaned.append(raw_line)
            continue

        # 2. Keep real citation lines
        if contains_citation(ln):
            cleaned.append(raw_line)
            continue

        # 3. Remove pure page numbers ONLY
        if PAGE_NUMBER_PATTERN.match(ln):
            continue

        # 4. Remove TRUE page headers (but NOT citations!)
        if any(re.match(p, ln, flags=re.IGNORECASE) for p in PAGE_HEADER_PATTERNS):
            continue

        # 5. Remove long dashed separators
        if re.match(r'^\s*[-_]{5,}\s*$', ln):
            continue

        # 6. Normalize spacing inside lines (NOT newlines)
        ln2 = re.sub(r'[ \t]{2,}', ' ', ln).rstrip()
        cleaned.append(ln2)

    # ---------------------------------------------
    # Preserve paragraph structure
    # ---------------------------------------------
    joined = "\n".join(cleaned)
    joined = re.sub(r'\n{3,}', '\n\n', joined)  # collapse >2 blank lines

    # Remove insane letter ladders like A B C D ...
    joined = re.sub(r'(^|\n)(?:[A-Z]\s+){5,}(\n|$)', '\n', joined)

    # Final trim
    return joined.strip()


# In[3]:


def extract_light_metadata(text):
    meta = {}

    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    meta['paragraph_count'] = len(paragraphs)
    meta['word_count'] = sum(len(p.split()) for p in paragraphs)
    meta['first_paragraphs'] = paragraphs[:2]

    # Find judge/bench lines
    def find(pattern):
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(0).strip() if m else None

    meta['coram_line'] = find(r'^.*\bCORAM\b.*$')
    meta['bench_line'] = find(r'^.*\bBENCH\b.*$')
    meta['has_judgment_header'] = bool(find(r'^.*\bJUDGMENT\b.*$'))

    return meta


# In[4]:


def extract_text_fitz(pdf_path, txt_path, meta_path=None):

    try:
        pages = []
        with fitz.open(pdf_path) as doc:
            for pg in doc:
                txt = pg.get_text("text")
                # fallback
                if not txt.strip():
                    txt = "\n".join(b[4] for b in pg.get_text("blocks") if b[4].strip())

                pages.append(txt.strip())

        raw = "\n\n".join(pages)

        cleaned = clean_case_text_keep_structure(raw)

        # Fallback — ensure text is not empty after cleaning
        if len(cleaned) < 500:
            cleaned = raw

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        if meta_path:
            meta = extract_light_metadata(cleaned)
            meta['source_pdf'] = os.path.basename(pdf_path)
            meta['txt_file'] = os.path.basename(txt_path)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"⚠️ Error: {pdf_path} → {e}")


# In[5]:


START = int(input("Enter Start Year: "))
END   = int(input("Enter End Year: "))

BASE_DIR = f"D:/LPA_MTech_Project/My_Datasets/SC_{START}-{END}"
OUT_DIR  = f"D:/LPA_MTech_Project/Extracted_Texts/Texts_{START}-{END}"

os.makedirs(OUT_DIR, exist_ok=True)


# In[6]:


def collect_pdfs(base_dir, start, end):
    pdfs = []
    for yr in range(start, end + 1):
        year_path = os.path.join(base_dir, str(yr), "english")
        if os.path.exists(year_path):
            found = [
                os.path.join(year_path, f)
                for f in os.listdir(year_path)
                if f.lower().endswith(".pdf")
            ]
            pdfs.extend(found)
            print(f"[{yr}] Found {len(found)} PDFs")
    print("TOTAL:", len(pdfs))
    return pdfs

pdf_files = collect_pdfs(BASE_DIR, START, END)


# In[7]:


with ThreadPoolExecutor(max_workers=6) as ex:
    for p in tqdm(pdf_files, desc="Extracting PDFs"):
        base = Path(p).stem
        txt_path = os.path.join(OUT_DIR, base + ".txt")
        meta_path = os.path.join(OUT_DIR, base + ".meta.json")
        ex.submit(extract_text_fitz, p, txt_path, meta_path)

print("✔ DONE — Text + metadata saved to:", OUT_DIR)


# In[8]:


sample = Path(pdf_files[0]).stem

print("TXT PREVIEW:\n")
with open(os.path.join(OUT_DIR, sample + ".txt"), "r", encoding="utf-8") as f:
    print(f.read()[:2000])


# In[9]:


with open(os.path.join(OUT_DIR, sample + ".meta.json"), "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), indent=2))


# In[ ]:




