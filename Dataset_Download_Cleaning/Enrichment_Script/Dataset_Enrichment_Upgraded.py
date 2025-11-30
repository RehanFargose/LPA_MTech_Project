#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os, re, json
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bs4 import BeautifulSoup
import re


# In[33]:


def extract_case_metadata(raw_html):
    """Extract Decision Date, Case No, Bench, Verdict, Judges, and Parties"""
    if not raw_html:
        return None, None, None, None, None, [], None, None

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # --- Main details ---
    decision_date = re.search(r"Decision\s*Date\s*[:\-]?\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})", text)
    case_no = re.search(r"Case\s*No\s*[:\-]?\s*([A-Z\s.()0-9/,-]+)", text)
    disposal_nature = re.search(r"Disposal\s*(?:Nature)?\s*[:\-]?\s*([A-Za-z\s&()]+)", text)
    bench_info = re.search(r"Bench\s*[:\-]?\s*([A-Za-z0-9\s]+)", text)
    direction_issue = re.search(r"Direction\s*Issue\s*[:\-]?\s*([A-Za-z\s&()]+)", text)

    decision_date = decision_date.group(1).strip() if decision_date else None
    case_no = case_no.group(1).strip() if case_no else None
    disposal_nature = disposal_nature.group(1).strip() if disposal_nature else None
    bench_info = bench_info.group(1).strip() if bench_info else None
    direction_issue = direction_issue.group(1).strip() if direction_issue else None

    # --- Verdict inference ---
    verdict_label = None
    if disposal_nature:
        if re.search(r"allow|grant|upheld|accepted|partly allow|partly allowed|disposed off|quash", disposal_nature, re.I):
            verdict_label = 1.0
        elif re.search(r"dismiss|refuse|reject|disallow", disposal_nature, re.I):
            verdict_label = 0.0
    elif direction_issue:
        if re.search(r"allow|partly", direction_issue, re.I):
            verdict_label = 1.0
        elif re.search(r"dismiss|refuse", direction_issue, re.I):
            verdict_label = 0.0

    # --- Judges (Coram) ---
    coram_match = re.search(r"Coram\s*[:\-]?\s*([^<]+)", raw_html, flags=re.I)
    judges = []
    if coram_match:
        raw_judge = re.sub(r"<[^>]+>", "", coram_match.group(1))
        raw_judge = re.sub(r"\s+", " ", raw_judge)
        judges = [j.strip() for j in re.split(r",|&|and", raw_judge) if j.strip()]
    bench_size = len(judges) if judges else (int(re.search(r"(\d+)", str(bench_info)).group(1)) if bench_info else 1)

    # --- Parties ---
    party_match = re.search(r"<strong>(.*?)<span.*?versus.*?</span>(.*?)</strong>", raw_html, flags=re.I)
    petitioner = party_match.group(1).strip() if party_match else None
    respondent = party_match.group(2).strip() if party_match else None

    return (
        decision_date,
        case_no,
        bench_info,
        disposal_nature,
        verdict_label,
        judges,
        petitioner,
        respondent,
    )


# In[34]:


# Take the 2 inputs for start and end years
start = int(input("Enter Starting Year: "))
end = int(input("Enter Ending Year: "))


# In[35]:


# =======================================
# 1️⃣ Folder structure
# =======================================
# base_json_dir = os.path.join("My_Datasets", "SC_2020-2025")
# text_dir = "Extracted_Texts"
base_root = r"D:\LPA_MTech_Project"

# base_json_dir = os.path.join(base_root, "My_Datasets", "SC_2020-2025")
# text_dir      = os.path.join(base_root, "Extracted_Texts", "Texts_2020-2025")
base_json_dir = os.path.join(base_root, "My_Datasets", f"SC_{start}-{end}")
text_dir      = os.path.join(base_root, "Extracted_Texts", f"Texts_{start}-{end}")

# Dynamically detect all metadata folders under each year
year_folders = []
for year in range(start, end+1):
    metadata_path = os.path.join(base_json_dir, str(year), "metadata")
    if os.path.exists(metadata_path):
        year_folders.append(metadata_path)

print("Detected metadata folders:")
for folder in year_folders:
    print(" →", folder)

# Collect all JSON files
json_files = []
for year_dir in year_folders:
    for f in os.listdir(year_dir):
        if f.endswith(".json"):
            json_files.append(os.path.join(year_dir, f))

print(f"\n✅ Found {len(json_files)} metadata files across {len(year_folders)} year folders.")


# In[36]:


# =======================================
# 2️⃣ Load + Merge JSON ↔ Text
# =======================================
records = []

for jfile in tqdm(json_files):
    base = os.path.splitext(os.path.basename(jfile))[0]
    text_path = os.path.join(text_dir, base + "_EN.txt")

    if not os.path.exists(text_path):
        continue  # skip missing

    try:
        with open(jfile, "r", encoding="utf-8") as jf:
            meta = json.load(jf)
        with open(text_path, "r", encoding="utf-8") as tf:
            text = tf.read()

        raw_html = meta.get("raw_html", "")
        year = meta.get("citation_year")

        # --- Extract structured metadata using the universal parser ---
        (
            decision_date,
            case_no,
            bench_info,
            disposal_nature,
            verdict_label,
            judges,
            petitioner,
            respondent
        ) = extract_case_metadata(raw_html)

        # --- Compute bench size safely ---
        bench_size = len(judges) if judges else (int(re.search(r"(\d+)", str(bench_info)).group(1)) if bench_info and re.search(r"\d+", bench_info) else 1)

        record = {
            "case_id": meta.get("path", base),
            "year": int(year) if year else None,
            "decision_date": decision_date,
            "case_no": case_no,
            "bench_info": bench_info,
            "bench_size": bench_size,
            "judges": judges,
            "petitioner": petitioner,
            "respondent": respondent,
            "disposal_nature": disposal_nature,
            "verdict_label": verdict_label,
            "raw_html": raw_html,
            "text": text
        }

        records.append(record)

    except Exception as e:
        print(f"⚠️ Error reading {jfile}: {e}")

df = pd.DataFrame(records)
print("✅ Records loaded:", len(df))
print("Columns:", df.columns.tolist())


# In[ ]:





# In[ ]:





# In[37]:


# =======================================
# 3️⃣ Feature Enrichment
# =======================================
df = df.dropna(subset=["text"])
df["year"] = pd.to_numeric(df.get("year", 0)).fillna(0).astype(int)
df["decade"] = (df["year"] // 10) * 10

# ---- Structural ----
df["num_words_text"] = df["text"].apply(lambda x: len(str(x).split()))
df["num_lines"] = df["text"].apply(lambda x: len(str(x).splitlines()))

# ---- Legal Signals ----
def extract_act_names(text):
    # return re.findall(r'([A-Z][A-Za-z\s]+Act[, ]\s*\d{4})', str(text))
    return re.findall(r'([A-Z][A-Za-z/&\s]+ Act[, ]\s*\d{4})', str(text))





# In[ ]:





# In[38]:


# ===============================
#  Party Type Classification
# ===============================

def party_type(name):
    if not isinstance(name, str):
        return "Unknown"
    n = name.lower()

    # --- Clear Government Identifiers ---
    if "union of india" in n or "ministry" in n:
        return "Union Govt"

    if "state of" in n:
        return "State Govt"

    # --- Government Departments / Authorities ---
    if any(x in n for x in ["commissioner", "collector", "officer", "authority", "board", "tribunal"]):
        return "Government Agency"

    # --- Courts / Judicial bodies ---
    if any(x in n for x in ["court", "high court", "supreme court"]):
        return "Court"

    # --- Companies / Corporations ---
    if any(x in n for x in ["ltd", "private", "pvt", "industries", "company", "corporation"]):
        return "Company"

    # --- Trusts / Societies / NGOs ---
    if any(x in n for x in ["trust", "ngo", "society"]):
        return "Organisation"

    # --- Default fallback ---
    return "Individual"


df["petitioner_type"] = df["petitioner"].apply(party_type)
df["respondent_type"] = df["respondent"].apply(party_type)


# In[39]:


# ===============================
#   Robust Verdict Mapping
# ===============================

def map_verdict(txt):
    if not txt:
        return None
    t = txt.lower()

    WIN = ["allowed", "granted", "set aside", "quashed", "modified", "partly allowed"]
    LOSS = ["dismissed", "rejected", "refused", "not maintainable"]

    if any(k in t for k in WIN):
        return 1
    if any(k in t for k in LOSS):
        return 0
    return None

# df["verdict_label"] = df["disposal_nature"].apply(map_verdict)
df["verdict_label"] = df.apply(
    lambda r: r["verdict_label"] if r["verdict_label"] is not None 
    else map_verdict(r["disposal_nature"]),
    axis=1
)


# In[40]:


# ===============================
#  Case Complexity Metrics
# ===============================

df["avg_sentence_length"] = df["text"].apply(
    lambda t: sum(len(s.split()) for s in t.split(".")) / (len(t.split(".")) + 1)
)

df["num_paragraphs"] = df["text"].apply(lambda t: t.count("\n\n"))

# df["complexity_score"] = df["avg_sentence_length"] * (df["num_citations"] + 1)


# In[41]:


# ===============================
#   Proceeding Type
# ===============================

def classify_proceeding(text):
    t = str(text).lower()
    if "special leave petition" in t or "slp" in t:
        return "SLP"
    if "criminal appeal" in t:
        return "Criminal Appeal"
    if "civil appeal" in t:
        return "Civil Appeal"
    if "writ petition" in t:
        return "Writ Petition"
    if "review petition" in t:
        return "Review"
    return "Other"

df["proceeding_type"] = df["text"].apply(classify_proceeding)


# In[42]:


# ===============================
#  IPC Titles Mapping
# ===============================

IPC_MAP = {
    # Old codes
    # "34": "Common intention",
    # "406": "Criminal breach of trust",
    # "419": "Impersonation",
    # "420": "Cheating",
    # "467": "Forgery of valuable security",
    # "468": "Forgery for cheating",
    # "471": "Using forged document",

    # Full Expansion
    "34": "Common intention",
    "35": "Criminal knowledge shared",
    "36": "Effect caused partly by act and partly by omission",
    "37": "Cooperation in criminal act",
    "38": "Persons involved liable to different offences",

    "120": "Concealing design to commit offence",
    "120A": "Criminal conspiracy (definition)",
    "120B": "Criminal conspiracy",

    "141": "Unlawful assembly",
    "143": "Punishment for unlawful assembly",
    "144": "Joining unlawful assembly armed with deadly weapon",
    "147": "Rioting",
    "148": "Rioting with deadly weapon",
    "149": "Unlawful assembly — common object",

    "153": "Wantonly giving provocation",
    "153A": "Promoting enmity between groups",
    "153B": "Imputations prejudicial to national integration",

    "166": "Public servant disobeying law",
    "167": "Public servant framing incorrect document",
    "168": "Public servant unlawfully engaging in trade",
    "169": "Public servant unlawfully buying or bidding",

    "171B": "Bribery",
    "171C": "Undue influence",
    "171D": "Personation at elections",
    "171E": "Punishment for bribery",
    "171F": "Punishment for undue influence",
    "171G": "False statement",
    "171H": "Illegal payments",
    "171I": "Failure to keep election accounts",

    "172": "Absconding to avoid summons",
    "173": "Preventing service of summons",
    "175": "Omission to produce document",
    "177": "Furnishing false information",

    "186": "Obstructing public servant",
    "188": "Disobedience to order promulgated by public servant",

    "191": "Giving false evidence",
    "192": "Fabricating false evidence",
    "193": "Punishment for false evidence",
    "196": "Using evidence known to be false",
    "197": "Issuing false certificate",
    "198": "Using false certificate",

    "199": "False statement made in declaration",
    "200": "Using false declaration",

    "201": "Causing disappearance of evidence",
    "202": "Intentional omission to give information",
    "203": "Giving false information",
    "204": "Destruction of document",

    "209": "Dishonestly making false claim",
    "211": "False charge of offence",

    "268": "Public nuisance",
    "269": "Negligent act likely to spread infection",
    "270": "Malignant act likely to spread infection",
    "272": "Adulteration of food",
    "273": "Sale of noxious food or drink",

    "292": "Obscene books and objects",
    "294": "Obscene acts and songs",

    "295": "Injuring place of worship",
    "295A": "Outraging religious feelings",
    "296": "Disturbing religious assembly",
    "297": "Trespassing on burial places",
    "298": "Hurting religious feelings",

    "300": "Definition of murder",
    "302": "Murder",
    "304": "Culpable homicide not amounting to murder",
    "304A": "Causing death by negligence",
    "304B": "Dowry death",
    "306": "Abetment of suicide",
    "307": "Attempt to murder",
    "308": "Attempt to commit culpable homicide",

    "312": "Causing miscarriage",
    "313": "Miscarriage without woman's consent",
    "314": "Death caused by miscarriage",
    "315": "Act to prevent child being born alive",
    "316": "Causing death of unborn child",

    "319": "Hurt (definition)",
    "320": "Grievous hurt (definition)",
    "323": "Punishment for hurt",
    "324": "Hurt by dangerous weapons",
    "325": "Punishment for grievous hurt",
    "326": "Voluntarily causing grievous hurt with dangerous weapons",
    "326A": "Acid attack",
    "326B": "Attempted acid attack",

    "339": "Wrongful restraint",
    "340": "Wrongful confinement",
    "341": "Punishment for wrongful restraint",
    "342": "Wrongful confinement",
    "343": "Confinement for 3+ days",
    "344": "Confinement for 10+ days",

    "349": "Force (definition)",
    "351": "Assault (definition)",
    "352": "Punishment for assault",
    "354": "Assault on woman with intent to outrage modesty",
    "354A": "Sexual harassment",
    "354B": "Assault on woman with intent to disrobe",
    "354C": "Voyeurism",
    "354D": "Stalking",

    "359": "Kidnapping (definition)",
    "360": "Kidnapping from India",
    "361": "Kidnapping from lawful guardianship",
    "363": "Punishment for kidnapping",
    "363A": "Kidnapping for begging",
    "364": "Kidnapping with murder intent",
    "364A": "Kidnapping for ransom",
    "365": "Kidnapping to confine",
    "366": "Kidnapping woman to compel marriage",
    "366A": "Procuration of minor girl",
    "366B": "Importation of girl",
    "367": "Kidnapping to cause hurt",
    "368": "Concealing kidnapped person",
    "369": "Kidnapping child to steal property",

    "375": "Rape (definition)",
    "376": "Punishment for rape",
    "376A": "Death of woman during rape",
    "376AB": "Rape of girl under 12",
    "376B": "Sexual intercourse by husband during separation",
    "376C": "Sexual intercourse by authority",
    "376D": "Gang rape",
    "376DA": "Gang rape of girl under 16",
    "376DB": "Gang rape of girl under 12",
    "376E": "Repeat offenders",

    "379": "Theft",
    "380": "Theft in dwelling house",
    "381": "Theft by clerk or servant",
    "382": "Theft after preparing for hurt",

    "390": "Robbery (definition)",
    "391": "Dacoity (definition)",
    "392": "Robbery punishment",
    "393": "Attempt to commit robbery",
    "394": "Voluntarily causing hurt in robbery",
    "395": "Dacoity",
    "396": "Dacoity with murder",
    "397": "Robbery/dacoity with attempt to cause death",
    "398": "Robbery attempt with deadly weapon",
    "399": "Preparation to commit dacoity",
    "400": "Belonging to gang of dacoits",
    "402": "Assembling for purpose of committing dacoity",

    "403": "Dishonest misappropriation",
    "404": "Misappropriation of deceased’s property",
    "405": "Criminal breach of trust (definition)",
    "406": "Criminal breach of trust",
    "409": "Criminal breach of trust by public servant/banker",

    "410": "Stolen property (definition)",
    "411": "Dishonestly receiving stolen property",
    "413": "Habitual dealing in stolen property",

    "415": "Cheating (definition)",
    "416": "Cheating by personation",
    "417": "Punishment for cheating",
    "418": "Cheating with knowledge of trust",
    "419": "Personation",
    "420": "Cheating and dishonestly inducing delivery of property",

    "425": "Mischief (definition)",
    "426": "Punishment for mischief",
    "427": "Mischief causing damage",
    "428": "Mischief by killing animal",
    "429": "Mischief to larger animals",

    "441": "Criminal trespass (definition)",
    "447": "Punishment for criminal trespass",
    "448": "House trespass",
    "449": "House trespass with intent to commit offence",
    "450": "House trespass with intent to cause hurt",
    "452": "House trespass after preparation for hurt",
    "454": "Lurking house trespass",
    "457": "Lurking house trespass or housebreaking by night",
    "458": "Housebreaking by night",
    "459": "Housebreaking with hurt",
    "460": "All persons jointly liable for housebreaking by night",

    "463": "Forgery (definition)",
    "464": "Making false document",
    "465": "Punishment for forgery",
    "466": "Forgery of court record",
    "467": "Forgery of valuable security",
    "468": "Forgery for cheating",
    "469": "Forgery for harming reputation",
    "471": "Using forged document",
    "474": "Possession of forged document",
    "475": "Counterfeiting seal",
    "476": "Counterfeiting device",

    "489A": "Counterfeiting currency",
    "489B": "Using counterfeit currency",
    "489C": "Possession of counterfeit currency",
    "489D": "Making instruments for counterfeiting",
}

# df["ipc_titles"] = df["ipc_sections"].apply(lambda lst: [IPC_MAP.get(x, "Unknown") for x in lst])


# In[ ]:





# In[ ]:





# In[43]:


# =======================================
# 🔥 Smart IPC Extraction (Group-aware)
# =======================================

def _is_likely_money_or_para(context):
    reject_tokens = [
        "rs", "₹", "crore", "crores", "lakh", "lakhs",
        "para", "paragraph", "clause", "instal", "install",
        "scheme", "nia", "bank guarantee", "guarantee"
    ]
    ctx = context.lower()
    for t in reject_tokens:
        if t in ctx:
            return True
    if re.search(r'[\d\.,]+\s*(crore|lakh|rs|₹)', ctx):
        return True
    return False


def _clean_num(num_str):
    """Return normalized IPC section number or None."""
    num_match = re.match(r"(\d+)([A-Za-z]*)", num_str.strip())
    if not num_match:
        return None

    num = int(num_match.group(1))
    if not (1 <= num <= 511):
        return None

    suffix = num_match.group(2).upper()
    return f"{num}{suffix}" if suffix else str(num)


def extract_ipc_mentions(text):
    if not isinstance(text, str):
        return []

    results = set()

    # ---------------------------------------------------
    # 1️⃣ SMART GROUP MODE
    # Matches: "Sections 34, 467, 420 and 406 IPC"
    # ---------------------------------------------------
    group_pattern = r"Sections?\s+([0-9A-Za-z ,and]+?)\s+(?:IPC|I\.P\.C\.|Indian Penal Code)"
    for m in re.finditer(group_pattern, text, flags=re.I):
        group_txt = m.group(1)
        # extract all numbers in the group
        nums = re.findall(r"\d+[A-Za-z]?", group_txt)
        for n in nums:
            cleaned = _clean_num(n)
            if cleaned:
                results.add(cleaned)

    # ---------------------------------------------------
    # 2️⃣ INDIVIDUAL MODE
    # Matches: "u/s 406", "s. 420 IPC", "Section 468"
    # ---------------------------------------------------
    single_pattern = r"(?:Section|Sec\.?|s\.?|u/s)\s*([0-9]{1,3}[A-Za-z]?)"
    for m in re.finditer(single_pattern, text, flags=re.I):
        sec_raw = m.group(1)
        start, end = m.span()

        # Context window for filtering
        ctx = text[max(0, start-50):min(len(text), end+50)]

        # Must avoid money/clause
        if _is_likely_money_or_para(ctx):
            continue

        # 🔥 MUST have explicit IPC mention nearby
        if "ipc" not in ctx.lower() and "indian penal code" not in ctx.lower():
            continue

        # Clean & validate
        cleaned = _clean_num(sec_raw)
        if cleaned:
            results.add(cleaned)

    return sorted(results)


def extract_additional_ipc_mentions(text):
    """Catches: '420 IPC', '406 I.P.C.', etc."""
    if not isinstance(text, str):
        return []

    results = set()
    pattern = r"\b([0-9]{1,3}[A-Za-z]?)\s*(?:IPC|I\.P\.C\.|Indian Penal Code)"
    for m in re.finditer(pattern, text, flags=re.I):
        cleaned = _clean_num(m.group(1))
        if cleaned:
            results.add(cleaned)
    return sorted(results)


# In[44]:


#  ===== Apply to dataframe (keeps same downstream columns) =====
# Replace previous assignments for ipc detection with the following:


# Merge without double-counting: union of both lists

def _union_lists(a, b): 
    return sorted(set((a or []) + (b or [])))


# In[45]:


df["act_names"] = df["text"].apply(extract_act_names)
df["ipc_mentions"] = df["text"].apply(extract_ipc_mentions)
df["ipc_mentions_extra"] = df["text"].apply(extract_additional_ipc_mentions)
df["ipc_sections"] = df.apply(lambda r: _union_lists(r["ipc_mentions"], r["ipc_mentions_extra"]), axis=1)

df["ipc_titles"] = df["ipc_sections"].apply(lambda lst: [IPC_MAP.get(x, "Unknown") for x in lst])

df["num_ipc_refs"] = df["ipc_sections"].apply(len)
df["has_ipc"] = (df["num_ipc_refs"] > 0).astype(int)
df["ipc_mentions"] = df["ipc_sections"]  

df["num_unique_acts"] = df["act_names"].apply(lambda x: len(set(x)))

# Print quick stats (similar to prior message)
total_cases = len(df)
cases_with_ipc = df["has_ipc"].sum()
print(f"✅ IPC detection refined: {cases_with_ipc}/{total_cases} cases ({cases_with_ipc/total_cases*100:.2f}%) now flagged as having IPC sections.")


# In[ ]:





# In[46]:


# ===============================
#  Section-wise Extraction
# ===============================

def extract_section(text, label):
    pattern = rf"(?i)\b{label}\b.*?[:\-]?\s*(.*?)(?=\n[A-Z][A-Za-z ]{{3,}}[:\-]|$)"
    m = re.search(pattern, text, flags=re.S)

    if not m:
        return None

    section = m.group(1)
    if not section or section.strip() == "":
        return None

    return section.strip()


df["facts_section"] = df["text"].apply(lambda t: extract_section(t, "Facts"))
df["issues_section"] = df["text"].apply(lambda t: extract_section(t, "Issues"))
df["arguments_section"] = df["text"].apply(lambda t: extract_section(t, r"\b(?:Arguments|Contentions)\b"))
df["reasoning_section"] = df["text"].apply(lambda t: extract_section(t, "(Reasoning|Analysis|Discussion)"))
df["conclusion_section"] = df["text"].apply(lambda t: extract_section(t, "(Conclusion|Held|Order)"))


# In[ ]:





# In[47]:


# ---- Sentiment + Context ----
pos_words = ["granted", "allowed", "favour", "upheld", "entitled"]
neg_words = ["dismissed", "rejected", "refused", "denied", "not maintainable"]
df["sentiment_score"] = df["text"].apply(lambda t: sum(w in t.lower() for w in pos_words) - sum(w in t.lower() for w in neg_words))
df["petitioner_mentions"] = df["text"].str.count(r"(?i)petitioner")
df["respondent_mentions"] = df["text"].str.count(r"(?i)respondent")
df["pet_vs_resp_ratio"] = df.apply(lambda r: (r["petitioner_mentions"] + 1)/(r["respondent_mentions"] + 1), axis=1)


# In[48]:


# ---- Keyword extraction ----
def extract_keywords(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', str(text).lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    common = [w for w, _ in Counter(words).most_common(top_n)]
    return common

df["keywords"] = df["text"].apply(lambda x: extract_keywords(x, 10))


# In[49]:


# ---- Case type inference ----
def infer_case_type(text):
    t = str(text).lower()
    if "criminal" in t: return "Criminal"
    if "civil" in t: return "Civil"
    if "tax" in t or "income tax" in t: return "Tax"
    if "constitution" in t: return "Constitutional"
    if "labour" in t or "industrial" in t: return "Labour"
    return "Other"

df["case_type"] = df["text"].apply(infer_case_type)


# In[ ]:


# # ---- Citations ----

# Master regex patterns for Indian case law citations
CITATION_PATTERNS = [
    # SCC citations (with Suppl)
    r"\(\d{4}\)\s*\d+\s*SCC\s*\d+",
    r"\(\d{4}\)\s*\d+\s*SUPPL\.?\s*SCC\s*\d+",
    r"\(\d{4}\)\s*SUPPL\.?\s*\(\d+\)\s*SCC\s*\d+",

    # AIR citations (any court: SC, PC, ALL, etc.)
    r"AIR\s*\d{4}\s*[A-Z]{1,4}\s*\d+",

    # SCR citations (including Suppl)
    r"\[\d{4}\]\s*\d+\s*SCR\s*\d+",
    r"\[\d{4}\]\s*\d+\s*SUPPL\.?\s*SCR\s*\d+",

    # SCC OnLine
    r"\d{4}\s*SCC\s*OnLine\s*[A-Za-z]+\s*\d+",

    # Cri LJ
    r"\(\d{4}\)\s*\d+\s*Cri\s*LJ\s*\d+",

    # Comp Cas
    r"\(\d{4}\)\s*\d+\s*Comp\s*Cas\s*\d+",

    # All ER
    r"\(\d{4}\)\s*\d+\s*ALL\s*ER\s*\d+",

    # ALT, AWC, AIC, MLJ, ALD
    r"\(\d{4}\)\s*\d+\s*(ALT|AWC|AIC|MLJ|ALD)\s*\d+",
]


def normalize_citation(c):
    """Normalize spacing and capitalization to canonical form."""
    c = re.sub(r"\s+", " ", c).strip()
    c = c.replace("SUPREME COURT", "SC")
    c = c.upper()
    return c


def extract_legal_citations(text):
    """Extract, normalize, dedupe, and merge parallel legal citations."""
    if not isinstance(text, str):
        return []

    citations = []

    # Run all patterns
    for pattern in CITATION_PATTERNS:
        found = re.findall(pattern, text, flags=re.I)
        citations.extend(found)

    # Flatten tuples from capture groups (AIR)
    flat = []
    for c in citations:
        if isinstance(c, tuple):
            flat.append(" ".join(c))
        else:
            flat.append(c)

    # Normalize all citations
    normalized = [normalize_citation(c) for c in flat]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in normalized:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    # Split parallel citations (using colon ":")
    final = []
    for c in unique:
        parts = [p.strip() for p in c.split(":") if p.strip()]
        final.extend(parts)

    # Final dedupe
    final_unique = []
    seen2 = set()
    for c in final:
        if c not in seen2:
            seen2.add(c)
            final_unique.append(c)

    return final_unique


# Apply to DataFrame
df["citations"] = df["text"].apply(extract_legal_citations)
df["num_citations"] = df["citations"].apply(len)

df["complexity_score"] = df["avg_sentence_length"] * (df["num_citations"] + 1)


# In[51]:


# ---- Constitution Articles ----
def extract_constitution_articles(text):
    return re.findall(r'Article\s*\d+[A-Za-z0-9]*', str(text), flags=re.I)

df["constitution_articles"] = df["text"].apply(extract_constitution_articles)
df["has_constitution_article"] = df["constitution_articles"].apply(lambda x: 1 if len(x) > 0 else 0)


# In[52]:


# =======================================
# 4️⃣ Save enriched parquet
# =======================================
out_path = f"D:\LPA_MTech_Project\Enriched_Datasets\SupremeCourt_Combined_{start}_{end}_enriched.parquet"
df.to_parquet(out_path, index=False)
print(f"✅ Enriched dataset saved: {out_path} — shape: {df.shape}")


# In[ ]:





# Dataset Quality Visualizer

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[54]:


# =======================================
# 1️⃣ Load dataset
# =======================================
df = pd.read_parquet(f"D:\LPA_MTech_Project\Enriched_Datasets\SupremeCourt_Combined_{start}_{end}_enriched.parquet")
print(f"✅ Loaded {len(df)} records with {df.shape[1]} columns")


# In[55]:


# =======================================
# 2️⃣ Missing value audit
# =======================================
print("\n--- Missing Values (Top 15) ---")
missing = df.isnull().mean().sort_values(ascending=False) * 100
print(missing.head(15).round(2))

plt.figure(figsize=(8,4))
sns.barplot(x=missing.head(10).index, y=missing.head(10).values)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Columns by Missing %")
plt.ylabel("% Missing")
plt.show()


# In[56]:


# =======================================
# 3️⃣ Data completeness & label balance
# =======================================
print("\n--- Verdict Label Distribution ---")
print(df['verdict_label'].value_counts(dropna=False))
sns.countplot(x='verdict_label', data=df)
plt.title("Verdict Label Balance (0=Dismissed, 1=Allowed)")
plt.show()


# In[57]:


# =======================================
# 4️⃣ Text quality metrics
# =======================================
print("\n--- Text Length Stats ---")
df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
print(df['text_len'].describe().round(2))

plt.figure(figsize=(7,4))
sns.histplot(df['text_len'], bins=50, kde=True)
plt.title("Judgment Text Length Distribution")
plt.xlabel("Number of Words")
plt.show()


# In[58]:


# =======================================
# 5️⃣ Feature richness audit
# =======================================
print("\n--- Bench & Case Type ---")
print("Average bench size:", round(df['bench_size'].mean(), 2))
print("\nCase type distribution:\n", df['case_type'].value_counts(normalize=True).mul(100).round(1))

print("\nHas IPC references:", df['has_ipc'].sum(), "/", len(df))
print("Has Constitution Articles:", df['has_constitution_article'].sum(), "/", len(df))


# In[59]:


# =======================================
# 6️⃣ Correlation between numeric features
# =======================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[num_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Numeric Feature Correlation Heatmap")
plt.show()


# In[60]:


# =======================================
# 7️⃣ Quick semantic sanity: top frequent Acts & Articles
# =======================================
acts = df['act_names'].explode().dropna().value_counts().head(10)
articles = df['constitution_articles'].explode().dropna().value_counts().head(10)

print("\n--- Top 10 Most Frequent Acts ---\n", acts)
print("\n--- Top 10 Constitution Articles ---\n", articles)


# In[61]:


# 2020_11_786_799


# In[ ]:


# jupyter nbconvert --to python Dataset_Enrichment_Upgraded.ipynb

