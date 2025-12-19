#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# ============================================================
# 🏛️ High Court Family Law Judgments Downloader (1990–2025)
# Uses AWS Open Data + Parquet Metadata
# Downloads ONLY family-law related PDFs
# ============================================================


# In[2]:


# jupyter nbconvert --to python Download_HC_Family_Cases.ipynb


# In[ ]:


import os
import boto3
import botocore
import tarfile
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import json
import re

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse


# In[4]:


# ---------------- CONFIG ----------------
START_YEAR = int(input("Enter Start Year: "))
END_YEAR   = int(input("Enter End Year: "))

LOCAL_BASE_DIR = f"D:/LPA_MTech_Project/My_Datasets/HC_FAMILY_{START_YEAR}_{END_YEAR}"

BUCKET_NAME = "indian-high-court-judgments"

# IMPORTANT: HC metadata is NOT under /data
META_S3_PATH = "s3://indian-high-court-judgments/metadata/parquet"


# In[5]:


BUCKET_NAME = "indian-high-court-judgments"

FAMILY_ACTS = [
    # Marriage & Divorce
    "hindu marriage act",
    "special marriage act",
    "indian divorce act",
    "parsi marriage and divorce act",

    # Maintenance & DV
    "protection of women from domestic violence act",
    "crpc 125", "crpc 127", "crpc 128",

    # Custody & Family Courts
    "guardians and wards act",
    "family courts act",

    # Personal laws (common in HC metadata)
    "muslim personal law",
    "mohammedan law"
]

FAMILY_IPC = [
    # Matrimonial cruelty & dowry
    "498a", "304b",

    # Property / Streedhan
    "406", "403",

    # Marriage validity / bigamy
    "494", "495", "496",

    # Sexual offences in family context
    "375", "376", "376b", "377",

    # Domestic violence related hurt
    "323", "324", "325", "326",

    # Mental cruelty / threats
    "506", "507", "509",

    # Suicide & abetment
    "306", "107",

    # Child custody / guardianship disputes
    "361", "362", "363",

    # False cases / perjury
    "191", "193", "211"
]


FAMILY_KEYWORDS = [
    "divorce", "judicial separation",
    "matrimonial", "marital dispute",
    "maintenance", "alimony",
    "custody", "guardianship",
    "visitation rights", "child custody",
    "restitution of conjugal rights",
    "domestic violence",
    "child support", "wife maintenance"
]


# In[6]:


# --------- S3 CLIENT (PUBLIC) ----------
s3 = boto3.client(
    "s3",
    config=botocore.client.Config(signature_version=botocore.UNSIGNED)
)

os.makedirs(LOCAL_BASE_DIR, exist_ok=True)


# In[7]:


# --------- LOAD METADATA ---------------
print("📖 Loading HC metadata parquet (partitioned)...")

dataset = ds.dataset(
    META_S3_PATH,
    format="parquet",
    partitioning="hive"
)

meta = dataset.to_table(
    filter=(ds.field("year") >= START_YEAR) & (ds.field("year") <= END_YEAR)
).to_pandas()

# meta = dataset.to_table().to_pandas()

print("Rows loaded:", len(meta))
print("Columns:", meta.columns.tolist())


# In[ ]:


# --------- NORMALIZE PDF FILENAME COLUMN ---------

if "pdf_link" in meta.columns:
    meta["pdf_name"] = meta["pdf_link"].apply(
        lambda x: os.path.basename(urlparse(x).path) if isinstance(x, str) else None
    )
else:
    raise RuntimeError("❌ pdf_link column not found in HC metadata")


# In[ ]:


meta[["pdf_link", "pdf_name"]].head()


# In[ ]:


# --------- SAFETY: Ensure expected columns exist ---------
for col in ["case_type", "act_names", "case_title"]:
    if col not in meta.columns:
        meta[col] = ""


# In[ ]:


# --------- FILTER: YEAR RANGE ----------
meta = meta[(meta["year"] >= START_YEAR) & (meta["year"] <= END_YEAR)]


# In[ ]:


# --------- FILTER: FAMILY LAW ----------
def is_family_case(row):
    text = re.sub(r"[^a-z0-9]", " ", " ".join([
        str(row.get("case_type", "")),
        str(row.get("act_names", "")),
        str(row.get("case_title", "")),
    ]).lower())

    return (
        any(k in text for k in FAMILY_KEYWORDS)
        or any(a in text for a in FAMILY_ACTS)
        or any(ipc in text for ipc in FAMILY_IPC)
        or "family" in text
    )



# In[ ]:


family_df = meta[meta.apply(is_family_case, axis=1)]
print(f"✅ Family law cases identified: {len(family_df)}")


# In[ ]:


# --------- GROUP BY TAR LOCATION -------
groups = family_df.groupby(["year", "court", "bench"])


# In[ ]:


# --------- DOWNLOAD + EXTRACT ----------

groups = family_df.groupby(["year", "court", "bench"])

for (year, court, bench), group_df in tqdm(groups, desc="Processing groups"):

    local_group_dir = Path(LOCAL_BASE_DIR) / str(year) / court / str(bench)
    extract_dir = local_group_dir / "pdfs"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Load tar index
    index_key = f"data/tar/year={year}/court={court}/bench={bench}/data.index.json"

    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=index_key)
        index_data = json.loads(obj["Body"].read())
    except Exception as e:
        print(f"❌ Missing index for {year}/{court}/{bench}: {e}")
        continue

    wanted_pdfs = set(group_df["pdf_name"].dropna())

    if not wanted_pdfs:
        continue

    
    for part in index_data["parts"]:
        tar_key = part["key"]
        local_tar = local_group_dir / Path(tar_key).name

        try:
            s3.download_file(BUCKET_NAME, tar_key, str(local_tar))
        except Exception as e:
            print(f"❌ Failed tar {tar_key}: {e}")
            continue



        if group_df.empty:
            continue

        with tarfile.open(local_tar) as tar:
            for member in tar.getmembers():
                if Path(member.name).name in wanted_pdfs:
                    member.name = Path(member.name).name  # flatten path
                    tar.extract(member, path=extract_dir)

        local_tar.unlink(missing_ok=True)


# In[ ]:


print("\nHigh Court family-law PDFs downloaded")


# In[ ]:





# In[ ]:




