import pandas as pd
import numpy as np
from collections import Counter
import re

# =======================================
# 1️⃣ LOAD DATA
# =======================================
FILE_PATH = "SupremeCourt_Combined_2020_2025_enriched.parquet"

print("\nLoading dataset...")
df = pd.read_parquet(FILE_PATH)
print(f"✅ Loaded {len(df):,} cases, {df.shape[1]} columns\n")


# =======================================
# 2️⃣ BASIC INTEGRITY CHECKS
# =======================================
print("========================================")
print(" BASIC INTEGRITY CHECKS")
print("========================================")

# Missing values
missing = df.isnull().mean().sort_values(ascending=False) * 100
print("\n🔍 Missing Values (%):")
print(missing[missing > 0].round(2))

# Duplicate case_ids
dupes = df["case_id"].duplicated().sum()
print(f"\n🔍 Duplicate case_id entries: {dupes}")

# Empty or too short text
empty_text = df[df["text"].str.len() < 50]
print(f"\n🔍 Cases with extremely short text (<50 chars): {len(empty_text)}")


# =======================================
# 3️⃣ VERDICT LABEL CHECK
# =======================================
print("\n========================================")
print(" VERDICT LABEL ANALYSIS")
print("========================================")

vcounts = df["verdict_label"].value_counts(dropna=False)
print("\n🔍 Verdict label distribution:")
print(vcounts)

if df["verdict_label"].isna().sum() > 0:
    print(f"\n⚠️ {df['verdict_label'].isna().sum()} cases have NULL verdict_label")


# =======================================
# 4️⃣ IPC EXTRACTION VALIDATION
# =======================================
print("\n========================================")
print(" IPC EXTRACTION QUALITY")
print("========================================")

# 4.1 How many cases have IPC
ipc_cases = df["has_ipc"].sum()
print(f"\n🔍 Cases with IPC references: {ipc_cases}/{len(df)} "
      f"({ipc_cases*100/len(df):.2f}%)")

# 4.2 IPC sections distribution
ipc_counts = df["ipc_sections"].explode().value_counts().head(20)
print("\n🔍 Top 20 IPC Sections:")
print(ipc_counts)

# 4.3 Cases with IPC but empty ipc_titles
bad_titles = df[(df["has_ipc"] == 1) & (df["ipc_titles"].apply(lambda x: len(x) == 0))]
print(f"\n🔍 Cases with IPC but missing titles: {len(bad_titles)}")


# =======================================
# 5️⃣ SECTION EXTRACTION QUALITY
# =======================================
print("\n========================================")
print(" SECTION EXTRACTION QUALITY")
print("========================================")

section_cols = ["facts_section", "issues_section", "arguments_section",
                "reasoning_section", "conclusion_section"]

for col in section_cols:
    nulls = df[col].isna().sum()
    print(f"🔍 {col}: Missing in {nulls} cases "
          f"({nulls*100/len(df):.2f}%)")

# Very short sections may indicate extraction issues
for col in section_cols:
    too_short = df[df[col].astype(str).str.len() < 20]
    print(f"⚠️ {col}: Extremely short (<20 chars) in {len(too_short)} cases")


# =======================================
# 6️⃣ PARTY TYPE VALIDATION
# =======================================
print("\n========================================")
print(" PARTY TYPE VALIDATION")
print("========================================")

print("\n🔍 Petitioner type distribution:")
print(df["petitioner_type"].value_counts())

print("\n🔍 Respondent type distribution:")
print(df["respondent_type"].value_counts())


# =======================================
# 7️⃣ CASE TYPE DISTRIBUTION
# =======================================
print("\n========================================")
print(" CASE TYPE DISTRIBUTION")
print("========================================")

print(df["case_type"].value_counts())


# =======================================
# 8️⃣ CITATION QUALITY CHECK
# =======================================
print("\n========================================")
print(" CITATIONS QUALITY CHECK")
print("========================================")

no_citations = df[df["num_citations"] == 0]
print(f"🔍 Cases with zero citations: {len(no_citations)}")

duplicate_citations = df[df["citations"].apply(lambda x: len(x) != len(set(x)))]
print(f"🔍 Cases with duplicate citations: {len(duplicate_citations)}")


# =======================================
# 9️⃣ ACT NAMES CHECK
# =======================================
print("\n========================================")
print(" ACT NAMES CHECK")
print("========================================")

act_counts = df["act_names"].explode().value_counts().head(20)
print("\n🔍 Top 20 referenced Acts:")
print(act_counts)


# =======================================
# 🔟 OUTLIER ANALYSIS (Text Length, Complexity)
# =======================================
print("\n========================================")
print(" OUTLIER ANALYSIS")
print("========================================")

# Text length outliers
df["text_len"] = df["text"].str.split().apply(len)

low_len = df[df["text_len"] < 100]
high_len = df[df["text_len"] > 20000]

print(f"🔍 Very short judgments (<100 words): {len(low_len)}")
print(f"🔍 Very long judgments (>20,000 words): {len(high_len)}")

# Complexity score outlier
high_complex = df[df["complexity_score"] > df["complexity_score"].quantile(0.99)]
print(f"🔍 High complexity (99th percentile): {len(high_complex)}")


# =======================================
# 1️⃣1️⃣ SAVE PROBLEMATIC RECORDS
# =======================================
print("\nSaving QC reports...")

empty_text.to_csv("QC_empty_text.csv", index=False)
bad_titles.to_csv("QC_bad_ipc_titles.csv", index=False)
duplicate_citations.to_csv("QC_duplicate_citations.csv", index=False)

print("✅ QC reports saved:\n"
      " - QC_empty_text.csv\n"
      " - QC_bad_ipc_titles.csv\n"
      " - QC_duplicate_citations.csv\n")

print("\n🎉 QUALITY CHECK COMPLETED 🎉")
