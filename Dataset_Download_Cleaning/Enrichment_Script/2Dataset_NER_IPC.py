#!/usr/bin/env python
# Dataset_NER_IPC_Final.py
# Purpose: Load existing enriched parquet, run batched NER to extract IPC sections,
# update ipc_sections / num_ipc_refs / has_ipc / ipc_titles, and save updated parquet.
# Safe for GTX 1650 (4GB). Uses pandas batching (no HF Datasets).

import re
import math
import time
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# -----------------------
# Config (tweak if needed)
# -----------------------
MODEL_NAME = "dslim/bert-base-NER"
BATCH_SIZE = 8           # safe default for GTX 1650 (4GB). Increase if you know your VRAM headroom.
MAX_CHARS = 3500         # truncate long docs for inference to save VRAM
DEVICE = 0 if torch.cuda.is_available() else -1
# -----------------------

# -----------------------
# IPC map (paste your full map; shortened here for brevity - paste your full map in real run)
# -----------------------
IPC_MAP = {
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
# -----------------------

# Utility: regex to find IPC-like tokens
IPC_PATTERN = re.compile(r"\b(\d{1,3}[A-Za-z]?)\b")

def extract_ipc_list(preds):
    """Extract clean IPC section numbers from NER tokens (preds is list of entity dicts)."""
    out = []
    for p in preds:
        token = p.get("word", "")
        m = IPC_PATTERN.search(token)
        if not m:
            continue
        sec = m.group(1)
        num_match = re.match(r"(\d+)", sec)
        if not num_match:
            continue
        n = int(num_match.group(1))
        if 1 <= n <= 511:
            out.append(sec.upper())
    # deduplicate & sort
    return sorted(set(out))


def make_pipeline(model_name=MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE):
    """Load tokenizer+model and create a token-classification pipeline (with GPU if available)."""
    print(f"Loading model {model_name} on device {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
        batch_size=batch_size,
    )
    return pipe


def batched_inference_and_update(df, ipc_pipe, batch_size=BATCH_SIZE, max_chars=MAX_CHARS, only_empty=False, show_progress=True):
    """
    Run batched inference using ipc_pipe over df['text'] and return a new list of ipc_sections (list[str]).
    If only_empty==True, only process rows where existing ipc_sections is empty; others are preserved.
    """
    n = len(df)
    batches = math.ceil(n / batch_size)
    new_ipc = [None] * n  # will fill with lists

    indices_to_process = range(n)
    if only_empty and "ipc_sections" in df.columns:
        indices_to_process = [i for i in range(n) if not df.iloc[i].get("ipc_sections")]
        if show_progress:
            print(f"Processing only {len(indices_to_process)} rows with empty ipc_sections (of {n})")

    it = tqdm(range(0, len(indices_to_process), batch_size)) if show_progress else range(0, len(indices_to_process), batch_size)
    for start_idx_in_list in it:
        # get actual row indices for this mini-batch
        batch_idx_list = indices_to_process[start_idx_in_list:start_idx_in_list+batch_size]
        texts = []
        for idx in batch_idx_list:
            t = df.iloc[idx]["text"]
            texts.append(t[:max_chars] if isinstance(t, str) else "")

        # Run NER pipeline - handle possible OOM by retrying on CPU
        try:
            preds_batch = ipc_pipe(texts)
        except RuntimeError as e:
            print("GPU inference failed, retrying on CPU:", str(e))
            ipc_pipe = make_pipeline(model_name=MODEL_NAME, device=-1, batch_size=1)
            preds_batch = ipc_pipe(texts)

        # preds_batch is a list of lists (one list per input)
        for pos, preds in enumerate(preds_batch):
            idx = batch_idx_list[pos]
            sections = extract_ipc_list(preds)
            new_ipc[idx] = sections

    # For rows we didn't process (if only_empty), preserve existing values
    if only_empty and "ipc_sections" in df.columns:
        for i in range(len(new_ipc)):
            if new_ipc[i] is None:
                existing = df.iloc[i].get("ipc_sections")
                new_ipc[i] = existing if isinstance(existing, list) else []

    # For full run, fill any None with empty list
    new_ipc = [x if isinstance(x, list) else [] for x in new_ipc]
    return new_ipc


def main(input_parquet, output_parquet, batch_size=BATCH_SIZE, only_empty=False):
    print("Loading parquet:", input_parquet)
    df = pd.read_parquet(input_parquet)

    # Normalize existing ipc columns (do not rely on HF datasets - avoid Arrow issues)
    if "ipc_sections" in df.columns:
        df["ipc_sections"] = df["ipc_sections"].apply(lambda x: x if isinstance(x, list) else [])

    # Build pipeline
    ipc_pipe = make_pipeline(device=DEVICE, batch_size=batch_size)

    # Run batched inference (pandas batching to avoid HF Dataset issues)
    start_time = time.time()
    new_ipc_sections = batched_inference_and_update(df, ipc_pipe, batch_size=batch_size, max_chars=MAX_CHARS, only_empty=only_empty, show_progress=True)
    elapsed = time.time() - start_time
    print(f"\nInference finished in {elapsed/60:.2f} minutes")

    # Assign back to dataframe
    df["ipc_sections"] = new_ipc_sections
    df["num_ipc_refs"] = df["ipc_sections"].apply(len)
    df["has_ipc"] = (df["num_ipc_refs"] > 0).astype(int)
    df["ipc_titles"] = df["ipc_sections"].apply(lambda lst: [IPC_MAP.get(x, "Unknown") for x in lst])
    # Keep backward compatibility
    df["ipc_mentions"] = df["ipc_sections"]

    # Save
    print("Saving updated parquet to:", output_parquet)
    df.to_parquet(output_parquet, index=False)
    print("Done. Total IPC cases:", df["has_ipc"].sum())


if __name__ == "__main__":
    # Take the 2 inputs for start and end years
    start = int(input("Enter Starting Year: "))
    end = int(input("Enter Ending Year: "))
    dataset_num = int(input("Enter Dataset number: "))
    parser = argparse.ArgumentParser(description="Batch IPC extraction using dslim/bert-base-NER (pandas batching, GPU)")
    parser.add_argument("--input", "-i", required=False, help="Input enriched parquet path", default=fr"D:\LPA_MTech_Project\Enriched_Datasets\SupremeCourt_Combined_{start}_{end}_enriched.parquet")
    parser.add_argument("--output", "-o", required=False, help="Output parquet path", default=fr"D:\LPA_MTech_Project\Enriched_Datasets\SC_{start}_{end}_enriched_IPC_{dataset_num}.parquet")
    parser.add_argument("--batch_size", "-b", required=False, type=int, default=BATCH_SIZE)
    parser.add_argument("--only_empty", action="store_true", help="Only process rows where ipc_sections is empty")
    args = parser.parse_args()

    input_p = Path(args.input).expanduser()
    output_p = Path(args.output).expanduser()
    main(str(input_p), str(output_p), batch_size=args.batch_size, only_empty=args.only_empty)
