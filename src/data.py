"""
src/data.py
Dataset ingestion, sequence reconstruction, and leakage-free splitting.
"""

import os
import re
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ProteinGym metadata and wild-type reference configurations
CONFIGS = {
    "TEM1": {
        "name": "TEM-1 Beta-Lactamase",
        "url": "https://huggingface.co/datasets/introvoyz041/ProteinGym/resolve/main/ProteinGym_substitutions/BLAT_ECOLX_Stiffler_2015.csv",
        "csv_filename": "BLAT_ECOLX.csv",
        "pdb_id": "1BTL",
        "pdb_url": "https://files.rcsb.org/download/1BTL.pdb",
        # Full precursor sequence (286 residues, includes 23-residue signal peptide)
        "wt_seq": (
            "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
            "CGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTA"
            "FLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSA"
            "LPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
        ),
        "mature_offset": 23  # Crystal structure 1BTL starts at index 23
    },
    "GFP": {
        "name": "avGFP",
        "url": "https://huggingface.co/datasets/introvoyz041/ProteinGym/resolve/main/ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv",
        "csv_filename": "GFP_AEQVI.csv",
        "pdb_id": "1EMA",
        "pdb_url": "https://files.rcsb.org/download/1EMA.pdb",
        "wt_seq": (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRY"
            "PDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSH"
            "NVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLE"
            "FVTAAGITHGMDELYK"
        ),
        "mature_offset": 0
    }
}


def download_file(url: str, dest_path: str) -> None:
    """Download a file with streaming if it doesn't already exist."""
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading: {os.path.basename(dest_path)}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"Saved: {dest_path}")


def parse_mutation_positions(mutant_str: str) -> list[int]:
    """Extract integer positions from mutations like 'M1A' or 'M1A:G25S'."""
    if pd.isna(mutant_str) or str(mutant_str).strip() in ["WT", "wt", ""]:
        return []
    matches = re.findall(r"\d+", str(mutant_str))
    return [int(m) for m in matches] if matches else []


def load_and_preprocess_dataset(target_key: str, data_dir: str = "./data", max_samples: int = 1500):
    """
    Downloads assay CSV and crystal PDB, standardizes columns, and builds
    Random, Positional, and Depth splits.
    """
    cfg = CONFIGS[target_key]
    csv_path = os.path.join(data_dir, cfg["csv_filename"])
    pdb_path = os.path.join(data_dir, f"{cfg['pdb_id']}.pdb")

    download_file(cfg["url"], csv_path)
    download_file(cfg["pdb_url"], pdb_path)

    df = pd.read_csv(csv_path)

    # Standardize column naming
    rename_dict = {}
    for col in df.columns:
        c_low = col.lower()
        if c_low in ["mutant", "mutants", "mutation", "variant"]:
            rename_dict[col] = "mutant"
        elif c_low in ["dms_score", "score", "fitness"]:
            rename_dict[col] = "DMS_score"
        elif c_low in ["mutated_sequence", "sequence", "mutant_sequence"]:
            rename_dict[col] = "mutated_sequence"
    df.rename(columns=rename_dict, inplace=True)

    # Reconstruct mutated_sequence if absent
    wt_seq = cfg["wt_seq"]
    if "mutated_sequence" not in df.columns:
        print(f"Reconstructing variant sequences for {cfg['name']}...")
        def mutate(m_str):
            seq = list(wt_seq)
            for sub in str(m_str).split(":"):
                match = re.search(r"^([A-Z])(\d+)([A-Z])$", sub.strip())
                if match:
                    pos, mut_aa = int(match.group(2)), match.group(3)
                    if 0 <= pos - 1 < len(seq):
                        seq[pos - 1] = mut_aa
            return "".join(seq)
        df["mutated_sequence"] = df["mutant"].apply(mutate)

    df["extracted_positions"] = df["mutant"].apply(parse_mutation_positions)
    df["num_mutations"] = df["extracted_positions"].apply(len)

    # Subsample if requested
    if max_samples and len(df) > max_samples:
        df = df.iloc[:max_samples].copy().reset_index(drop=True)

    # Construct strict splits
    splits = {}
    all_indices = np.arange(len(df))

    # 1. Random Split (Interpolation)
    idx_tr_rand, idx_te_rand = train_test_split(all_indices, test_size=0.20, random_state=42)
    splits["random"] = (idx_tr_rand, idx_te_rand)

    # 2. Positional Split (Spatial Extrapolation: withhold positions where pos % 5 == 0)
    pos_lists = df["extracted_positions"].tolist()
    idx_tr_pos = [i for i, p in enumerate(pos_lists) if len(p) > 0 and all(pos % 5 != 0 for pos in p)]
    idx_te_pos = [i for i, p in enumerate(pos_lists) if len(p) > 0 and any(pos % 5 == 0 for pos in p)]
    splits["positional"] = (np.array(idx_tr_pos, dtype=np.int64), np.array(idx_te_pos, dtype=np.int64))

    # 3. Mutational Depth Split (Epistatic Extrapolation: train k=1, test k>=2)
    idx_tr_depth = [i for i, k in enumerate(df["num_mutations"]) if k == 1]
    idx_te_depth = [i for i, k in enumerate(df["num_mutations"]) if k >= 2]
    if len(idx_te_depth) >= 25:
        splits["mutational_depth"] = (np.array(idx_tr_depth, dtype=np.int64), np.array(idx_te_depth, dtype=np.int64))
    else:
        print(f"[{cfg['name']}] No multi-mutants found. Mutational depth split skipped.")

    return df, splits, cfg