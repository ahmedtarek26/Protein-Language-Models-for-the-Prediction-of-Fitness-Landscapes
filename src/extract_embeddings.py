"""
src/extract_embeddings.py
Site-specific delta feature extraction (Δz = z_mut - z_wt) for ESM-2 and ESM3,
plus 3D C-alpha coordinate extraction from PDB structures.
"""

import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Bio.PDB import PDBParser


def extract_esm2_deltas(df, wt_seq: str, cache_path: str, device: torch.device) -> np.ndarray:
    """Computes site-specific delta vectors using ESM-2 (650M)."""
    if os.path.exists(cache_path):
        print(f"Loading cached ESM-2 delta embeddings from {cache_path}")
        return np.load(cache_path)

    from transformers import AutoTokenizer, EsmModel

    print("Computing ESM-2 (650M) site-directed delta vectors...")
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    with torch.no_grad():
        wt_inputs = tokenizer(wt_seq, return_tensors="pt").to(device)
        wt_rep = model(**wt_inputs).last_hidden_state[0]  # Shape: (L+2, 1280)

    sequences = df["mutated_sequence"].tolist()
    embeddings = []
    batch_size = 16

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2 Batches"):
        batch_seqs = sequences[i : i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state

        for b, seq in enumerate(batch_seqs):
            mut_indices = [idx for idx, (w, m) in enumerate(zip(wt_seq, seq)) if w != m]
            if len(mut_indices) > 0:
                delta = torch.zeros(wt_rep.shape[-1], device=device, dtype=torch.float32)
                for idx in mut_indices:
                    token_pos = idx + 1  # Account for leading <cls> token
                    delta += (outputs[b, token_pos].float() - wt_rep[token_pos].float())
                delta /= len(mut_indices)
                embeddings.append(delta.cpu().numpy())
            else:
                embeddings.append(np.zeros(wt_rep.shape[-1], dtype=np.float32))

    X = np.array(embeddings)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, X)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return X


def extract_esm3_deltas(df, wt_seq: str, pdb_path: str, offset: int, cache_path: str, device: torch.device) -> np.ndarray:
    """Computes multimodal structure-conditioned delta vectors using ESM3 (1.4B)."""
    if os.path.exists(cache_path):
        print(f"Loading cached ESM3 delta embeddings from {cache_path}")
        return np.load(cache_path)

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    print("Computing ESM3 (1.4B) multimodal structural delta vectors...")
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device).float()
    model.eval()

    L_seq = len(wt_seq)

    try:
        wt_structure = ESMProtein.from_pdb(pdb_path)
        wt_encoded = model.encode(wt_structure)
        pdb_tokens = wt_encoded.structure.cpu()[1:-1]

        full_tokens = torch.full((L_seq,), 4096, dtype=torch.long)
        valid_len = min(len(pdb_tokens), L_seq - offset)
        full_tokens[offset : offset + valid_len] = pdb_tokens[:valid_len]
        st_tensor = F.pad(full_tokens, (1, 1), value=4096).to(device)
        has_structure = True
    except Exception as e:
        print(f"Warning: Could not process PDB structure ({e}). Proceeding sequence-only.")
        st_tensor = None
        has_structure = False

    with torch.inference_mode():
        wt_tokens = model.encode(ESMProtein(sequence=wt_seq)).sequence.unsqueeze(0).to(device)
        fwd_kwargs = {"sequence_tokens": wt_tokens}
        if has_structure and st_tensor is not None:
            fwd_kwargs["structure_tokens"] = st_tensor.unsqueeze(0)
        wt_out = model.forward(**fwd_kwargs)
        wt_rep = (wt_out.embeddings if hasattr(wt_out, "embeddings") else wt_out.sequence_logits)[0].cpu()

    hidden_dim = wt_rep.shape[-1]
    sequences = df["mutated_sequence"].tolist()
    embeddings = []
    batch_size = 2

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM3 Batches"):
        batch_seqs = sequences[i : i + batch_size]
        B = len(batch_seqs)

        seq_t = torch.stack([model.encode(ESMProtein(sequence=s)).sequence for s in batch_seqs]).to(device)
        fwd_kwargs = {"sequence_tokens": seq_t}
        if has_structure and st_tensor is not None:
            fwd_kwargs["structure_tokens"] = st_tensor.unsqueeze(0).expand(B, -1).clone()

        with torch.inference_mode():
            out = model.forward(**fwd_kwargs)
            rep = (out.embeddings if hasattr(out, "embeddings") else out.sequence_logits).cpu()

        for b, seq in enumerate(batch_seqs):
            mut_indices = [idx for idx, (w, m) in enumerate(zip(wt_seq, seq)) if w != m]
            if len(mut_indices) > 0:
                delta = torch.zeros(hidden_dim)
                for idx in mut_indices:
                    token_pos = idx + 1
                    delta += (rep[b, token_pos] - wt_rep[token_pos])
                delta /= len(mut_indices)
                embeddings.append(delta.numpy())
            else:
                embeddings.append(np.zeros(hidden_dim, dtype=np.float32))

    X = np.array(embeddings)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, X)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return X


def extract_ca_coordinates(pdb_path: str, wt_len: int, cache_path: str) -> np.ndarray:
    """Extracts C-alpha 3D Cartesian coordinates from an experimental PDB file."""
    if os.path.exists(cache_path):
        return np.load(cache_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = next(model.get_chains())

    ca_coords = {}
    for residue in chain:
        if "CA" in residue:
            ca_coords[residue.id[1]] = residue["CA"].get_coord()

    coords = np.zeros((wt_len, 3), dtype=np.float32)
    for pos in range(1, wt_len + 1):
        coords[pos - 1] = ca_coords.get(pos, np.random.randn(3))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, coords)
    return coords