"""
main.py
Single entrypoint to run data ingestion, embedding caching, benchmarking, and visualization.
"""

import os
import argparse
import torch
import pandas as pd

from src.data import load_and_preprocess_dataset
from src.extract_embeddings import extract_esm2_deltas, extract_esm3_deltas
from src.evaluate import run_benchmark
from src.visualize import plot_benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="Protein Language Model Generalization Benchmark")
    parser.add_argument("--target", choices=["TEM1", "GFP", "all"], default="all", help="Target protein assay to evaluate")
    parser.add_argument("--samples", type=int, default=1500, help="Max variants to benchmark per assay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running PLM Benchmark on Device: {device} ===")

    targets = ["TEM1", "GFP"] if args.target == "all" else [args.target]
    all_results = []

    for tgt in targets:
        print(f"\n=======================================================")
        print(f"  PROCESSING TARGET: {tgt}")
        print(f"=======================================================")

        # 1. Load Data & Splits
        df, splits, cfg = load_and_preprocess_dataset(tgt, max_samples=args.samples)

        # 2. Extract or Load Cached Embeddings
        esm2_cache = f"./embeddings/{tgt.lower()}_esm2_delta.npy"
        esm3_cache = f"./embeddings/{tgt.lower()}_esm3_delta.npy"
        pdb_path = f"./data/{cfg['pdb_id']}.pdb"

        X_esm2 = extract_esm2_deltas(df, cfg["wt_seq"], esm2_cache, device)
        X_esm3 = extract_esm3_deltas(df, cfg["wt_seq"], pdb_path, cfg["mature_offset"], esm3_cache, device)

        # 3. Benchmark
        df_res = run_benchmark(df, splits, X_esm2, X_esm3, cfg["name"], device)
        all_results.append(df_res)

    # 4. Consolidate and Plot
    final_results = pd.concat(all_results, ignore_index=True)
    os.makedirs("./results", exist_ok=True)
    final_results.to_csv("./results/master_benchmark_results.csv", index=False)

    print("\n" + "=" * 78)
    print("                    FINAL BENCHMARK RESULTS")
    print("=" * 78)
    print(final_results.to_string(index=False))
    print("=" * 78)

    plot_benchmark_summary(final_results)


if __name__ == "__main__":
    main()