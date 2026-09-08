"""
main.py
Main entrypoint to run data ingestion, embedding extraction, model benchmarking, 
and full publication-grade visualization generation.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np

from src.data import load_and_preprocess_dataset
from src.extract_embeddings import extract_esm2_deltas, extract_esm3_deltas, extract_ca_coordinates
from src.evaluate import run_benchmark
from src.visualize import (
    plot_benchmark_summary,
    plot_fitness_heatmap,
    plot_epistatic_distance_distribution,
    plot_split_schematic
)


def main():
    parser = argparse.ArgumentParser(description="Protein Language Model Generalization Benchmark")
    parser.add_argument("--target", choices=["TEM1", "GFP", "all"], default="all", help="Target assay to evaluate")
    parser.add_argument("--samples", type=int, default=1500, help="Maximum number of variants per assay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on compute device: {device}")

    targets = ["TEM1", "GFP"] if args.target == "all" else [args.target]
    all_results = []
    cached_dfs = {}
    cached_coords = {}
    cached_cfgs = {}

    for tgt in targets:
        print(f"\n{'=' * 65}")
        print(f"  PROCESSING ASSAY TARGET: {tgt}")
        print(f"{'=' * 65}")

        # 1. Ingest Data and Build Splits
        df, splits, cfg = load_and_preprocess_dataset(tgt, max_samples=args.samples)

        # 2. Extract or Load Precomputed Embeddings & 3D Coordinates
        esm2_cache = f"./embeddings/{tgt.lower()}_esm2_delta.npy"
        esm3_cache = f"./embeddings/{tgt.lower()}_esm3_delta.npy"
        coords_cache = f"./embeddings/{tgt.lower()}_coords.npy"
        pdb_path = f"./data/{cfg['pdb_id']}.pdb"

        X_esm2 = extract_esm2_deltas(df, cfg["wt_seq"], esm2_cache, device)
        X_esm3 = extract_esm3_deltas(df, cfg["wt_seq"], pdb_path, cfg["mature_offset"], esm3_cache, device)
        coords = extract_ca_coordinates(pdb_path, len(cfg["wt_seq"]), coords_cache)

        cached_dfs[tgt] = df
        cached_coords[tgt] = coords
        cached_cfgs[tgt] = cfg

        # 3. Train and Benchmark Models
        df_res = run_benchmark(df, splits, X_esm2, X_esm3, coords, cfg["name"], device)
        all_results.append(df_res)

    # 4. Consolidate Results Table
    final_df = pd.concat(all_results, ignore_index=True)
    os.makedirs("./results", exist_ok=True)
    final_df.to_csv("./results/master_benchmark_results.csv", index=False)

    print("\n" + "=" * 80)
    print("                      FINAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print("=" * 80)

    # 5. Generate All Diagnostic Plots
    print("\nRendering publication figures...")
    
    # 1. Master benchmark comparison bar plot
    plot_benchmark_summary(final_df, "./results/cross_protein_master_diagnostics.png")

    # 2. Data split schematic
    plot_split_schematic(start_pos=1, end_pos=25, output_path="./results/data_split_schematic.png")

    # 3. Targeted visualizations for avGFP (chromophore heatmap & 3D epistasis)
    if "GFP" in cached_dfs:
        plot_fitness_heatmap(
            df=cached_dfs["GFP"],
            wt_seq=cached_cfgs["GFP"]["wt_seq"],
            start_pos=60,
            end_pos=85,
            output_path="./results/mutational_heatmap.png"
        )
        plot_epistatic_distance_distribution(
            df=cached_dfs["GFP"],
            coords=cached_coords["GFP"],
            output_path="./results/epistatic_distance_distribution.png"
        )



if __name__ == "__main__":
    main()