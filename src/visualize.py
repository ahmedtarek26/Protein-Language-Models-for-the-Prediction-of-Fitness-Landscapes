"""
src/visualize.py
Publication-ready diagnostic visualization scripts:
  1. Master benchmark bar chart across splits.
  2. 2D Mutational Landscape Heatmap (ESM-1v style).
  3. 3D Epistatic Proximity Distribution (Cell Systems style).
  4. Train/Test Data Split Grid Matrix (Cell Systems style).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def plot_benchmark_summary(df_results: pd.DataFrame, output_path: str = "./results/cross_protein_master_diagnostics.png"):
    """Generates comparative multi-panel bar charts across proteins and split protocols."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    proteins = df_results["Protein"].unique()
    fig, axes = plt.subplots(1, len(proteins), figsize=(9 * len(proteins), 5.5), squeeze=False)
    hue_col = "Model" if "Model" in df_results.columns else "Head"

    for idx, prot in enumerate(proteins):
        sub_df = df_results[df_results["Protein"] == prot]
        ax = axes[0, idx]

        barplot = sns.barplot(
            data=sub_df,
            x="Split",
            y="Spearman",
            hue=hue_col,
            palette="viridis",
            edgecolor="black",
            linewidth=0.8,
            ax=ax
        )
        ax.set_title(f"{prot}: Generalization Across Splits", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Model", loc="upper right", fontsize=8.0)

        for p in barplot.patches:
            h = p.get_height()
            if not np.isnan(h) and h > 0.02:
                ax.annotate(
                    f"{h:.2f}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center", va="bottom",
                    fontsize=7.5, xytext=(0, 2),
                    textcoords="offset points"
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Benchmark summary saved to: {output_path}")


def plot_fitness_heatmap(
    df: pd.DataFrame, 
    wt_seq: str, 
    start_pos: int = 60, 
    end_pos: int = 85, 
    output_path: str = "./results/mutational_heatmap.png",
    title: str = "Deep Mutational Scanning Landscape (Variant Effects)"
):
    """
    Renders a publication-grade 2D DMS mutational landscape heatmap (ESM-1v style).
    - x-axis: Sequence position index with wild-type amino acid.
    - y-axis: 20 canonical amino acid substitutions.
    - Markers: Black dots (•) indicate the wild-type residue at each position.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    # 1. Parse all single-mutant records from the DataFrame
    single_mutants = []
    for _, row in df.iterrows():
        mut = str(row["mutant"]).strip()
        if ":" in mut or mut in ["WT", "wt", ""]:
            continue
        match = re.search(r"^([A-Za-z])(\d+)([A-Za-z])$", mut)
        if match:
            wt_aa, pos, mut_aa = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
            single_mutants.append((wt_aa, pos, mut_aa, float(row["DMS_score"])))

    # 2. Check if the specified window has data; auto-adjust if completely empty
    available_positions = [m[1] for m in single_mutants]
    window_positions = [p for p in available_positions if start_pos <= p <= end_pos]
    
    if len(window_positions) == 0 and len(available_positions) > 0:
        print(f"Warning: No single mutants found in window [{start_pos}, {end_pos}].")
        # Auto-center around the densest 25-residue region
        pos_counts = pd.Series(available_positions).value_counts().sort_index()
        densest_pos = pos_counts.rolling(window=25, min_periods=1).sum().idxmax()
        start_pos = max(1, densest_pos - 12)
        end_pos = min(len(wt_seq), start_pos + 25)
        print(f"Automatically adjusting window to region with data: [{start_pos}, {end_pos}].")
    elif len(available_positions) == 0:
        print("Error: No single-point mutations found in this DataFrame.")
        return

    positions = list(range(start_pos, min(end_pos + 1, len(wt_seq) + 1)))
    heatmap_matrix = np.full((len(AMINO_ACIDS), len(positions)), np.nan)

    # 3. Populate matrix with experimental scores
    for wt_aa, pos, mut_aa, score in single_mutants:
        if pos in positions and mut_aa in aa_to_idx:
            p_idx = positions.index(pos)
            a_idx = aa_to_idx[mut_aa]
            heatmap_matrix[a_idx, p_idx] = score

    # 4. Set symmetric color scale centered at 0 (ignoring NaNs)
    valid_scores = heatmap_matrix[~np.isnan(heatmap_matrix)]
    if len(valid_scores) > 0:
        abs_max = np.nanpercentile(np.abs(valid_scores), 98)
        abs_max = max(abs_max, 1.0)
    else:
        abs_max = 1.0

    # 5. Render figure
    fig, ax = plt.subplots(figsize=(max(8.5, len(positions) * 0.35), 6.0), dpi=300)
    ax.set_facecolor("#f0f0f0") # Light gray background for unmeasured/missing variants

    cmap = sns.diverging_palette(240, 10, s=90, l=50, as_cmap=True) # Blue (active) to Red (deleterious)

    sns.heatmap(
        heatmap_matrix,
        cmap=cmap,
        center=0.0,
        vmin=-abs_max,
        vmax=abs_max,
        cbar_kws={"label": "DMS Experimental Fitness Score", "shrink": 0.8},
        yticklabels=AMINO_ACIDS,
        xticklabels=[f"{wt_seq[p-1]}{p}" if 1 <= p <= len(wt_seq) else str(p) for p in positions],
        linewidths=0.5,
        linecolor="white",
        mask=np.isnan(heatmap_matrix),
        ax=ax
    )

    # 6. Add ESM-1v style black dots (•) at the wild-type residue positions
    for p_idx, pos in enumerate(positions):
        if 1 <= pos <= len(wt_seq):
            wt_aa = wt_seq[pos - 1]
            if wt_aa in aa_to_idx:
                a_idx = aa_to_idx[wt_aa]
                ax.text(p_idx + 0.5, a_idx + 0.5, "•", ha="center", va="center", color="black", fontsize=11)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Wild-Type Residue & Sequence Position", fontsize=10, fontweight="bold")
    ax.set_ylabel("Substituted Amino Acid", fontsize=10, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Mutational heatmap saved to: {output_path}")


def plot_epistatic_distance_distribution(df: pd.DataFrame, coords: np.ndarray, 
                                        output_path: str = "./results/epistatic_distance_distribution.png"):
    """
    Renders 3D physical distance distribution of mutations (Cell Systems Fig 1B style).
    Compares pairwise C-alpha distances of all double mutants against high-epistasis pairs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_distances = []
    epistatic_distances = []

    # Filter for double mutants (k = 2)
    double_df = df[df["num_mutations"] == 2].copy()
    if len(double_df) < 10:
        print("Not enough double mutants to plot epistatic distance distribution.")
        return

    # Approximate epistatic deviation: deviation from lower quartile
    median_fitness = double_df["DMS_score"].median()

    for _, row in double_df.iterrows():
        pos = row["extracted_positions"]
        if len(pos) == 2:
            p1, p2 = min(pos[0] - 1, len(coords) - 1), min(pos[1] - 1, len(coords) - 1)
            dist = np.linalg.norm(coords[p1] - coords[p2])
            all_distances.append(dist)
            # Variants showing positive epistasis (retaining high fitness despite 2 mutations)
            if row["DMS_score"] > median_fitness:
                epistatic_distances.append(dist)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    bins = np.linspace(0, 45, 25)

    ax.hist(all_distances, bins=bins, density=True, alpha=0.5, color="#3182bd", edgecolor="black", label="All double mutants")
    ax.hist(epistatic_distances, bins=bins, density=True, alpha=0.6, color="#de2d26", edgecolor="black", label="Positive / Interactive epistasis")

    ax.axvline(8.0, color="black", linestyle="--", linewidth=1.5, label="GeoEpiNet threshold (8 Å)")
    ax.set_title("3D Structural Proximity of Epistatic Mutations (avGFP)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(r"Pairwise $C_\alpha$ Physical Distance ($D_{ij}$ in Å)", fontsize=10)
    ax.set_ylabel("Frequency Density", fontsize=10)
    ax.legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Epistatic distance distribution saved to: {output_path}")


def plot_split_schematic(start_pos: int = 1, end_pos: int = 25, 
                         output_path: str = "./results/data_split_schematic.png"):
    """
    Renders a schematic grid showing Random vs. Positional Splits (Cell Systems Fig 3A/B style).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    positions = list(range(start_pos, end_pos + 1))
    n_pos = len(positions)
    n_aa = 8 # Display subset of amino acids for clarity

    # 1. Random Split Grid (scattered train/test across positions)
    np.random.seed(42)
    rand_grid = np.where(np.random.rand(n_aa, n_pos) > 0.20, 1, 0) # 1 = Train, 0 = Test

    # 2. Positional Split Grid (entire column withheld if pos % 5 == 0)
    pos_grid = np.ones((n_aa, n_pos))
    for j, p in enumerate(positions):
        if p % 5 == 0:
            pos_grid[:, j] = 0 # Held-out test columns

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=300)
    cmap = sns.color_palette(["#3182bd", "#a1d99b"]) # Blue = Test, Green = Train

    for ax, grid, title in zip(axes, [rand_grid, pos_grid], ["(A) Randomized Split", "(B) Positional Split (mod 5 wall)"]):
        sns.heatmap(grid, cmap=cmap, cbar=False, linewidths=0.5, linecolor="white", ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Sequence Position Index", fontsize=9.5)
        ax.set_ylabel("Amino Acid Substitution", fontsize=9.5)
        ax.set_xticks(np.arange(n_pos) + 0.5)
        ax.set_xticklabels(positions, fontsize=7.5)
        ax.set_yticks(np.arange(n_aa) + 0.5)
        ax.set_yticklabels(AMINO_ACIDS[:n_aa], fontsize=8)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#a1d99b", edgecolor="black", label="Training Data"),
        Patch(facecolor="#3182bd", edgecolor="black", label="Test Data (Held-out)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Data split schematic saved to: {output_path}")