"""
src/visualize.py
Publication-ready diagnostic visualization scripts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_benchmark_summary(df_results: pd.DataFrame, output_path: str = "./results/benchmark_summary.png"):
    """
    Renders comparative performance bar charts across proteins and split protocols.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    proteins = df_results["Protein"].unique()
    fig, axes = plt.subplots(1, len(proteins), figsize=(9 * len(proteins), 5.5), squeeze=False)

    for idx, prot in enumerate(proteins):
        sub_df = df_results[df_results["Protein"] == prot]
        ax = axes[0, idx]
        
        barplot = sns.barplot(
            data=sub_df,
            x="Split",
            y="Spearman",
            hue="Head",
            palette="viridis",
            edgecolor="black",
            linewidth=0.8,
            ax=ax
        )
        ax.set_title(f"{prot}: Monotonic Rank Generalization", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Model", loc="upper right", fontsize=8.5)

        # Value annotations
        for p in barplot.patches:
            h = p.get_height()
            if not np.isnan(h) and h > 0.02:
                ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2.0, h),
                            ha="center", va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Summary visualization saved: {output_path}")