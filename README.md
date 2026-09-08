# Protein Language Model Generalization Benchmark (ESM-2 vs. ESM3)

This repository evaluates whether conditioning protein foundation models on 3D crystal structures bridges out-of-distribution generalization gaps on Deep Mutational Scanning (DMS) fitness landscapes.

## Overview
- **Foundation Representations:** Sequence-only (`ESM-2 650M`) vs. Multimodal Structural Tokens (`ESM3 1.4B`).
- **Benchmark Targets:** TEM-1 $\beta$-lactamase (catalytic resistance) and avGFP (optical brightness).
- **Leakage-Free Splits:**
  1. `Random`: Standard in-distribution baseline.
  2. `Positional`: Spatial extrapolation reserving residue positions where `pos % 5 == 0` strictly for test.
  3. `Mutational Depth`: Evaluates epistatic generalizability ($k=1 \to k \ge 2$).

## Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/protein-fitness-plm.git](https://github.com/your-username/protein-fitness-plm.git)
cd protein-fitness-plm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
python main.py --target all --samples 1500