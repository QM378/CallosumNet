# CallosumNet

**Unlearning on Spatio-Temporal Graphs through Subgraph Virtual Edge Reconstruction**

CallosumNet is a partition-and-integrate framework for spatio-temporal graph unlearning, biologically inspired by the corpus callosum structure. It partitions a spatio-temporal graph into enhanced subgraphs, stitches them via a lightweight meta-graph, and enables exact (100%) node unlearning by retraining only the affected subgraph—without touching the rest.

> **Paper:** *Unlearning on Spatio-Temporal Graphs through Subgraph Virtual Edge Reconstruction* (Under review at PVLDB 2026)

## Key Features

- **100% Exact Unlearning** — deleted nodes have zero residual influence on model outputs, fully compliant with GDPR/CCPA
- **Sub-linear Unlearning Cost** — retrain only the affected subgraph(s) + lightweight GGB layer (1–3 epochs), not the entire model
- **Near Gold-Model Accuracy** — 3.7% average relative MAE degradation vs. the gold model, compared with 29%–304% for baselines
- **Backbone Agnostic** — supports STGCN, ST-GAT, ST-GATV2, and ST-SAGE
- **Four Real-World Datasets** — RWW (23 nodes), PeMS08 (170 nodes), Global Weather (1,000 nodes), Human Mobility Flow (3,220 nodes)

## Architecture

CallosumNet consists of two core components:

1. **Enhanced Subgraph Construction (ESC)** — partitions the graph along a correlation-driven backbone and inserts virtual ganglion edges + K-Ring boundary connections to preserve local spatio-temporal structure
2. **Global Ganglion Bridging (GGB)** — promotes key/boundary/ganglion nodes into a sparse meta-graph and fuses cross-partition information via a Transformer-based integration layer

## Installation

```bash
# Create environment (tested on Python 3.10)
conda create -n callosumnet python=3.10 -y
conda activate callosumnet

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** PyTorch, PyTorch Geometric, NumPy, SciPy, scikit-learn. See `environment.yml` for the full specification.

## Data Preparation

**PeMS08** is included in the `PeMS08/` directory and ready to use.

For the other three datasets:

| Dataset | Nodes | Time Steps | Source |
|---------|------:|----------:|--------|
| RWW (Sewage Water Depth) | 23 | 18,000 | [HydroNet](https://github.com/QM378/HydroNet) |
| PeMS08 (Traffic Flow) | 170 | 17,856 | [Included](./PeMS08/) |
| Global Weather (Temperature) | 1,000 | 18,000 | [NOAA PSL](https://psl.noaa.gov/) |
| Human Mobility Flow | 3,220 | 3,000 | [Kang et al. 2020](https://github.com/GeoDS/COVID19USFlows) |

Place each dataset folder in the project root directory.

## Quick Start

### Full-Graph Baseline (No Partitioning)

```bash
python callosumnet.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode full_graph \
  --hidden_features 128 \
  --epochs 60 --patience 8 \
  --unlearn_rate 0.00 --seed 40 --profile_time
```

### CallosumNet (ESC + GGB)

```bash
python callosumnet.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.00 --seed 40 --profile_time
```

### Unlearning (10% Node Deletion)

```bash
python callosumnet.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.10 --seed 61 --profile_time
```

## Ablation Studies

```bash
# ESC only (no GGB aggregator)
python callosumnet.py \
  --datasets PeMS08 --model_type STGCN \
  --experiment_mode subgraph_only --num_partitions 4 \
  --hidden_features 128 --epochs 60 --patience 8 \
  --ablation no_aggregation --unlearn_rate 0.00 --seed 41

# GGB only (no intra-subgraph ganglion nodes)
python callosumnet.py \
  --datasets PeMS08 --model_type STGCN \
  --experiment_mode subgraph_aggregation --num_partitions 4 \
  --hidden_features 128 --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 --agg_dmodel 48 \
  --agg_fusion cross --ablation no_intra_ganglion \
  --unlearn_rate 0.00 --seed 42

# Ganglion-path only (no Token-Transformer)
python callosumnet.py \
  --datasets PeMS08 --model_type STGCN \
  --experiment_mode subgraph_aggregation --num_partitions 4 \
  --hidden_features 128 --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 --agg_dmodel 48 \
  --agg_fusion ganglion --ablation none \
  --unlearn_rate 0.00 --seed 44

# No virtual edges
python callosumnet.py \
  --datasets PeMS08 --model_type STGCN \
  --experiment_mode subgraph_aggregation --num_partitions 4 \
  --hidden_features 128 --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 --agg_dmodel 48 \
  --agg_fusion cross --ablation partition_m_4 \
  --lambda_1 0 --lambda_2 0 --unlearn_rate 0.00 --seed 51
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--datasets` | Dataset name (PeMS08, RWW, Weather, Mobility) | PeMS08 |
| `--model_type` | ST-GNN backbone (STGCN, STGAT, STGATv2, STSAGE) | STGCN |
| `--experiment_mode` | `full_graph`, `subgraph_only`, or `subgraph_aggregation` | — |
| `--num_partitions` | Number of ESC subgraphs (M) | 4 |
| `--hidden_features` | Hidden dimension of the ST-GNN backbone | 128 |
| `--epochs` | Training epochs for subgraph models | 60 |
| `--agg_epochs` | Training epochs for GGB layer | 30 |
| `--agg_dmodel` | Transformer hidden dimension in GGB | 48 |
| `--agg_fusion` | Fusion mode: `cross` (full) or `ganglion` (ganglion-only) | cross |
| `--ablation` | Ablation flag (none, no_aggregation, no_intra_ganglion, etc.) | none |
| `--unlearn_rate` | Fraction of nodes to unlearn (0.0–1.0) | 0.0 |
| `--lambda_1` | L1 sparsity regularization on meta-graph | 0.01 |
| `--lambda_2` | L2 regularization on ganglion embeddings | 0.001 |
| `--seed` | Random seed for reproducibility | 40 |
| `--profile_time` | Enable wall-clock timing | False |



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
