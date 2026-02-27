CallosumNet chops a huge spatio-temporal graph into **M tiny, enhanced sub-models** and stitches them back together with a lightweight meta-graph.  
**You** can erase any node (or ten thousand) by retraining **≤ 6 %** of the parameters for **1 – 3 epochs**—accuracy drops ≤ 2 %.




1. Installation


# 1) create a fresh env – tested on Python 3.10
conda create -n callo python=3.10 -y
conda activate callo

# 2) core deps
pip install -r requirements.txt


2 · Folder anatomy
CallosumNet/
├── data
├── run.py             



3.command:



python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 2 \
  --hidden_features 32 \
  --epochs 60  --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.10 --seed 61 --profile_time

python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 48 \
  --epochs 60  --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.10 --seed 61 --profile_time

python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 8 \
  --hidden_features 64 \
  --epochs 60  --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.10 --seed 61 --profile_time

# 0️⃣ FULL-GRAPH (Reference baseline; skip if previously executed)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode full_graph \
  --hidden_features 128 \
  --epochs 60 --patience 8 \
  --unlearn_rate 0.00 --seed 40 --profile_time

# 1️⃣ ESC + GGB (Complete framework: Cross + Token fusion) ← Main experiment (Row A)
python script2.py \
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

# 2️⃣ ESC ONLY (Train subgraphs without aggregator) ← Ablation (Row B)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_only \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 \
  --patience 8 \
  --ablation no_aggregation \
  --unlearn_rate 0.00 --seed 41 --profile_time

# 3️⃣ GGB ONLY (Aggregator without intra-subgraph Ganglion nodes) ← Ablation (Row C)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation no_intra_ganglion \
  --unlearn_rate 0.00 --seed 42 --profile_time

# 5️⃣ Remove Token-Transformer (Ganglion-path only) ← Ablation (Row F)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion ganglion \
  --ablation none \
  --unlearn_rate 0.00 --seed 44 --profile_time

# 6️⃣ Remove Cross-subgraph original edges (Only virtual edges and Ganglion remain) ← Ablation (Row G)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation no_original_edges \
  --unlearn_rate 0.00 --seed 45 --profile_time

# 7️⃣ Lightweight Token Transformer (d_model=32) ← Ablation (Row H)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 32 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.00 --seed 46 --profile_time

# 8️⃣ Reduced Hidden Size (hidden_features=60; parameters reduced by ~75%) ← Ablation (Row I)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 60 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation none \
  --unlearn_rate 0.00 --seed 47 --profile_time




# 🔟 Enable Online Unlearning (10% rate) ← Unlearning Test (Row K)
python script2.py \
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
  --unlearn_rate 0.10 \
  --seed 49 --profile_time

# 1️⃣2️⃣ No Virtual Edges (Test connectivity with virtual edges disabled)
python script2.py \
  --datasets PeMS08 \
  --model_type STGCN \
  --experiment_mode subgraph_aggregation \
  --num_partitions 4 \
  --hidden_features 128 \
  --epochs 60 --agg_epochs 30 \
  --patience 8 --agg_patience 6 \
  --agg_dmodel 48 \
  --agg_fusion cross \
  --ablation partition_m_4 \
  --lambda_1 0 --lambda_2 0 \
  --unlearn_rate 0.00 --seed 51 --profile_time


