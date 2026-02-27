import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import logging
import random
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, f1_score
import pandas as pd
import argparse
import os
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
import itertools
from openpyxl import load_workbook
from math import ceil, sqrt
import psutil
from pathlib import Path
from contextlib import contextmanager
import inspect

try:  # ====== PyTorch ≥ 2.0 ==================================================
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler
    _new_sig = inspect.signature(_autocast_new)

    @contextmanager
    def autocast(*args, **kwargs):

        if 'device_type' not in kwargs:
            if len(args) == 0:  
                kwargs['device_type'] = 'cuda'
            else:  
                kwargs['device_type'] = 'cuda'
        with _autocast_new(*args, **kwargs):
            yield

except ImportError:  # ====== PyTorch ≤ 1.x ====================================
    from torch.cuda.amp import autocast as _autocast_old
    from torch.cuda.amp import GradScaler

    @contextmanager
    def autocast(*args, **kwargs):
        kwargs.pop('device_type', None)
        with _autocast_old(*args, **kwargs):
            yield
# ---------------------------------------------------------------



try:
    from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv
    from torch_geometric.utils import from_networkx
except ImportError:
    print("Please install torch-geometric: pip install torch-geometric")
    exit(1)

# ========= utils_seed.py =========
def set_global_seed(seed: int):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


   #STGCN', 'ST-GAT', 'ST-GATV2', 'ST-SAGE', 'DCRNN
   # python script.py --datasets=PeMS08 --model_type=ST-GAT --experiment_mode=full_graph --epochs=1 --patience=5 --profile_time 


# Device configuration
scaler = GradScaler()
torch.backends.cudnn.benchmark = True 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaseSTModel(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_features, out_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x, edge_index):
        raise NotImplementedError


class STGCN(BaseSTModel):
    def __init__(self, num_nodes, num_features, hidden_features, out_features, 
                 num_gcn_layers=2, num_gru_layers=2, dropout=0.0):
        super().__init__(num_nodes, num_features, hidden_features, out_features)
        # 1x1 convolution for initial feature extraction
        self.conv1x1 = nn.Conv2d(num_features, hidden_features, kernel_size=(1, 1))
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(hidden_features, hidden_features))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_features, hidden_features))
        # GRU instead of LSTM
        self.gru = nn.GRU(hidden_features, hidden_features, num_gru_layers, 
                         batch_first=True, dropout=dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index):
        try:
            batch_size, seq_len, num_nodes, num_features = x.size()
            # Apply 1x1 convolution
            x = x.view(batch_size, num_nodes, seq_len, num_features).permute(0, 3, 1, 2)
            x = self.conv1x1(x)  # (batch, hidden_features, num_nodes, seq_len)
            x = torch.relu(x)
            x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, self.hidden_features)
            # Apply GCN
            x_gcn = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                for gcn in self.gcn_layers:
                    x_t = gcn(x_t, edge_index)
                    x_t = torch.relu(x_t)
                x_gcn.append(x_t)
            x_stack = torch.stack(x_gcn, dim=1)
            # Apply GRU
            x_gru, _ = self.gru(x_stack)
            x_gru_last = x_gru[:, -1, :]
            # Apply dropout (if any) and FC
            out = self.dropout(x_gru_last)
            out = self.fc(out)
            out = out.reshape(batch_size, num_nodes, self.out_features)
            return out
        except Exception as e:
            logging.error(f"STGCN forward error: {e}")
            raise


class STGAT(BaseSTModel):
    def __init__(self, num_nodes, num_features, hidden_features, out_features, 
                 num_gat_layers=2, num_gru_layers=2, dropout=0.0, heads=2):
        super().__init__(num_nodes, num_features, hidden_features, out_features)
        # 1x1 convolution for initial feature extraction
        self.conv1x1 = nn.Conv2d(num_features, hidden_features, kernel_size=(1, 1))
        # GAT layers with reduced heads
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_features, hidden_features, heads=heads))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(hidden_features * heads, hidden_features, heads=heads))
        # GRU instead of LSTM
        self.gru = nn.GRU(hidden_features * heads, hidden_features, num_gru_layers, 
                         batch_first=True, dropout=dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index):
        try:
            batch_size, seq_len, num_nodes, num_features = x.size()
            # Apply 1x1 convolution
            x = x.view(batch_size, num_nodes, seq_len, num_features).permute(0, 3, 1, 2)
            x = self.conv1x1(x)  # (batch, hidden_features, num_nodes, seq_len)
            x = torch.relu(x)
            x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, self.hidden_features)
            # Apply GAT
            x_gat = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                for gat in self.gat_layers:
                    x_t = gat(x_t, edge_index)
                    x_t = torch.relu(x_t)
                x_gat.append(x_t)
            x_stack = torch.stack(x_gat, dim=1)
            # Apply GRU
            x_gru, _ = self.gru(x_stack)
            x_gru_last = x_gru[:, -1, :]
            # Apply dropout (if any) and FC
            out = self.dropout(x_gru_last)
            out = self.fc(out)
            out = out.reshape(batch_size, num_nodes, self.out_features)
            return out
        except Exception as e:
            logging.error(f"STGAT forward error: {e}")
            raise


class STGATV2(BaseSTModel):
    def __init__(self, num_nodes, num_features, hidden_features, out_features, 
                 num_gat_layers=2, num_gru_layers=2, dropout=0.0, heads=2):
        super().__init__(num_nodes, num_features, hidden_features, out_features)
        # 1x1 convolution for initial feature extraction
        self.conv1x1 = nn.Conv2d(num_features, hidden_features, kernel_size=(1, 1))
        # GATv2 layers with reduced heads
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATv2Conv(hidden_features, hidden_features, heads=heads))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATv2Conv(hidden_features * heads, hidden_features, heads=heads))
        # GRU instead of LSTM
        self.gru = nn.GRU(hidden_features * heads, hidden_features, num_gru_layers, 
                         batch_first=True, dropout=dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index):
        try:
            batch_size, seq_len, num_nodes, num_features = x.size()
            # Apply 1x1 convolution
            x = x.view(batch_size, num_nodes, seq_len, num_features).permute(0, 3, 1, 2)
            x = self.conv1x1(x)  # (batch, hidden_features, num_nodes, seq_len)
            x = torch.relu(x)
            x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, self.hidden_features)
            # Apply GATv2
            x_gat = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                for gat in self.gat_layers:
                    x_t = gat(x_t, edge_index)
                    x_t = torch.relu(x_t)
                x_gat.append(x_t)
            x_stack = torch.stack(x_gat, dim=1)
            # Apply GRU
            x_gru, _ = self.gru(x_stack)
            x_gru_last = x_gru[:, -1, :]
            # Apply dropout (if any) and FC
            out = self.dropout(x_gru_last)
            out = self.fc(out)
            out = out.reshape(batch_size, num_nodes, self.out_features)
            return out
        except Exception as e:
            logging.error(f"STGATV2 forward error: {e}")
            raise

class STSAGE(BaseSTModel):
    def __init__(self, num_nodes, num_features, hidden_features, out_features, 
                 num_sage_layers=2, num_gru_layers=2, dropout=0.0):
        super().__init__(num_nodes, num_features, hidden_features, out_features)
        # 1x1 convolution for initial feature extraction
        self.conv1x1 = nn.Conv2d(num_features, hidden_features, kernel_size=(1, 1))
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(hidden_features, hidden_features))
        for _ in range(num_sage_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_features, hidden_features))
        # GRU instead of LSTM
        self.gru = nn.GRU(hidden_features, hidden_features, num_gru_layers, 
                         batch_first=True, dropout=dropout)
        # Fully connected layer
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index):
        try:
            batch_size, seq_len, num_nodes, num_features = x.size()
            # Apply 1x1 convolution
            x = x.view(batch_size, num_nodes, seq_len, num_features).permute(0, 3, 1, 2)
            x = self.conv1x1(x)  # (batch, hidden_features, num_nodes, seq_len)
            x = torch.relu(x)
            x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, self.hidden_features)
            # Apply GraphSAGE
            x_sage = []
            for t in range(seq_len):
                x_t = x[:, t, :]
                for sage in self.sage_layers:
                    x_t = sage(x_t, edge_index)
                    x_t = torch.relu(x_t)
                x_sage.append(x_t)
            x_stack = torch.stack(x_sage, dim=1)
            # Apply GRU
            x_gru, _ = self.gru(x_stack)
            x_gru_last = x_gru[:, -1, :]
            # Apply dropout (if any) and FC
            out = self.dropout(x_gru_last)
            out = self.fc(out)
            out = out.reshape(batch_size, num_nodes, self.out_features)
            return out
        except Exception as e:
            logging.error(f"STSAGE forward error: {e}")
            raise

class OrigamiGraphPartitioner:
    """Origami Graph Partitioner (OGP) for spatiotemporal graphs."""
    def __init__(self, adj_matrix, features, num_partitions, output_dir='results'):
        self.adj_matrix = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
        self.features = features
        self.num_partitions = num_partitions
        self.output_dir = output_dir
        self.N = self.adj_matrix.shape[0]
        self.T, self.N, self.F = features.shape
        logging.info(f"Initialized OGP: adj_matrix shape {self.adj_matrix.shape}, features shape {self.features.shape}")

    def compute_direction(self):
        try:
            features_np = self.features.cpu().numpy() if isinstance(self.features, torch.Tensor) else self.features
            correlations = np.zeros((self.N, self.N))
            for v in range(self.N):
                for u in range(self.N):
                    if self.adj_matrix[v, u] > 0:
                        corr = np.corrcoef(features_np[:-1, v, 0], 
                                           features_np[1:, u, 0])[0, 1]
                        correlations[v, u] = corr if not np.isnan(corr) else 0
            # Log correlation statistics
            corr_stats = {
                'max': correlations.max(),
                'mean': correlations.mean(),
                'median': np.median(correlations[correlations != 0]) if np.any(correlations != 0) else 0
            }
            logging.info(f"Correlation stats: max={corr_stats['max']:.4f}, mean={corr_stats['mean']:.4f}, median={corr_stats['median']:.4f}")

            # Dynamic threshold based on median + 0.2
            threshold = corr_stats['median'] + 0.2 if corr_stats['median'] > 0 else 0.5
            if correlations.max() > threshold:
                start_node = np.argmax(correlations.sum(axis=1))
                direction = [start_node]
                visited = {start_node}
                while len(direction) < self.N:
                    current = direction[-1]
                    next_node = np.argmax(correlations[current] * 
                                        (1 - np.isin(np.arange(self.N), list(visited))))
                    if correlations[current, next_node] < 0.05 or next_node in visited:
                        break
                    direction.append(next_node)
                    visited.add(next_node)
                remaining = list(set(range(self.N)) - set(direction))
                direction.extend(remaining)
            else:
                G = nx.from_numpy_array(self.adj_matrix)
                try:
                    periphery = nx.periphery(G)
                    if len(periphery) >= 2:
                        path = nx.shortest_path(G, periphery[0], periphery[1])
                    else:
                        path = [periphery[0]] if periphery else [0]
                except:
                    path = list(range(self.N))
                direction = path
                remaining = list(set(range(self.N)) - set(direction))
                direction.extend(remaining)
            logging.info(f"Direction computed: {len(direction)} nodes")
            return direction
        except Exception as e:
            logging.error(f"Compute direction error: {e}")
            raise

    def fold_partition(self):
        try:
            direction = self.compute_direction()
            nodes_per_region = max(1, self.N // self.num_partitions)
            regions = []
            for i in range(self.num_partitions):
                start = i * nodes_per_region
                end = (i + 1) * nodes_per_region if i < self.num_partitions - 1 else len(direction)
                region_nodes = direction[start:end]
                if region_nodes:
                    regions.append(region_nodes)
            while len(regions) < self.num_partitions:
                regions.append([direction[-1]])
            if len(regions) > self.num_partitions:
                regions[self.num_partitions-1].extend([n for r in regions[self.num_partitions:] for n in r])
                regions = regions[:self.num_partitions]

            sub_adjs = []
            for nodes in regions:
                sub_adj = np.zeros_like(self.adj_matrix)
                for u in nodes:
                    for v in nodes:
                        sub_adj[u, v] = self.adj_matrix[u, v]
                sub_adjs.append(sub_adj)

            agg_adj = np.zeros_like(self.adj_matrix)
            for i, nodes_i in enumerate(regions):
                for j, nodes_j in enumerate(regions):
                    if i != j:
                        for u in nodes_i:
                            for v in nodes_j:
                                agg_adj[u, v] = 1 if self.adj_matrix[u, v] > 0 else 0
            logging.info(f"Aggregation matrix: shape {agg_adj.shape}, non-zero {np.sum(agg_adj > 0)}")
            np.save(os.path.join(self.output_dir, 'agg_adj.npy'), agg_adj)
            self.regions = regions
            self.quantify_cut_edges()
            return sub_adjs, agg_adj, self.features, regions
        except Exception as e:
            logging.error(f"Fold partition error: {e}")
            raise

    def quantify_cut_edges(self, model=None, X_test=None, y_test=None, scalers=None):
        try:
            features_np = self.features.cpu().numpy() if isinstance(self.features, torch.Tensor) else self.features
            cut_edges = []
            total_corr = 0
            for u in range(self.N):
                for v in range(self.N):
                    if self.adj_matrix[u, v] > 0:
                        corr = np.corrcoef(features_np[:-1, u, 0], 
                                           features_np[1:, v, 0])[0, 1]
                        corr = corr if not np.isnan(corr) else 0
                        total_corr += corr
                        if any(u in nodes_i and v not in nodes_i for nodes_i in self.regions):
                            cut_edges.append((u, v, corr))
            delta_cut = sum(corr for _, _, corr in cut_edges)
            results = model.evaluate(X_test, y_test, scalers) if model else {
                'MAE': 0.0, 'R2': 0.0, 'TopoSim': 0.0
            }
            df = pd.DataFrame({
                'M': [self.num_partitions],
                'cut_edge_count': [len(cut_edges)],
                'delta_cut': [delta_cut],
                'MAE': [results['MAE']],
                'R2': [results['R2']],
                'TopoSim': [results['TopoSim']]
            })
            df.to_csv(os.path.join(self.output_dir, 'cut_edges.csv'), index=False)
            logging.info(f"Saved cut_edges.csv: {len(cut_edges)} cut edges")
            return cut_edges, delta_cut
        except Exception as e:
            logging.error(f"Cut edge quantification error: {e}")
            raise


class SubgraphAdjuster:
    """Class for subgraph adjustment and unlearning."""
    def __init__(self, adj_matrix, sub_adjs, agg_adj, subgraph_nodes, output_dir='results', seed: int = 0):
        self.adj_matrix = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
        self.sub_adjs = sub_adjs
        max_ganglions = len(sub_adjs) * (len(sub_adjs) - 1) // 2 + len(sub_adjs)
        self.agg_adj = np.zeros((self.adj_matrix.shape[0] + max_ganglions, 
                                 self.adj_matrix.shape[1] + max_ganglions))
        self.agg_adj[:self.adj_matrix.shape[0], :self.adj_matrix.shape[1]] = agg_adj
        self.subgraph_nodes = subgraph_nodes
        self.output_dir = output_dir
        self.N = self.adj_matrix.shape[0]
        self.num_partitions = len(sub_adjs)
        self.unlearn_info = {'nodes': {}, 'edges': {}}
        self.key_nodes = {}
        self.boundary_nodes = []
        self.ganglion_mlps = nn.ModuleList()
        self.ganglion_edges = []
        self.seed = seed
        logging.info(f"Initialized SubgraphAdjuster with {self.num_partitions} subgraphs")

    def locate_unlearn(self, U_N=None, U_E=None):
        self.unlearn_info = {'nodes': {}, 'edges': {}}
        for node in U_N or []:
            subgraph = -1
            is_agg = True
            for j, nodes in enumerate(self.subgraph_nodes):
                if node in nodes:
                    subgraph = j
                    is_agg = False
                    break
            self.unlearn_info['nodes'][node] = {'subgraph': subgraph, 'is_agg': is_agg}
        for u, v in U_E or []:
            subgraph = -1
            is_agg = self.agg_adj[u, v] == 1
            for j, sub_adj in enumerate(self.sub_adjs):
                if sub_adj[u, v] > 0:
                    subgraph = j
                    is_agg = False
                    break
            self.unlearn_info['edges'][(u, v)] = {'subgraph': subgraph, 'is_agg': is_agg}
        logging.info(f"Unlearn info: nodes={len(self.unlearn_info['nodes'])}, edges={len(self.unlearn_info['edges'])}")

    def generate_unlearn(self, ratio, seed=None, unlearn_edges=False):
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        # np.random.seed(seed)
        U_N = []
        U_E = []
        if ratio > 0:
            node_ids = np.arange(self.N)
            num_nodes = int(ratio * self.N)
            U_N = rng.choice(node_ids, size=num_nodes, replace=False).tolist()
            if unlearn_edges:
                edge_indices = np.where(self.adj_matrix > 0)
                edges = list(zip(edge_indices[0], edge_indices[1]))
                num_edges = int(ratio * len(edges))
                U_E = rng.choice(len(edges), size=num_edges, replace=False)
                U_E = [edges[i] for i in U_E]
        self.locate_unlearn(U_N, U_E)
        logging.info(f"Generated unlearn: {len(U_N)} nodes, {len(U_E)} edges")
        return U_N, U_E

    def unlearn_nodes(self):
        for node, info in self.unlearn_info['nodes'].items():
            j = info['subgraph']
            if j >= 0:
                sub_adj = self.sub_adjs[j]
                sub_adj[node, :] = 0
                sub_adj[:, node] = 0
                self.sub_adjs[j] = sub_adj
                np.save(os.path.join(self.output_dir, f'sub_adj_p{j}.npy'), sub_adj)
                self.subgraph_nodes[j] = np.where((sub_adj.sum(axis=1) > 0) | (sub_adj.sum(axis=0) > 0))[0]
                if len(self.subgraph_nodes[j]) < 3 and j > 0:
                    self.merge_subgraphs(j, j-1)
        logging.info(f"Unlearned nodes, updated sub_adjs: {len(self.sub_adjs)} subgraphs")

    def unlearn_agg_adj(self):
        for node, info in self.unlearn_info['nodes'].items():
            if info['is_agg'] or info['subgraph'] >= 0:
                self.agg_adj[node, :] = 0
                self.agg_adj[:, node] = 0
        for (u, v), info in self.unlearn_info['edges'].items():
            if info['is_agg']:
                self.agg_adj[u, v] = 0
        np.save(os.path.join(self.output_dir, 'agg_adj.npy'), self.agg_adj)
        logging.info(f"Updated agg_adj, non-zero: {np.sum(self.agg_adj > 0)}")

    def add_virtual_edges(self):
        for j, sub_adj in enumerate(self.sub_adjs):

            # ---------- 基本信息 ----------
            nodes = np.where((sub_adj.sum(1) > 0) | (sub_adj.sum(0) > 0))[0]
            if len(nodes) <= 1:
                logging.warning(f"Subgraph {j} has {len(nodes)} nodes, skipping connectivity")
                continue
            sub_G = nx.from_numpy_array(sub_adj)
            boundary = [
                n for n in nodes
                if any(self.agg_adj[n, m] > 0 for m in range(self.N) if m not in nodes)
            ]
            if len(boundary) >= 3:
                pos = nx.spring_layout(sub_G.subgraph(nodes), seed=0)
                boundary_sorted = sorted(boundary,
                                         key=lambda n: math.atan2(pos[n][1], pos[n][0]))
                # 顺时针连成环
                for u, v in zip(boundary_sorted,
                                boundary_sorted[1:] + boundary_sorted[:1]):
                    sub_adj[u, v] = sub_adj[v, u] = 1
                logging.info(f"[Ring] sub {j}: added {len(boundary_sorted)} ring edges")
            pr = nx.pagerank(sub_G, alpha=0.85)
            top_nodes = sorted(pr, key=pr.get, reverse=True)[:10]

            deg = sub_adj.sum(1) + sub_adj.sum(0)
            isolated = nodes[deg[nodes] == 0]
            for node in isolated:
                if len(nodes) > 2:
                    nbrs = np.random.choice(top_nodes, size=2, replace=False)
                    sub_adj[node, nbrs] = 1
                    sub_adj[nbrs, node] = 1
            components = list(nx.connected_components(nx.from_numpy_array(sub_adj).subgraph(nodes)))
            if len(components) > 1:
                reps = []
                for comp in components:
                    comp_pr = {n: pr.get(n, 0) for n in comp}
                    reps.append(max(comp_pr, key=comp_pr.get))
                for rep in reps:
                    for hub in top_nodes:
                        if rep != hub:
                            sub_adj[rep, hub] = sub_adj[hub, rep] = 1
                if not nx.is_connected(nx.from_numpy_array(sub_adj).subgraph(nodes)):
                    sec_hubs = sorted(pr, key=pr.get, reverse=True)[10:15]
                    for rep in reps:
                        for hub in sec_hubs:
                            if rep != hub:
                                sub_adj[rep, hub] = sub_adj[hub, rep] = 1
            self.sub_adjs[j] = sub_adj
            np.save(os.path.join(self.output_dir, f"sub_adj_p{j}.npy"), sub_adj)

            sub_G_new = nx.from_numpy_array(sub_adj)
            is_conn   = nx.is_connected(sub_G_new.subgraph(nodes)) if len(nodes) > 1 else False
            logging.info(
                f"Subgraph {j}: ring {len(boundary)} | isolated {len(isolated)} | "
                f"component {len(components)-1} | connected: {is_conn}"
            )
    def find_key_nodes(self):
        """
        选 PageRank top-max(ceil(0.1*|V_i|), 5) 作为 key nodes
        """
        self.key_nodes = {}
        for j, sub_adj in enumerate(self.sub_adjs):
            sub_G = nx.from_numpy_array(sub_adj)
            pr    = nx.pagerank(sub_G, alpha=0.85)
            k     = max(5, math.ceil(0.10 * len(pr)))
            self.key_nodes[j] = sorted(pr, key=pr.get, reverse=True)[:k]
            logging.info(f"[Key] sub {j} | k={k} | nodes={self.key_nodes[j]}")
        return self.key_nodes

    def identify_boundary_nodes(self):
        boundary_nodes = set()
        for u in range(self.N):
            for v in range(self.N):
                if self.agg_adj[u, v] > 0:
                    boundary_nodes.add(u)
                    boundary_nodes.add(v)
        self.boundary_nodes = list(boundary_nodes)
        logging.info(f"Identified {len(self.boundary_nodes)} boundary nodes")
        return self.boundary_nodes




    def add_ganglion_nodes(self):
        self.ganglion_mlps = nn.ModuleList()
        self.ganglion_edges = []
        out_F   = 12    
        gid_base = self.N
        for p in range(self.num_partitions):
            nodes = self.key_nodes[p][:2]
            if len(nodes) < 2:
                nodes = nodes * 2
            mlp = nn.Sequential(
                nn.Linear(2 * out_F, 128), nn.ReLU(),
                nn.Linear(128, 128)
            ).to(device)
            g_id = len(self.ganglion_mlps)
            self.ganglion_mlps.append(mlp)
            for u in nodes:
                self.ganglion_edges.append((u, g_id, "intra"))
                self.agg_adj[u, gid_base + g_id] = 1

        for i, j in itertools.combinations(range(self.num_partitions), 2):
            nodes_i = self.key_nodes[i][:3]
            nodes_j = self.key_nodes[j][:3]
            mlp = nn.Sequential(
                nn.Linear(6 * out_F, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.LayerNorm(128)
            ).to(device)
            g_id = len(self.ganglion_mlps)
            self.ganglion_mlps.append(mlp)
            for u in nodes_i + nodes_j:
                self.ganglion_edges.append((u, g_id, "cross"))
                self.agg_adj[u, gid_base + g_id] = 1
        need = gid_base + len(self.ganglion_mlps)
        if self.agg_adj.shape[0] < need:
            new = np.zeros((need, need))
            new[: self.agg_adj.shape[0], : self.agg_adj.shape[1]] = self.agg_adj
            self.agg_adj = new

        np.save(os.path.join(self.output_dir, "agg_adj.npy"), self.agg_adj)
        logging.info(f"Initialized {len(self.ganglion_mlps)} ganglion MLPs "
                    f"(intra {self.num_partitions}  +  super C={len(self.ganglion_mlps)-self.num_partitions})")
    def add_dense_key_clique(self, k: int = 5):
        key_sets = [nodes[:k] for nodes in self.key_nodes.values()]
        added = 0
        for i in range(len(key_sets)):
            for j in range(i + 1, len(key_sets)):
                for u in key_sets[i]:
                    for v in key_sets[j]:
                        if self.agg_adj[u, v] == 0:
                            self.agg_adj[u, v] = self.agg_adj[v, u] = 1
                            added += 1
        logging.info(f"[Clique] added {added} key-clique edges")

    def add_virtual_agg_edges(self):
        self.agg_adj[:self.N, :self.N] = 0

        part_of = np.full(self.N, -1, dtype=int)
        for idx, nodes in enumerate(self.subgraph_nodes):
            part_of[nodes] = idx

        for u in range(self.N):
            for v in range(self.N):
                if self.adj_matrix[u, v] and part_of[u] != part_of[v]:
                    self.agg_adj[u, v] = self.agg_adj[v, u] = 1

        bnd = self.boundary_nodes            # ≈ 129 for PeMS-08 × 4 part
        for i in range(len(bnd)):
            u, v = bnd[i], bnd[(i + 1) % len(bnd)]
            self.agg_adj[u, v] = self.agg_adj[v, u] = 1
        logging.info(f"[Ring] meta boundary added {len(bnd)} edges")

        self.add_dense_key_clique(k=5)

        for i in range(self.num_partitions):
            for j in range(i + 1, self.num_partitions):
                u = self.key_nodes.get(i, [self.subgraph_nodes[i][0]])[0]
                v = self.key_nodes.get(j, [self.subgraph_nodes[j][0]])[0]
                G_tmp = nx.from_numpy_array(self.agg_adj[:self.N, :self.N])
                if not nx.has_path(G_tmp, u, v):
                    self.agg_adj[u, v] = self.agg_adj[v, u] = 1

        def _force_connect(adj, total_nodes):
            G = nx.from_numpy_array(adj[:total_nodes, :total_nodes])
            comps = list(nx.connected_components(G))
            if len(comps) <= 1:
                return False      # already connected
            main_rep = next(iter(max(comps, key=len)))
            for comp in comps:
                if main_rep in comp:
                    continue
                other = next(iter(comp))
                adj[main_rep, other] = adj[other, main_rep] = 1
            return True

        changed = _force_connect(self.agg_adj, self.N)
        np.save(os.path.join(self.output_dir, 'agg_adj.npy'), self.agg_adj)
        G_final = nx.from_numpy_array(self.agg_adj[:self.N, :self.N])
        logging.info(f"Updated agg_adj: nnz={self.agg_adj[:self.N, :self.N].sum()} | "
                    f"connected={nx.is_connected(G_final)} | "
                    f'force_connect={changed}')

        



    def merge_subgraphs(self, src_idx, dst_idx):
        self.sub_adjs[dst_idx] = self.sub_adjs[dst_idx] + self.sub_adjs[src_idx]
        self.sub_adjs.pop(src_idx)
        self.subgraph_nodes[dst_idx] = np.union1d(self.subgraph_nodes[dst_idx], self.subgraph_nodes[src_idx])
        self.subgraph_nodes.pop(src_idx)
        self.num_partitions -= 1
        np.save(os.path.join(self.output_dir, f'sub_adj_p{dst_idx}.npy'), self.sub_adjs[dst_idx])
        logging.info(f"Merged subgraph {src_idx} into {dst_idx}, new partitions: {self.num_partitions}")


class CrossAttnBlock(nn.Module):
    """2-layer cross-attention Q=ganglion, K/V=token"""
    def __init__(self, d_model: int):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q, kv):
        h,_ = self.mha1(q, kv, kv)
        h = self.ln(h + q)
        out,_ = self.mha2(h, kv, kv)
        return self.ln(out + h)          # (B,N,d)


def scalers_denorm(tensor, scalers_dict):
    assert tensor.shape[-1] == 12, 

    if tensor.dtype in (torch.bfloat16, torch.float16):
        tensor = tensor.float()  
    arr    = tensor.detach().cpu().numpy()
    *front, num_nodes, future = arr.shape
    for n in range(num_nodes):
        arr[..., n, :] = scalers_dict[n].inverse_transform(
            arr[..., n, :].reshape(-1, 1)
        ).reshape(*front, future)
    return torch.from_numpy(arr).to(tensor.device)

class EnhancedGanglionAggregator(nn.Module):
    def __init__(self, num_nodes, hidden_features, out_features,
                 num_partitions, ablation='none',
                 lambda_1=1e-5, lambda_2=1e-5,
                 d_model: int = 128, fusion: str = "ganglion"):
        super().__init__()
        self.fusion_mode = fusion
        self.num_nodes, self.out_features = num_nodes, out_features
        self.d_model = d_model
        self.ablation = ablation
        self.lambda_1, self.lambda_2 = lambda_1, lambda_2
        self.fusion = fusion              # "ganglion" | "token" | "cross"
        self.edge_gate, self.gate_lambda = None, 1e-5
        # ======== ganglion branch ========
        self.g_proj = nn.Linear(out_features, d_model, bias=False)
        self.g_tr   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, 2*d_model,
                                       batch_first=True, dropout=0.03),
            num_layers=3)
        # ======== token branch ========
        self.tok_proj = nn.Linear(1, d_model, bias=True)
        self.token_pe = nn.Parameter(torch.randn(1, 12*num_nodes, d_model))
        self.t_tr     = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, 4*d_model,
                                       batch_first=True, dropout=0.1),
            num_layers=3)
        # ======== fusion ========
        if fusion == "cross":
            self.cross = CrossAttnBlock(d_model)
        self.out_fc = nn.Linear(d_model, out_features)

        # ganglion-MLP projection
        self.ganglion_proj = nn.Linear(128, out_features)


        # ==== Token path =====
        self.tok_proj = nn.Linear(1, d_model, bias=True)
        self.token_pe = nn.Parameter(torch.randn(1, 12 * num_nodes, d_model))
        self.global_tr = nn.TransformerEncoder(      
            nn.TransformerEncoderLayer(
                d_model, 8, 4 * d_model,
                batch_first=True, dropout=0.1),
            num_layers=3)

        # ==== Ganglion-meta path =====
        self.in_proj = nn.Linear(out_features, d_model, bias=False)   
        self.transformer = nn.TransformerEncoder(                     
            nn.TransformerEncoderLayer(
                d_model, 8, 2 * d_model,
                batch_first=True, dropout=0.03),
            num_layers=3)

        # ==== output head =====
        self.fc = nn.Linear(d_model, out_features)    

    def forward(self, emb, key_nodes, boundary_nodes,
                gang_edges, gang_mlps, agg_adj):
        B, N, _ = emb.shape

        tok = emb.permute(0, 2, 1).reshape(B, -1, 1)          # (B,12·N,1)
        tok = self.tok_proj(tok) + self.token_pe[:, :12 * N]   
        tok = self.global_tr(tok)                              # (B,12·N,d)
        tok_out = self.fc(tok)                                 # (B,12·N,12)
        tok_out = tok_out.view(B, 12, N, -1)                   # (B,12,N,12)
        tok_out = tok_out.permute(0, 2, 1, 3).mean(dim=2)      # (B,N,12)
        if self.fusion_mode == "token":
            return tok_out

        self.build_meta_graph(key_nodes, boundary_nodes, gang_edges, agg_adj)

        g_embs  = self.ganglion_activation(emb, gang_edges, gang_mlps)
        emb_enh = self.enhance_embeddings(emb, g_embs, gang_edges)   # (B,N,12)

        mask  = torch.tensor((agg_adj[:N, :N] > 0), device=emb.device)
        if self.edge_gate is None:
            self.edge_gate = nn.Parameter(torch.randn_like(mask.float()) * mask)
        gate  = torch.sigmoid(self.edge_gate) * mask
        adj_w = gate / (gate.sum(-1, keepdim=True) + 1e-6)
        emb_enh = emb_enh + torch.matmul(adj_w, emb_enh)
        self._edge_gate_reg = torch.sum(gate ** 2)

        meta = torch.stack([emb_enh[:, n, :] for n in self.meta_nodes], dim=1)
        meta = self.fc(self.transformer(self.in_proj(meta)))          # (B,|M|,12)

        gang_out = emb.clone()
        for i, n in enumerate(self.meta_nodes):
            if n < N:
                gang_out[:, n, :] = meta[:, i, :]

        if self.fusion_mode == "ganglion":
            return gang_out

        # =====================================================
        # 3) HYBRID (token + ganglion)   ——  element-wise average
        # =====================================================
        return 0.5 * (tok_out + gang_out)



    # =========================================================
    #  REPLACE the whole ganglion_activation() with this block
    # =========================================================
    def ganglion_activation(self,
                            embeddings,            # (B, N, out_F)
                            ganglion_edges,        # List[(u, g_id, type)]
                            ganglion_mlps):        # nn.ModuleList
        try:
            out_F = self.out_features
            B, N, F = embeddings.shape
            assert F == out_F

            # --------- 1. 按 g_id 分组 ---------
            edge_groups = {}
            for u, g, e_type in ganglion_edges:
                edge_groups.setdefault(g, []).append(u)

            g_emb_list = []
            for g_id, mlp in enumerate(ganglion_mlps):
                k_expect = mlp[0].in_features // out_F     # 2 or 6
                nodes = edge_groups.get(g_id, [])[:k_expect]
                if len(nodes) < k_expect:
                    nodes = (nodes + nodes[:1] * k_expect)[:k_expect]
                inp = torch.stack([embeddings[:, n, :] for n in nodes],
                                  dim=1)                     # (B, k, F)

                attn = torch.softmax(
                    torch.matmul(inp, inp.transpose(-1, -2)) / math.sqrt(F),
                    dim=-1
                )
                inp = torch.matmul(attn, inp)                # (B, k, F)

                # --------- 3. MLP ---------
                inp_flat = inp.reshape(B, -1)                # (B, k·F)
                g_emb = mlp(inp_flat)                        # (B, 128)
                g_emb_list.append(g_emb)

            self.ganglion_embeddings = g_emb_list
            return g_emb_list

        except Exception as e:
            logging.error(f"Ganglion activation error: {e}")
            raise


    def enhance_embeddings(self, embeddings, ganglion_embeddings, ganglion_edges):
        """Enhance node embeddings with ganglion contributions using attention."""
        try:
            # Check embeddings dimension
            if embeddings.size(1) != self.num_nodes:
                logging.error(f"Embeddings node dimension {embeddings.size(1)} does not match num_nodes {self.num_nodes}")
                raise ValueError(f"Embeddings node dimension mismatch: {embeddings.size(1)} vs. {self.num_nodes}")
            enhanced = embeddings.clone()  # Shape: (batch, num_nodes, out_features)
            batch_size = embeddings.size(0)
            node_contributions = torch.zeros(batch_size, self.num_nodes, self.out_features, device=device)
            edge_counts = torch.zeros(self.num_nodes, device=device)
            for i, (u, g, edge_type) in enumerate(ganglion_edges):
                if i < len(ganglion_embeddings):
                    ganglion_emb = self.ganglion_proj(ganglion_embeddings[i])  # Shape: (batch, out_features)
                    if u >= self.num_nodes:
                        logging.warning(f"Node index {u} exceeds num_nodes {self.num_nodes}, skipping")
                        continue
                    node_contributions[:, u, :] += ganglion_emb
                    edge_counts[u] += 1
            edge_counts = edge_counts.clamp(min=1)
            for u in range(self.num_nodes):
                if edge_counts[u] > 0:
                    node_contributions[:, u, :] /= edge_counts[u]
            enhanced += node_contributions
            return enhanced
        except Exception as e:
            logging.error(f"Enhance embeddings error: {e}")
            raise

# ======== REPLACE 整个 build_meta_graph =========
    def build_meta_graph(self, key_nodes, boundary_nodes,
                         ganglion_edges, agg_adj):
        try:
            real_nodes = list(range(self.num_nodes))        
            self.meta_nodes = real_nodes                    
            self.meta_edges = []

            rows, cols = np.where(agg_adj[:self.num_nodes, :self.num_nodes] > 0)
            for u, v in zip(rows.tolist(), cols.tolist()):
                self.meta_edges.append((u, v))

            self.meta_edges += [
                (u, g + self.num_nodes)        
                for u, g, _ in ganglion_edges
            ]


            return self.meta_nodes, self.meta_edges
        except Exception as e:
            logging.error(f"Build meta-graph error: {e}")
            raise

    def train_aggregator(
        self,
        embeddings, y_train,
        edge_index,              
        key_nodes, boundary_nodes,
        ganglion_edges, ganglion_mlps, agg_adj,
        *,
        epochs   : int = 60,
        mbatch   : int = 512,
        scalers  = None,
        patience : int = 8,
    ):

        if scalers is not None:
            self.scalers = scalers
        assert hasattr(self, "scalers")

        y_train = y_train.to(embeddings.device, non_blocking=True)

        params = list(self.parameters()) + \
                 [p for mlp in ganglion_mlps for p in mlp.parameters()]
        opt   = optim.AdamW(params, lr=5e-4)
        crit  = nn.L1Loss()

        N = embeddings.size(0)
        order  = torch.randperm(N, device=embeddings.device)
        split  = int(N * 0.8)
        tr_id, va_id = order[:split], order[split:]

        emb_tr, y_tr = embeddings[tr_id], y_train[tr_id]
        emb_va, y_va = embeddings[va_id], y_train[va_id]

        best_va, stall = 1e9, 0
        best_state     = None
        epochs_used    = epochs     

        for ep in range(1, epochs + 1):

            self.train(); [m.train() for m in ganglion_mlps]
            tot_mae_norm = 0.0

            for s in range(0, split, mbatch):
                e = min(split, s + mbatch)
                emb_b, y_b = emb_tr[s:e], y_tr[s:e]

                opt.zero_grad(set_to_none=True)
                with autocast(dtype=torch.bfloat16):
                    pred = self(emb_b, key_nodes, boundary_nodes,
                                ganglion_edges, ganglion_mlps, agg_adj)


                    loss_main = crit(pred, y_b)


                    edge_reg = self.lambda_1 * self._edge_gate_reg
                    gang_reg = self.lambda_2 * sum(
                        torch.norm(h) ** 2 for h in self.ganglion_embeddings)

                    loss = loss_main + edge_reg + gang_reg

                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                tot_mae_norm += loss_main.item() * (e - s)

            train_mae = tot_mae_norm / split

            # -------------------- 验证 --------------------
            self.eval(); [m.eval() for m in ganglion_mlps]
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                pred_va = self(emb_va, key_nodes, boundary_nodes,
                               ganglion_edges, ganglion_mlps, agg_adj)
                val_mae = crit(pred_va, y_va).item()

            logging.info(f"[Agg] ep {ep:>3}/{epochs} – "
                         f"trainMAE(norm) {train_mae:.4f} | "
                         f"valMAE(norm) {val_mae:.4f}")

            # 早停
            if val_mae < best_va - 1e-4:
                best_va, stall = val_mae, 0
                best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
            else:
                stall += 1
                if stall >= patience:
                    logging.info(f"[Agg] early-stop @epoch {ep}")
                    epochs_used = ep        
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return epochs_used        

class STGPC(nn.Module):
    """Spatio-Temporal Graph Paper-Cutting Framework"""
    def __init__(self,
                 num_nodes, num_features, hidden_features, out_features,
                 num_partitions=3,
                 lambda_1=0.005, lambda_2=0.01,
                 output_dir='results',
                 model_type='STGCN',
                 ablation='none',
                 experiment_mode='subgraph_aggregation'):
        super().__init__()

        if experiment_mode == 'full_graph':
            num_partitions = 1       
        self.num_partitions = num_partitions

        self.num_nodes        = num_nodes
        self.hidden_features  = hidden_features
        self.out_features     = out_features
        self.lambda_1         = lambda_1
        self.lambda_2         = lambda_2
        self.output_dir       = output_dir
        self.model_type       = model_type
        self.ablation         = ablation
        self.experiment_mode  = experiment_mode

        model_classes = {
            'STGCN'   : STGCN,
            'ST-GAT'  : STGAT,
            'ST-GATV2': STGATV2,
            'ST-SAGE' : STSAGE,
        }
        SubModel = model_classes[model_type]

        self.subgraph_models = nn.ModuleList([
            SubModel(num_nodes, num_features, hidden_features, out_features)
            for _ in range(self.num_partitions)
        ])


        self.aggregator = (
            EnhancedGanglionAggregator(
                num_nodes, hidden_features, out_features,
                self.num_partitions, ablation,
                lambda_1, lambda_2,
                d_model = getattr(args, "agg_dmodel", 128),
                fusion  = getattr(args, "agg_fusion", "cross")  
            )
            if ablation != 'no_aggregation' and experiment_mode == 'subgraph_aggregation'
            else None
        )

        self.sub_adj_dict       = {}
        self.subgraph_assignments = []
        self.sub_adjs           = []
        self.agg_adj            = None
        self.subgraph_nodes     = []
        self.toposim_values     = []
        self.adjuster           = None
        self.ganglion_edges     = []
        self.ganglion_mlps      = nn.ModuleList()

        logging.info(
            f"Initialized STGPC with {self.num_partitions} subgraph model(s), "
            f"mode={experiment_mode}, backbone={model_type}"
        )


    def compute_toposim(self, adj_matrix):
        try:
            G = nx.from_numpy_array(adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix)
            all_paths = []
            for source, path_dict in nx.all_pairs_shortest_path(G):
                for target, path_list in path_dict.items():
                    all_paths.append(tuple(path_list))
            orig_paths = set(all_paths)
            sub_paths = set()
            for _, nodes in self.subgraph_assignments:
                sub_G = G.subgraph(nodes)
                for source, path_dict in nx.all_pairs_shortest_path(sub_G):
                    for target, path_list in path_dict.items():
                        sub_paths.add(tuple(path_list))
            intersection = len(orig_paths & sub_paths)
            union = len(orig_paths)
            toposim = intersection / union if union > 0 else 0.0
            self.toposim_values.append(toposim)
            toposim_mean = np.mean(self.toposim_values) if self.toposim_values else 0.0
            logging.info(f"TopoSim mean: {toposim_mean:.6f}")
            return toposim_mean
        except Exception as e:
            logging.error(f"TopoSim computation error: {e}")
            raise


    def filter_edges(self, edge_index, nodes):
        try:
            mask = [edge_index[0, i].item() in nodes and edge_index[1, i].item() in nodes 
                    for i in range(edge_index.size(1))]
            sub_edge_index = edge_index[:, mask].clone()
            node_mapping = {old: new for new, old in enumerate(nodes)}
            for j in range(sub_edge_index.size(1)):
                sub_edge_index[0, j] = node_mapping[sub_edge_index[0, j].item()]
                sub_edge_index[1, j] = node_mapping[sub_edge_index[1, j].item()]
            return sub_edge_index.to(device)
        except Exception as e:
            logging.error(f"Edge filtering error: {e}")
            raise

    def forward(self, X, edge_index):

        try:
            batch_size   = X.size(0)


            out_features = 12             
            assert getattr(self, "out_features", 12) == 12, (

            )
            if self.experiment_mode == "full_graph":
                return self.subgraph_models[0](X, edge_index)

            embeddings = torch.zeros(
                batch_size, self.num_nodes, out_features, device=device
            )
            for i, (data, nodes) in enumerate(self.subgraph_assignments):
                sub_edge_index = self.filter_edges(edge_index, nodes)
                X_sub          = X[:, :, nodes]                 # (B,T,|V_sub|,F)
                emb            = self.subgraph_models[i](X_sub, sub_edge_index)
                embeddings[:, nodes, :] = emb   

            if self.aggregator and self.ablation != "no_aggregation":
                key_nodes      = [n for ns in self.adjuster.key_nodes.values() for n in ns]
                boundary_nodes = self.adjuster.boundary_nodes
                return self.aggregator(
                    embeddings,
                    key_nodes, boundary_nodes,
                    self.adjuster.ganglion_edges,
                    self.adjuster.ganglion_mlps,
                    self.adjuster.agg_adj,
                )
            return embeddings

        except Exception as e:
            logging.error(f"Forward error: {e}")
            raise


    def train_subgraphs(
        self, X_train, y_train, X_val, y_val,
        edge_index, adj_matrix,
        epochs=10, batch_size=None, patience=10,
        freeze=True, scalers=None,
    ):
        try:
            stage1_start   = time.time()
            sub_mae_dict   = {}
            epochs_used_sub = []     
            criterion      = nn.L1Loss()

            for i, (_, nodes) in enumerate(self.subgraph_assignments):
                model = self.subgraph_models[i]
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                def lr_lambda(ep):
                    if ep < 5:
                        return (ep + 1) / 5
                    progress = (ep - 5) / max(1, epochs - 5)
                    return 0.5 * (1 + math.cos(math.pi * progress)) * 0.999 + 0.001
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

                X_tr_sub, y_tr_sub = X_train[:, :, nodes], y_train[:, nodes]
                X_va_sub, y_va_sub = X_val[:, :, nodes],   y_val[:, nodes]

                bs = batch_size or args.batch_size
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_tr_sub, y_tr_sub),
                    batch_size=bs, shuffle=True, num_workers=8,
                    pin_memory=True, persistent_workers=True,
                )
                val_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_va_sub, y_va_sub),
                    batch_size=bs, num_workers=4,
                    pin_memory=True, persistent_workers=True,
                )

                best_val_loss, patience_cnt = float("inf"), 0
                epochs_used_this = 0       

                for epoch in range(epochs):
                    epochs_used_this = epoch + 1 
                    if getattr(args, "profile_time", False):
                        torch.cuda.synchronize(device); t0 = time.time()

                    model.train()
                    for Xb_cpu, yb_cpu in train_loader:
                        Xb = Xb_cpu.to(device, non_blocking=True)
                        yb = yb_cpu.to(device, non_blocking=True)
                        sub_edge = self.filter_edges(edge_index, nodes)

                        with autocast(dtype=torch.bfloat16):
                            pred = model(Xb, sub_edge)
                            loss = criterion(pred, yb)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer); scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    scheduler.step()


                    model.eval(); val_loss_sum = 0.0
                    with torch.no_grad():
                        for Xv_cpu, yv_cpu in val_loader:
                            Xv = Xv_cpu.to(device, non_blocking=True)
                            yv = yv_cpu.to(device, non_blocking=True)
                            pred_v = model(Xv, sub_edge)
                            val_loss_sum += criterion(pred_v, yv).item() * Xv.size(0)
                    val_loss = val_loss_sum / len(val_loader.dataset)
                    sub_mae_dict[f"sub_{i}"] = round(val_loss, 4)


                    if val_loss < best_val_loss:
                        best_val_loss, patience_cnt = val_loss, 0
                        torch.save(
                            model.state_dict(),
                            os.path.join(self.output_dir, f"subgraph_p{i}_best.pth"),
                        )
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            logging.info(f"Early-stop sub {i} at epoch {epoch+1}")
                            model.load_state_dict(
                                torch.load(
                                    os.path.join(self.output_dir, f"subgraph_p{i}_best.pth"),
                                    map_location=device,
                                )
                            )
                            break

                    if getattr(args, "profile_time", False):
                        torch.cuda.synchronize(device)
                        if epoch == 0 or (epoch + 1) % 30 == 0 or epoch + 1 == epochs:
                            logging.info(
                                f"Sub {i} | Epoch {epoch+1}/{epochs} finished in {time.time()-t0:.2f}s"
                            )

                epochs_used_sub.append(epochs_used_this)  
                if freeze:
                    for p in model.parameters():
                        p.requires_grad = False
                    logging.info(f"Froze parameters for subgraph {i}")


            self.t_stage1_sec = time.time() - stage1_start


            torch.cuda.synchronize()
            self.gpu_peak_stage1_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()


            pd.DataFrame([{
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": args.seed, **sub_mae_dict
            }]).to_csv(
                os.path.join(self.output_dir, "subgraph_mae.csv"),
                mode="a",
                header=not os.path.exists(os.path.join(self.output_dir, "subgraph_mae.csv")),
                index=False,
            )

            return max(epochs_used_sub)

        except Exception as e:
            logging.error(f"Subgraph training error: {e}")
            raise


 
    def partition_graph(self, adj_matrix, X, raw_features,
                        use_multi_band=True, U_N=None, U_E=None):
        try:
            partitioner = OrigamiGraphPartitioner(
                adj_matrix, raw_features,
                self.num_partitions, self.output_dir
            )
            sub_adjs, agg_adj, _, regions = partitioner.fold_partition()

            self.sub_adj_dict   = {}
            self.subgraph_nodes = []
            first_assignments   = []

            for i, sub_adj in enumerate(sub_adjs):
                nodes = np.where((sub_adj.sum(1) > 0) | (sub_adj.sum(0) > 0))[0]
                if len(nodes) < 3:
                    logging.warning(f"Subgraph {i} has {len(nodes)} nodes, merging…")
                    if i > 0: 
                        first_assignments[i-1][1].extend(nodes)
                        self.subgraph_nodes[i-1].extend(nodes)
                        continue
                sub_G = nx.from_numpy_array(sub_adj)
                data  = from_networkx(sub_G)
                data.node_indices = nodes.tolist()

                first_assignments.append((data, nodes))
                self.subgraph_nodes.append(nodes)
                self.sub_adj_dict[i] = sub_adj
                np.save(os.path.join(self.output_dir, f"sub_adj_p{i}.npy"), sub_adj)

            if len(first_assignments) != self.num_partitions:
                raise ValueError(f"Got {len(first_assignments)} subgraphs, "
                                f"expected {self.num_partitions}")
            self.subgraph_assignments = first_assignments
            self.sub_adjs  = sub_adjs
            self.agg_adj   = agg_adj
            self.adjuster  = SubgraphAdjuster(
                adj_matrix, self.sub_adjs, self.agg_adj,
                self.subgraph_nodes, self.output_dir, seed=args.seed
            )
            self.adjuster.subgraph_assignments = self.subgraph_assignments
            self.adjuster.subgraph_models      = self.subgraph_models
            self.adjuster.filter_edges         = self.filter_edges

            self.adjuster.find_key_nodes()
            self.adjuster.identify_boundary_nodes()
            self.adjuster.add_ganglion_nodes()
            self.adjuster.add_virtual_edges() 
            new_assignments = []
            for i, sub_adj in enumerate(self.sub_adjs):
                nodes = self.subgraph_nodes[i]
                sub_G = nx.from_numpy_array(sub_adj)
                data  = from_networkx(sub_G)
                data.node_indices = nodes.tolist()
                new_assignments.append((data, nodes))
            self.subgraph_assignments = new_assignments  
            self.ganglion_edges = self.adjuster.ganglion_edges
            self.ganglion_mlps  = self.adjuster.ganglion_mlps
            logging.info(f"Updated STGPC: {len(self.ganglion_edges)} ganglion edges, "
                        f"{len(self.ganglion_mlps)} ganglion MLPs")
            self.compute_toposim(adj_matrix)

        except Exception as e:
            logging.error(f"Partitioning error: {e}")
            raise

    def hot_swap_unlearn(self, X, y, edge_index, U_N=None, U_E=None, action=None, hot_swap_rate=0.0):
        try:
            if action == 'unlearn' and (U_N is not None or U_E is not None):
                self.adjuster.locate_unlearn(U_N, U_E)
                self.adjuster.unlearn_nodes()
                self.adjuster.unlearn_agg_adj()
                self.adjuster.add_virtual_edges()
                self.adjuster.find_key_nodes()
                self.adjuster.identify_boundary_nodes()
                self.adjuster.add_ganglion_nodes()
                self.adjuster.add_virtual_agg_edges()
                self.sub_adjs = self.adjuster.sub_adjs
                self.agg_adj = self.adjuster.agg_adj
                self.subgraph_nodes = self.adjuster.subgraph_nodes
                self.num_partitions = self.adjuster.num_partitions
                self.sub_adj_dict = {i: adj for i, adj in enumerate(self.sub_adjs)}
                # Inject required attributes into SubgraphAdjuster
                self.adjuster.subgraph_assignments = self.subgraph_assignments
                self.adjuster.subgraph_models = self.subgraph_models
                self.adjuster.filter_edges = self.filter_edges
                self.ganglion_edges = self.adjuster.ganglion_edges
                self.ganglion_mlps = self.adjuster.ganglion_mlps
                for i, sub_adj in enumerate(self.sub_adjs):
                    np.save(os.path.join(self.output_dir, f'sub_adj_p{i}.npy'), sub_adj)
                logging.info(f"Stored {len(self.sub_adj_dict)} subgraph adjacency matrices")
                logging.info(f"Unlearned nodes/edges, updated to {self.num_partitions} subgraphs")
                logging.info(f"Updated STGPC: {len(self.ganglion_edges)} ganglion edges, {len(self.ganglion_mlps)} ganglion MLPs")
            
            if getattr(self.aggregator, "fusion_mode", "") == "token":
                with torch.no_grad():
                    for n in U_N or []:
                        start = n
                        end   = n + 1
                        self.aggregator.token_pe[0, start:end].uniform_(-0.02, 0.02)


        except Exception as e:
            logging.error(f"Hot-swap/unlearn error: {e}")
            raise

    def visualize_attention(self, output_dir='results'):
        try:
            for i in range(self.num_partitions):
                loss_file = os.path.join(output_dir, f'subgraph_p{i}_losses.csv')
                if os.path.exists(loss_file):
                    df = pd.read_csv(loss_file)
                    plt.plot(df['epoch'], df['train_mae'], label=f'P{i} Train MAE')
                    plt.plot(df['epoch'], df['val_mae'], label=f'P{i} Val MAE')
                    plt.plot(df['epoch'], df['train_r2'], label=f'P{i} Train R²')
                    plt.plot(df['epoch'], df['val_r2'], label=f'P{i} Val R²')
            if self.aggregator:
                attn_weights = self.aggregator.transformer.layers[0].self_attn.out_proj.weight
                plt.figure()
                sns.heatmap(attn_weights.cpu().detach().numpy(), cmap='viridis')
                plt.title('Ganglion Attention Heatmap')
                plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'))
                plt.close()
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'subgraph_error_curves.png'))
            plt.close()
            logging.info(f"Saved visualizations to {output_dir}")
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            raise

    def evaluate_cut_edges(self):
        try:
            df = pd.read_csv(os.path.join(self.output_dir, 'cut_edges.csv'))
            results = self.evaluate(self.X_test, self.y_test, self.scalers, self.edge_index)
            df.loc[df['M'] == self.num_partitions, 'MAE']     = results['MAE']
            df.loc[df['M'] == self.num_partitions, 'R2']      = results['R2']
            df.loc[df['M'] == self.num_partitions, 'TopoSim'] = self.compute_toposim(self.adj_matrix)
            df.to_csv(os.path.join(self.output_dir, 'cut_edges.csv'), index=False)
            logging.info("Updated cut_edges.csv with results")
        except Exception as e:
            logging.error(f"Evaluate cut edges error: {e}")
            raise

    def encode_subgraphs(self, X, edge_index, bs: int = 512):
        try:
            self.eval()
            B_all = X.size(0)
            emb_all = torch.zeros(B_all, self.num_nodes, self.out_features,
                                device=device, dtype=torch.float32)

            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X),
                batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

            ofs = 0
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                for (xb_cpu,) in loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    b  = xb.size(0)
                    dtype_now = torch.get_autocast_dtype('cuda')  
                    emb_batch  = torch.zeros(b, self.num_nodes, self.out_features,
                                            device=device, dtype=dtype_now)

                    for i, (_, nodes) in enumerate(self.subgraph_assignments):
                        if len(nodes) == 0: 
                            continue
                        x_sub  = xb[:, :, nodes]
                        ei_sub = self.filter_edges(edge_index, nodes).to(device)
                        eb     = self.subgraph_models[i](x_sub, ei_sub)  
                        emb_batch[:, nodes, :] = eb                   

                    emb_all[ofs:ofs+b] = emb_batch.float()
                    ofs += b
                    torch.cuda.empty_cache()

            logging.info(f"encode_subgraphs → {emb_all.shape}")
            return emb_all                                                   

        except Exception as e:
            logging.error(f"encode_subgraphs error: {e}")
            raise

    def update_ganglion_models(self, embeddings, y_train, epochs=3):
        """Update ganglion MLPs using provided embeddings."""
        try:
            optimizer = optim.Adam([p for mlp in self.ganglion_mlps for p in mlp.parameters()], lr=0.0001)
            criterion = nn.L1Loss()
            for epoch in range(epochs):
                optimizer.zero_grad()
                for mlp in self.ganglion_mlps:
                    mlp.train()

                key_nodes = [node for nodes in self.adjuster.key_nodes.values() for node in nodes]
                pred = self.aggregator(embeddings, key_nodes, self.adjuster.boundary_nodes, 
                                      self.ganglion_edges, self.ganglion_mlps, self.adjuster.agg_adj)
                loss = criterion(pred, y_train)
                loss.backward()

                optimizer.step()
                logging.info(f"Epoch {epoch+1}, Ganglion MLP loss: {loss.item():.6f}")
        except Exception as e:
            logging.error(f"Ganglion MLP update error: {e}")
            raise

    def run_experiment(self, X_train, y_train, X_val, y_val,
                       X_test,  y_test,  edge_index, adj_matrix,
                       raw_features, scalers, args):
        """
        full_graph / subgraph_only / subgraph_aggregation / stage2_only
        结束时把所有字段写入
          ① results/runs_summary.csv
          ② <output_dir>/run_record.xlsx
        """
        try:

            self.X_test, self.y_test   = X_test, y_test
            self.edge_index            = edge_index
            self.scalers               = scalers
            self.adj_matrix            = adj_matrix
            results        = {}
            t0_global      = time.time()


            if 'partition_m_' in self.ablation:
                self.num_partitions = int(self.ablation.split('_')[-1])


            if getattr(args, "stage2_only", False):
                self.t_stage1_sec = 0.0
                self.partition_graph(adj_matrix, X_train, raw_features)
                for i, _ in enumerate(self.subgraph_assignments):
                    ckpt = os.path.join(args.load_dir, f"subgraph_p{i}_best.pth")
                    self.subgraph_models[i].load_state_dict(
                        torch.load(ckpt, map_location=device))
                    for p in self.subgraph_models[i].parameters():
                        p.requires_grad = False
                logging.info("Sub-graph weights loaded & frozen — Stage-2 starts")

                with torch.no_grad():
                    embed_train = self.encode_subgraphs(
                        X_train, edge_index,
                        bs=getattr(args, "encode_batch_size", 512))

                stage2_start    = time.time()
                key_nodes       = [n for ns in self.adjuster.key_nodes.values() for n in ns]
                agg_epochs_used = self.aggregator.train_aggregator(
                    embeddings     = embed_train,
                    y_train        = y_train,
                    edge_index     = edge_index,        
                    key_nodes      = key_nodes,
                    boundary_nodes = self.adjuster.boundary_nodes,
                    ganglion_edges = self.ganglion_edges,
                    ganglion_mlps  = self.ganglion_mlps,
                    agg_adj        = self.adjuster.agg_adj,
                    scalers        = self.scalers,
                    patience       = args.agg_patience,
                    epochs         = getattr(args, "agg_epochs", args.epochs),
                    mbatch         = getattr(args, "agg_mbatch", 512),
                )
                self.t_stage2_sec   = time.time() - stage2_start
                gpu_peak_stage2_mb = torch.cuda.max_memory_allocated() / 1e6
                cpu_peak_mb        = psutil.Process(os.getpid()).memory_info().rss / 1e6

                results = self.evaluate(X_test, y_test, scalers, edge_index)

                sub_epochs_used         = 0
                self.gpu_peak_stage1_mb = 0.0
                self._finalize_and_save(
                    args, results,
                    sub_epochs_used, agg_epochs_used,
                    gpu_peak_stage2_mb, cpu_peak_mb)
                return results
            if self.experiment_mode == "full_graph":
                self.subgraph_assignments = [(None, list(range(self.num_nodes)))]
                sub_epochs_used = self.train_subgraphs(
                    X_train, y_train, X_val, y_val,
                    edge_index, adj_matrix,
                    epochs=args.epochs, scalers=scalers)
                agg_epochs_used      = 0
                self.t_stage2_sec    = 0.0
                gpu_peak_stage2_mb   = 0.0
                cpu_peak_mb          = psutil.Process(os.getpid()).memory_info().rss / 1e6
                results = self.evaluate(X_test, y_test, scalers, edge_index)

            elif self.experiment_mode == "subgraph_only":
                self.partition_graph(adj_matrix, X_train, raw_features)
                sub_epochs_used = self.train_subgraphs(
                    X_train, y_train, X_val, y_val,
                    edge_index, adj_matrix,
                    epochs=args.epochs, scalers=scalers)
                agg_epochs_used      = 0
                self.t_stage2_sec    = 0.0
                gpu_peak_stage2_mb   = 0.0
                cpu_peak_mb          = psutil.Process(os.getpid()).memory_info().rss / 1e6
                results = self.evaluate(X_test, y_test, scalers, edge_index)

            else:  # ---------------- subgraph_aggregation ----------------
                # ---------- Stage-1 ----------
                self.partition_graph(adj_matrix, X_train, raw_features)
                if getattr(args, "delete_first", False) and \
                        args.unlearn_rate > 0 and self.adjuster is not None:
                    U_N, U_E = self.adjuster.generate_unlearn(
                        args.unlearn_rate, seed=args.seed)
                    self.hot_swap_unlearn(
                        X_train, y_train, edge_index, U_N, U_E, action='unlearn')

                sub_epochs_used = self.train_subgraphs(
                    X_train, y_train, X_val, y_val,
                    edge_index, adj_matrix,
                    epochs=args.epochs, scalers=scalers,
                    freeze=False)
                for sg in self.subgraph_models:
                    for p in sg.parameters():
                        p.requires_grad = False
                    if hasattr(sg, "fc"):
                        for p in sg.fc.parameters():
                            p.requires_grad = True
                    if hasattr(sg, "gat_layers"):
                        for p in sg.gat_layers[-1].parameters():
                            p.requires_grad = True
                    if hasattr(sg, "sage_layers"):
                        for p in sg.sage_layers[-1].parameters():
                            p.requires_grad = True

                # ---------- Stage-2 ----------
                stage2_start = time.time()
                agg_epochs_used = 0
                if self.aggregator and self.ablation != "no_aggregation":
                    with torch.no_grad():
                        embed_train = self.encode_subgraphs(
                            X_train, edge_index,
                            bs=getattr(args, "encode_batch_size", 512))

                    key_nodes = [n for ns in self.adjuster.key_nodes.values()
                                 for n in ns]
                    agg_epochs_used = self.aggregator.train_aggregator(
                        embeddings     = embed_train,
                        y_train        = y_train,
                        edge_index     = edge_index,
                        key_nodes      = key_nodes,
                        boundary_nodes = self.adjuster.boundary_nodes,
                        ganglion_edges = self.ganglion_edges,
                        ganglion_mlps  = self.ganglion_mlps,
                        agg_adj        = self.adjuster.agg_adj,
                        scalers        = self.scalers,
                        patience       = args.agg_patience,
                        epochs         = getattr(args, "agg_epochs", args.epochs),
                        mbatch         = getattr(args, "agg_mbatch", 512),
                    )
                self.t_stage2_sec   = time.time() - stage2_start
                gpu_peak_stage2_mb  = torch.cuda.max_memory_allocated() / 1e6
                cpu_peak_mb         = psutil.Process(os.getpid()).memory_info().rss / 1e6
                results = self.evaluate(X_test, y_test, scalers, edge_index)

            if (not getattr(args, "delete_first", False)) and \
                    args.unlearn_rate > 0 and self.adjuster is not None:
                U_N, U_E = self.adjuster.generate_unlearn(
                    args.unlearn_rate, seed=args.seed)
                self.hot_swap_unlearn(
                    X_train, y_train, edge_index, U_N, U_E, action='unlearn')
                res_unl = self.evaluate(X_test, y_test, scalers, edge_index)
                results.update({f"unlearned_{k}": v for k, v in res_unl.items()})

            if self.adjuster is not None:
                self.evaluate_cut_edges()

            logging.info(
                f"Experiment finished in {time.time()-t0_global:.1f}s ; "
                f"Stage-1 {getattr(self,'t_stage1_sec',0):.1f}s | "
                f"Stage-2 {self.t_stage2_sec:.1f}s")
            self._finalize_and_save(
                args, results,
                sub_epochs_used, agg_epochs_used,
                gpu_peak_stage2_mb, cpu_peak_mb)

            return results

        except Exception as e:
            logging.error(f"Experiment error: {e}")
            raise

    def _finalize_and_save(
        self, args, results,
        sub_epochs_used, agg_epochs_used,
        gpu_peak_stage2_mb, cpu_peak_mb,
    ):
        param_full = 232972                         
        param_sub  = self.num_partitions * 52092      
        param_total = count_trainable(self)        
        param_agg  = param_total - param_sub

        save_run_summary(
            args,
            key        = f"{args.datasets[0]}_{args.model_type}_{args.experiment_mode}_{args.ablation}",
            metrics    = results,
            t_stage1   = getattr(self, "t_stage1_sec", 0.0),
            t_stage2   = self.t_stage2_sec,
            sub_ep     = sub_epochs_used,
            agg_ep     = agg_epochs_used,
            gpu_peak1  = getattr(self, "gpu_peak_stage1_mb", 0.0),
            gpu_peak2  = round(gpu_peak_stage2_mb, 2),
            cpu_peak   = round(cpu_peak_mb, 2),
            param_full = param_full,
            param_sub  = param_sub,
            param_agg  = param_agg,
            param_total= param_total,
        )

    def evaluate(self, X, y, scalers, edge_index, eval_bs: int = None):
        try:
            self.eval()
            eval_bs = eval_bs or getattr(args, "eval_batch_size", 64)

            ds = torch.utils.data.TensorDataset(X, y)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=eval_bs, shuffle=False,
                num_workers=4, pin_memory=True
            )

            preds, gts = [], []
            with torch.no_grad():
                for xb_cpu, yb_cpu in loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    ei = edge_index.to(device, non_blocking=True)

                    # ------------ forward ------------
                    pb = self(xb, ei)              # (bs, N, 12)
                    preds.append(pb.cpu())
                    gts.append(yb_cpu)

            pred_all = torch.cat(preds, dim=0)
            y_all    = torch.cat(gts , dim=0)

            results = evaluate_model(pred_all, y_all, scalers,
                                     model=self, adj_matrix=self.adj_matrix)
            logging.info(f"Final evaluation results: {results}")
            return results

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("OOM even with mini-batch eval，eval_batch_size")
            raise

def load_data(dataset_name,train_size=0.7,val_size=0.15,test_size=0.15,sequence_length=12,future_steps=12,max_rows=None,):

    try:
        dataset_paths = {
            "RWW": "data/RWW",
            "PeMS08": "data/PeMS08",
            "RWW_40": "data/RWW_40",
            "RWW_50": "data/RWW_50",
            "RWW_61": "data/RWW_61",
            "PeMS08_40": "data/PeMS08_40",
            "PeMS08_50": "data/PeMS08_50",
            "PeMS08_61": "data/PeMS08_61",
            "covid": "data/covid",
            "covid_40": "data/covid_40",
            "covid_50": "data/covid_50",
            "covid_61": "data/covid_61",
            "weather_40": "data/weather_40",
            "weather_50": "data/weather_50",
            "weather_61": "data/weather_61",
            "weather": "data/weather",
        }
        adj_path  = f"{dataset_paths[dataset_name]}/adj.csv"
        flow_path = f"{dataset_paths[dataset_name]}/flow.xlsx"

        adj_matrix = pd.read_csv(adj_path, index_col=0).values
        logging.info(f"Loaded {adj_path}: shape {adj_matrix.shape}")

        df = pd.read_excel(flow_path).astype(float)
        if max_rows is not None:
            df = df.iloc[:max_rows]
            logging.info(f"Loaded {flow_path}: limited to {max_rows} rows")
        else:
            logging.info(f"Loaded {flow_path}: shape {df.shape}, full dataset")

        num_nodes = adj_matrix.shape[0]
        assert df.shape[1] == num_nodes, "Adjacency / flow node mismatch"

        scalers, norm = {}, np.zeros_like(df.values)
        for i in range(num_nodes):
            s = MinMaxScaler()
            norm[:, i] = s.fit_transform(df.values[:, i].reshape(-1, 1)).ravel()
            scalers[i] = s
        raw_features = norm.reshape(-1, num_nodes, 1)

        X, y = [], []
        for t in range(len(norm) - sequence_length - future_steps + 1):
            X.append(norm[t : t + sequence_length])
            y.append(norm[t + sequence_length : t + sequence_length + future_steps])
        X = np.array(X).reshape(-1, sequence_length, num_nodes, 1)
        y = np.array(y).reshape(-1, num_nodes, future_steps)

        tot = len(X)
        tr_end = int(tot * train_size)
        va_end = tr_end + int(tot * val_size)

        X_train = torch.from_numpy(X[:tr_end]).float()
        y_train = torch.from_numpy(y[:tr_end]).float()
        X_val   = torch.from_numpy(X[tr_end:va_end]).float()
        y_val   = torch.from_numpy(y[tr_end:va_end]).float()
        X_test  = torch.from_numpy(X[va_end:]).float()
        y_test  = torch.from_numpy(y[va_end:]).float()

        edge_index = torch.LongTensor(adj_matrix).nonzero().t().contiguous()  # CPU

        logging.info(
            f"{dataset_name}: train {X_train.shape[0]}, val {X_val.shape[0]}, "
            f"test {X_test.shape[0]}"
        )
        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            edge_index,
            adj_matrix,
            torch.from_numpy(raw_features).float(),
            scalers,
        )

    except Exception as e:
        logging.error(f"Data loading error: {e}")
        raise



def evaluate_model(predictions, y, scalers, model=None, adj_matrix=None):
    try:
        predictions = predictions.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        batch_size, num_nodes, future_steps = predictions.shape
        pred_denorm = np.zeros_like(predictions)
        y_denorm = np.zeros_like(y)
        for i in range(num_nodes):
            for t in range(future_steps):
                pred_denorm[:, i, t] = scalers[i].inverse_transform(
                    predictions[:, i, t].reshape(-1, 1)).flatten()
                y_denorm[:, i, t] = scalers[i].inverse_transform(
                    y[:, i, t].reshape(-1, 1)).flatten()
        mse = np.mean((pred_denorm - y_denorm) ** 2)
        mae = np.mean(np.abs(pred_denorm - y_denorm))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_denorm.flatten(), pred_denorm.flatten())
        true_trend = np.sign(y_denorm[..., 1:] - y_denorm[..., :-1]).flatten()
        pred_trend = np.sign(pred_denorm[..., 1:] - pred_denorm[..., :-1]).flatten()
        mask = ~np.isnan(true_trend) & ~np.isnan(pred_trend)
        trend_f1 = f1_score(true_trend[mask], pred_trend[mask], average='weighted')
        toposim = model.compute_toposim(adj_matrix) if model and adj_matrix is not None else 0.0
        results = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Trend_F1': trend_f1,
            'TopoSim': toposim
        }
        logging.info(f"Evaluation results: {results}")
        return results
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        raise

def main(args):
    set_global_seed(args.seed)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "experiment.log"),
            level   = logging.INFO,
            format  = "%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Running with args: {vars(args)}")

        results_dict = {}

        for dataset in args.datasets:
            data = load_data(dataset, max_rows=args.data_rows)
            X_train, y_train, X_val, y_val, X_test, y_test, \
            edge_index, adj_matrix, raw_features, scalers = data
        
            args.num_nodes    = X_train.shape[2]
            args.num_features = X_train.shape[3]
            args.out_features = y_train.shape[-1]
            logging.info(
                f"Dataset {dataset}: num_nodes {args.num_nodes}, "
                f"num_features {args.num_features}, out_features {args.out_features}"
            )

            if args.auto_hidden:
                hf_used = math.ceil(args.hidden_ref * math.sqrt(args.num_nodes / args.ref_nodes))

                logging.info(f"[AutoHidden] hidden_features set to {hf_used}")
            else:
                hf_used = args.hidden_features            # CLI 64

            if args.auto_hidden:
                model_cls = {
                    'STGCN'   : STGCN,
                    'ST-GAT'  : STGAT,
                    'ST-GATV2': STGATV2,
                    'ST-SAGE' : STSAGE,
                }[args.model_type]

                baseline_model = model_cls(
                    num_nodes       = args.num_nodes,
                    num_features    = args.num_features,
                    hidden_features = 128,    
                    out_features    = args.out_features,
                ).to(device)

                baseline_params = count_trainable(baseline_model)

                del baseline_model
                torch.cuda.empty_cache()
            else:
                baseline_params = None 

            model = STGPC(
                num_nodes       = args.num_nodes,
                num_features    = args.num_features,
                hidden_features = hf_used,
                out_features    = args.out_features,
                num_partitions  = args.num_partitions,
                lambda_1        = args.lambda_1,
                lambda_2        = args.lambda_2,
                output_dir      = args.output_dir,
                model_type      = args.model_type,
                ablation        = args.ablation,
                experiment_mode = args.experiment_mode,
            ).to(device)
            model = torch.compile(model, mode="reduce-overhead")

            total_params = count_trainable(model)

            if baseline_params is not None:
                logging.info(f"[Param] baseline={baseline_params:,}  ours={total_params:,}")
            else:
                logging.info(f"[Param] ours={total_params:,}")

            param_csv = os.path.join(args.output_dir, "param_count.csv")
            pd.DataFrame([{
                "dataset"         : dataset,
                "model"           : args.model_type,
                "M"               : args.num_partitions,
                "baseline_params" : baseline_params if baseline_params is not None else "-",
                "ours_params"     : total_params,
            }]).to_csv(
                param_csv,
                mode   = "a",
                index  = False,
                header = not os.path.exists(param_csv),
            )

            results = model.run_experiment(*data, args)

            key = f"{dataset}_{args.model_type}_{args.experiment_mode}_{args.ablation}"
            results_dict[key] = results
            logging.info(f"Results for {key}: {results}")

        pd.DataFrame(results_dict).to_excel(
            os.path.join(args.output_dir, "results.xlsx")
        )
        logging.info("Results saved to results.xlsx")

    except Exception as e:
        logging.error(f"Main execution error: {e}")
        raise

def int_or_none(value):
    """Custom argparse type to handle 'None' or integer for --data_rows"""
    if value.lower() == 'none':
        return None
    return int(value)



def save_run_summary(
    args, key, metrics,
    t_stage1, t_stage2,
    sub_ep, agg_ep,
    gpu_peak1, gpu_peak2, cpu_peak,
    param_full, param_sub, param_agg, param_total,
):
    try:
        row = {
            "timestamp"     : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cmd"           : "python " + " ".join(sys.argv[1:]),
            "dataset_model" : key,
            "dataset"       : key.split('_')[0],
            "model"         : args.model_type,
            "partitions"    : args.num_partitions,
            "unlearn_rate"  : args.unlearn_rate,
            "delete_first"  : getattr(args, "delete_first", False),
            "epochs"        : args.epochs,
            "agg_epochs"    : args.agg_epochs,
            "hidden_features": args.hidden_features,
            "seed"          : args.seed,
            "stage1_sec"    : round(t_stage1, 2),
            "stage2_sec"    : round(t_stage2, 2),
            "total_sec"     : round(t_stage1 + t_stage2, 2),
            "sub_epochs_used": sub_ep,
            "agg_epochs_used": agg_ep,
            "gpu_name"      : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_mem_peak_stage1_mb": round(gpu_peak1, 2),
            "gpu_mem_peak_stage2_mb": round(gpu_peak2, 2),
            "cpu_mem_peak_mb":        round(cpu_peak , 2),
            "param_full_graph": param_full,
            "param_subgraphs" : param_sub,
            "param_aggregator": param_agg,
            "param_total"     : param_total,
            **metrics,    
        }

        df = pd.DataFrame([row])
        csv_path = Path("results") / "runs_summary.csv"
        df.to_csv(csv_path,
                  mode="a",
                  index=False,
                  header=not csv_path.exists())
        df.to_excel(Path(args.output_dir) / "run_record.xlsx", index=False)

    except Exception as e:
        logging.error(f"[save_run_summary]：{e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spatiotemporal Graph Experiment')
    parser.add_argument('--agg_epochs', type=int, default=60, help='extra epochs for Aggregator after sub-graphs are frozen')
    parser.add_argument('--datasets', type=str, default='RWW,PeMS08,covid', help='Comma-separated datasets')
    parser.add_argument('--model_type', type=str, default='STGCN', 
                        choices=['STGCN', 'ST-GAT', 'ST-GATV2', 'ST-SAGE', 'DCRNN'])
    parser.add_argument('--experiment_mode', type=str, default='subgraph_aggregation', 
                        choices=['full_graph', 'subgraph_only', 'subgraph_aggregation'])
    parser.add_argument('--ablation', type=str, default='none', 
                        choices=['none', 'no_aggregation', 'no_intra_ganglion', 
                                 'no_ganglion_mlp', 'no_original_edges', 
                                 'partition_m_2', 'partition_m_4', 'partition_m_8'])
    parser.add_argument('--data_rows', type=int_or_none, default=None, help='Number of rows to load (None for full)')
    parser.add_argument('--num_partitions', type=int, default=3)
    parser.add_argument('--hidden_features', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--unlearn_rate', type=float, default=0.05)
    parser.add_argument('--lambda_1', type=float, default=0.0005)
    parser.add_argument('--lambda_2', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--stage2_only', action='store_true', help='skip sub-graph training and only train the aggregator')
    parser.add_argument('--auto_hidden', action='store_true', help='scale hidden so total params ~ full-graph')
    parser.add_argument('--profile_time', action='store_true', help='record wall-clock time of each epoch')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='batch size used during evaluation/inference')
    parser.add_argument('--agg_mbatch', type=int, default=128,
                        help='mini-batch size used in Stage-2 aggregator training')
    parser.add_argument('--agg_dmodel', type=int, default=48,
                        help='d_model of the Transformer inside the aggregator')
    parser.add_argument('--agg_fusion', type=str, default='cross', choices=['ganglion', 'token', 'cross'], help='choose fusion branch inside the aggregator')
    parser.add_argument('--seed', type=int, default=61, help='Random seed for reproducibility')
    parser.add_argument('--hidden_ref', type=int, default=128)
    parser.add_argument('--ref_nodes', type=int, default=170)
    parser.add_argument('--agg_patience', type=int, default=8, help='early-stop patience for the aggregator')
    parser.add_argument('--encode_batch_size', type=int, default=1024, help='mini-batch size when generating sub-graph embeddings')
    parser.add_argument('--delete_first', action='store_true', help='sample & delete nodes *before* any training happens')


    args = parser.parse_args()
    if isinstance(args.datasets, str):     
        args.datasets = args.datasets.split(',')
    args.output_dir = os.path.join('results', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    main(args)
