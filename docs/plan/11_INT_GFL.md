---
file: 11_INT_GFL.md
title: "Integration: OpenFGL/gfl (federated infrastructure tier-3)"
load_priority: task-load
prerequisites: [01_OVERVIEW.md, 09_INT_HOST_REPO.md]
related: [04_SERVER_CONDENSATION.md, 06_TRAINING_EVAL.md, 03_CLIENT_CONDENSATION.md]
covers_sections: "Part XI §43-55: gfl identification, folder map, 3-tier layout, FedGM mapping, fedcond_qa custom algorithm, free baselines, integration workflow, updated risks/timeline"
project: FedCondGraphRAG
---

# Integration: OpenFGL/gfl (federated infrastructure tier-3)

> **How to use this file.** Đây là **tầng thứ 3** trong integration stack (sau G-Retriever và LinearRAG ở `09_INT_HOST_REPO.md`). Load khi: (1) setup repo lần đầu, (2) implement stage C (server condensation extends FedGM), hoặc (3) chạy baselines federated. Stack 3-tier tóm tắt ở §55.

---

# Part XI — Federated Learning Infrastructure: OpenFGL/gfl Integration

> **Tại sao có Part này.** Repo [`chuotchuilacduong/gfl`](https://github.com/chuotchuilacduong/gfl) thực ra là một fork/extension của [**OpenFGL**](https://github.com/xkLi-Allen/OpenFGL) (Li et al., 2024, arXiv:2408.16288) — một framework benchmark cho federated graph learning với sẵn ≥ 10 thuật toán FL, 34 dataset, 12 GNN model, partitioning (Louvain/Metis), DP infra, và quan trọng nhất: **toàn bộ 4 baseline federated graph condensation** mà DANCE so sánh (FedC4, FedGVD, FedGM, FedGTA). Tích hợp gfl giúp tiết kiệm thêm ~2K LoC cho stage C (server condensation) và **toàn bộ baseline section**, vì các baselines này đã được implement sẵn và đã được sửa lỗi qua nhiều round.
>
> Part này là extension của **Part IX** (Code Integration Strategy). Đọc Part IX trước; Part XI bổ sung một tầng thứ ba sau G-Retriever và LinearRAG.

## 43. Vì sao gfl là right tool

Khi đối chiếu với plan hiện tại:

| Module trong plan | Phải code lại (trước khi biết gfl) | Có sẵn trong gfl? |
|---|---|---|
| Federated partition (Section 9.2) | hash(title) — đơn giản | ✅ **Louvain, Metis, Dirichlet, label-skew** (`openfgl/utils/`) |
| Subgraph FL trainer (Section 21) | Phải code FGLTrainer skeleton | ✅ **`FGLTrainer`** sẵn — `openfgl/flcore/trainer.py` |
| Server-side condensation (Section 16-18) | PGE + gradient matching | ✅ **FedGM** = chính paper Zhang et al. 2025a — có code; reuse logic |
| FedAvg / FedProx / FedSage+ baselines | Phải code 3-5 thuật toán | ✅ Hơn 10 thuật toán FL sẵn — chỉ chọn flag |
| Federated graph condensation baselines | Phải code 4 baselines DANCE Bảng 1 | ✅ **FedC4, FedGVD, FedGM, FedRGD** sẵn |
| GNN backbones (GCN, GAT, GraphSAGE) | Phải copy từ G-Retriever | ✅ 12 GNN models sẵn — gồm cả những cái G-Retriever không có (SGC, MLP, etc.) |
| DP / privacy mechanisms | Defer to v2 | ✅ DP framework sẵn (`--dp_mech`, `--noise_scale`) |
| Communication cost tracking | Phải code | ✅ `--comm_cost` flag sẵn |
| Multi-client simulation | Phải code multiprocessing | ✅ Sẵn |
| Evaluation modes | Phải code | ✅ 4 modes: local-on-local, local-on-global, global-on-local, global-on-global |

→ gfl xử lý **toàn bộ infrastructure layer**. Chúng ta chỉ cần inject Tri-Graph + LinearRAG + DANCE logic.

## 44. Repo identification

Sau khi inspect `main.py`, `config.py` của `chuotchuilacduong/gfl`:

- Path mặc định: `/home/zhanghao/fgl/OpenFGL/openfgl/dataset` → xác nhận đây là OpenFGL fork.
- `from flcore.trainer import FGLTrainer` ↔ OpenFGL upstream `from openfgl.flcore.trainer import FGLTrainer`.
- `supported_fl_algorithm` ở `config.py` chứa: `fedavg`, `fedprox`, `scaffold`, `moon`, `feddc`, `fedproto`, `fedtgp`, `fedpub`, `fedstar`, `fedgta`, `fedtad`, `gcfl_plus`, `fedsage_plus`, `adafgl`, `feddep`, `fggp`, `fgssl`, `fedgl`, `fedhgn3`, **`fedgm`**, **`fedrgd`**, `hyperion`, `fedigl`, `fedlog`, `fedomg`, `fedaux`, `centralized`, **`fedgvd`**, `fedgkg`, **`fedc4`**.
- Có riêng `flcore/fedgm/fedgm_config.py`, `flcore/hyperion/hyperion_config.py` → mỗi algorithm có module độc lập với config riêng.
- Có flag `--method ∈ {GCond, SGDD, DosCond}` → reuse các condensation methods khác làm ablation.
- Có sẵn `--num_global_syn_nodes`, `--server_condense_iters`, `--condense_iters` → **chính xác** match với Section 17 (server synthetic graph) của plan.

> ⚠️ `gfl` không có README chi tiết và không có paper riêng (citation chưa được public). Khi cần documentation chi tiết, **đọc của OpenFGL upstream** (`xkLi-Allen/OpenFGL`) — API tương thích.

## 45. Cấu trúc thư mục gfl (top-level)

```text
gfl/
├── flcore/                 # ← FL algorithms (mỗi sub-folder = 1 thuật toán)
│   ├── trainer.py          # FGLTrainer — orchestrator chung
│   ├── fedavg/
│   ├── fedprox/
│   ├── fedsage_plus/
│   ├── fedgta/
│   ├── fedgm/              # ← FedGM (Zhang 2025a) — REFERENCE cho stage C
│   │   ├── fedgm_config.py
│   │   ├── fedgm_server.py
│   │   ├── fedgm_client.py
│   │   └── ...
│   ├── fedc4/              # ← FedC4 (Chen 2025) — baseline DANCE
│   ├── fedgvd/             # ← FedGVD (Dai 2025) — baseline DANCE
│   ├── fedrgd/             # ← FedRGD — variant của FedGM
│   ├── hyperion/
│   └── ...
├── task/                   # ← Task definitions
│   └── (node_cls, link_pred, node_clust, graph_cls, ...)
├── model/                  # ← GNN backbones (GCN, GAT, SGC, GraphSAGE, MLP, ...)
├── utils/                  # ← Partitioning, seeds, metrics
│   ├── basic_utils.py      # seed_everything
│   └── (partition utils)
├── data/                   # ← Dataset loaders
├── config.py               # ← Giant argparse — toàn bộ args
├── main.py                 # ← Entry point (39 lines)
├── env.yaml                # Conda env
└── requirements.txt
```

API public chính:

```python
from flcore.trainer import FGLTrainer
trainer = FGLTrainer(args)
trainer.train()
```

`args` được parse từ `config.py` với hơn 100 flags. Quan trọng nhất:
- `--fl_algorithm` chọn thuật toán
- `--simulation_mode subgraph_fl_louvain` cho partitioning
- `--num_clients`, `--num_rounds`, `--client_frac`
- `--dataset`, `--task`, `--model`
- `--num_global_syn_nodes`, `--server_condense_iters` cho FedGM family
- `--dp_mech`, `--noise_scale` cho DP

## 46. Layout cập nhật — 3-tier integration

So với Part IX (G-Retriever + LinearRAG), giờ có 3 codebase được merge:

```text
FedCondGraphRAG/                            # fork của G-Retriever (host)
├── src/                                    # G-Retriever core (giữ)
│   ├── dataset/        ...
│   ├── model/          ...
│   └── utils/          ...
├── train.py, inference.py                  # giữ
│
├── fedcond_grag/                           # custom logic (đa số code mới)
│   ├── data/                               # HotpotQA loader, federated split
│   ├── graph_building/                     # Tri-Graph wrap LinearRAG
│   ├── graph_condensation/                 # DANCE re-implementation (Part X)
│   ├── retrieval/                          # LinearRAG retriever wrapper
│   ├── training/                           # stage A/B/D entry scripts
│   ├── external/
│   │   ├── linearrag/                      # vendored from DEEP-PolyU/LinearRAG
│   │   └── gfl/                            # ← NEW: vendored from chuotchuilacduong/gfl
│   │       ├── flcore/                     # FL algorithms (FedAvg, FedGM, FedC4, ...)
│   │       ├── task/                       # task definitions
│   │       ├── model/                      # GNN backbones (extra: SGC, MLP, GraphSAGE)
│   │       ├── utils/                      # partitioning + helpers
│   │       └── config.py                   # giant argparse
│   └── server_condensation/                # CUSTOM — extends fedgm logic
│       ├── fedcond_qa/                     # CUSTOM FL algorithm
│       │   ├── fedcond_qa_config.py
│       │   ├── fedcond_qa_server.py        # extends fedgm_server.py
│       │   ├── fedcond_qa_client.py
│       │   ├── surrogate_tasks.py          # node-type, link-pred (no class label)
│       │   └── __init__.py
│       └── adapters.py                     # bridge gfl ↔ Tri-Graph format
│
├── configs/                                # YAML configs
└── scripts/                                # shell wrappers
```

### 46.1 Server-side condensation: REPLACED bằng "fedcond_qa" trên nền fedgm

Trong Part IX, mục `fedcond_grag/server_condensation/` ban đầu chứa:
- `anchor_gradient.py`, `synthetic_graph.py`, `pge.py`, `gradient_matching.py`

→ **Thay** bằng custom FL algorithm `fedcond_qa/` đăng ký vào registry của gfl. Mỗi file copy từ `flcore/fedgm/` rồi sửa.

## 47. FedGM as reference cho Server Condensation (Stage C)

**FedGM** (Zhang et al. 2025a, "Rethinking Federated Graph Learning: A Data Condensation Perspective", arXiv:2505.02573) là **chính baseline mà DANCE so sánh** trong Bảng 1, và là kiến trúc gần nhất với stage C của chúng ta. Trong gfl, FedGM được implement ở `flcore/fedgm/`. Đọc kỹ trước khi viết stage C.

### 47.1 Mapping FedGM → Stage C của FedCondGraphRAG

| FedGM concept (gfl) | Stage C của chúng ta (Section 16-18) | Action |
|---|---|---|
| Client local subgraph condensation | Client Tri-Graph → C_m (đã làm ở Stage B) | **Skip**: stage B đã xử lý |
| `--num_global_syn_nodes` (server synthetic nodes) | `K_g` (Section 17.1) | Map trực tiếp |
| Server-side condensation iterations (`--server_condense_iters`) | Outer loop của gradient matching | Map trực tiếp |
| Per-client condensation iterations (`--condense_iters`) | Number of ISTA inner iterations (DANCE Algo 4) | Map (đã có ở Section 14) |
| `--method ∈ {GCond, SGDD, DosCond}` | Condensation strategy | Bắt đầu `GCond` (gradient matching cơ bản) |
| FedGM aggregation: weighted sum của client condensed graphs | Anchor gradient aggregation (Section 16.2) | Reuse logic |
| FedGM gradient matching loss | `L_match = 1 - cos(g_global, g_anchor)` (Section 18.1) | Reuse |

### 47.2 Subclass strategy

Tạo `fedcond_qa_server.py` kế thừa từ `FedGMServer`:

```python
# fedcond_grag/server_condensation/fedcond_qa/fedcond_qa_server.py
from fedcond_grag.external.gfl.flcore.fedgm.fedgm_server import FedGMServer

class FedCondQAServer(FedGMServer):
    """Server cho FedCondGraphRAG, kế thừa FedGM."""
    
    def __init__(self, args):
        super().__init__(args)
        # Thêm node_type embedding cho synthetic nodes (Tri-Graph specific)
        self.type_emb = nn.Embedding(3, args.hid_dim)  # 3 = entity/sentence/passage
    
    def init_synthetic_graph(self, anchor_graphs):
        """Override: init X_global theo node_type ratio từ anchors."""
        # Sample types from average anchor distribution
        type_counts = self._aggregate_type_counts(anchor_graphs)
        ratios = type_counts / type_counts.sum()
        n_per_type = (ratios * self.args.num_global_syn_nodes).long()
        # Init mỗi block từ cluster mean của anchor nodes cùng type
        ...
    
    def compute_anchor_gradients(self, anchor_graphs):
        """Override: gradient từ SURROGATE task (no class labels)."""
        # gfl FedGM dùng CE trên class labels.
        # Chúng ta thay bằng node-type prediction + link prediction.
        loss = self._surrogate_task_loss(anchor_graphs)
        return torch.autograd.grad(loss, self.gnn.parameters())
    
    def server_condense_step(self):
        """Override: thêm PGE adjacency learning."""
        # FedGM gốc có A_global cố định hoặc dense.
        # Chúng ta thay bằng PGE (Section 17.2).
        A_global = self.pge(self.X_global)
        # ... gradient matching trên (X_global, A_global)
```

### 47.3 Custom config

Tạo `fedcond_qa_config.py` follow đúng pattern `fedgm_config.py`:

```python
# fedcond_grag/server_condensation/fedcond_qa/fedcond_qa_config.py
config = {
    "method": "GCond",                  # base method
    "op_epoche": 5,                     # outer loop epochs
    "server_condense_iters": 50,        # server condensation iters
    "condense_iters": 50,               # client-side (ISTA inner)
    "local_epochs": 0,                  # KEY DIFFERENCE: no client local training
    "num_global_syn_nodes": 1024,       # K_g
    # PGE-specific
    "pge_hidden": 256,
    "pge_topk": 8,
    # Surrogate task
    "surrogate_type_weight": 1.0,
    "surrogate_link_weight": 0.5,
    "surrogate_align_weight": 0.1,
}
```

### 47.4 Register vào main.py registry

Trong `main.py` của gfl đã có pattern:

```python
fedgm_based_algorithms = ['fedgm', 'fedrgd', 'fedgkg']
if args.fl_algorithm in fedgm_based_algorithms:
    from flcore.fedgm.fedgm_config import config as fedgm_cfg
    ...
```

Mở rộng thành:

```python
fedgm_based_algorithms = ['fedgm', 'fedrgd', 'fedgkg', 'fedcond_qa']  # ADD
if args.fl_algorithm in fedgm_based_algorithms:
    if args.fl_algorithm == 'fedcond_qa':
        from fedcond_grag.server_condensation.fedcond_qa.fedcond_qa_config import config as cfg
    else:
        from flcore.fedgm.fedgm_config import config as cfg
    ...
```

Trong `flcore/__init__.py` hoặc `flcore/trainer.py` (tuỳ vào cách gfl resolve algorithm):

```python
ALGORITHM_REGISTRY = {
    "fedavg": (FedAvgServer, FedAvgClient),
    "fedgm":  (FedGMServer,  FedGMClient),
    "fedcond_qa": (FedCondQAServer, FedCondQAClient),   # ADD
    ...
}
```

## 48. Custom FL Algorithm "fedcond_qa" — what to add

Khi cài đặt `fedcond_qa` đăng ký vào gfl, theo convention OpenFGL mỗi algorithm cần:

```text
fedcond_qa/
├── __init__.py
├── fedcond_qa_config.py          # extra config dict
├── fedcond_qa_server.py          # subclass FedGMServer
├── fedcond_qa_client.py          # subclass FedGMClient
├── fedcond_qa_trainer.py         # optional: nếu cần custom train loop
└── surrogate_tasks.py            # node-type + link-pred loss
```

`fedcond_qa_client.py` quan trọng nhất — đây là nơi inject **Stage B** (DANCE-style condensation):

```python
# fedcond_qa_client.py
from fedcond_grag.external.gfl.flcore.fedgm.fedgm_client import FedGMClient
from fedcond_grag.graph_condensation import client_condensor

class FedCondQAClient(FedGMClient):
    """Client làm Tri-Graph + S-E-P motif + DANCE condensation."""
    
    def __init__(self, args, client_id, local_data):
        super().__init__(args, client_id, local_data)
        # Local data ở đây là Tri-Graph (đã build sẵn từ Stage A)
        self.tri_graph = local_data
    
    def local_train(self):
        """Override: KHÔNG train GNN local. Chỉ run condensation."""
        if self.round % self.args.condense_refresh_every == 0:
            self.condensed_graph = client_condensor.condense(
                tri_graph=self.tri_graph,
                last_round_gnn=self.gnn,
                config=self.args,
            )
        # Return condensed graph (KHÔNG return model delta như FedGM gốc)
        return self.condensed_graph
    
    def upload(self):
        """Override: upload C_m thay vì Δω."""
        return self.condensed_graph     # PyG Data object
```

→ Như vậy, Stage B (Part III §11-15) trở thành **thuần `local_train()` method** của `FedCondQAClient`. FL infrastructure của gfl lo phần round loop, sampling, aggregation gọi.

## 49. Baselines free từ gfl

Quan trọng cho Section 23-24 (Evaluation): các baseline mà DANCE so sánh và mà chúng ta cũng cần so sánh đều có sẵn:

| Baseline | DANCE Bảng 1 | gfl flag | Mục đích |
|---|---|---|---|
| FedAvg | ✅ | `--fl_algorithm fedavg` | Trivial baseline |
| FedSage+ | ✅ | `--fl_algorithm fedsage_plus` | Subgraph-FL canonical |
| FedGTA | ✅ | `--fl_algorithm fedgta` | Topology-aware aggregation |
| GCond | ✅ | `--fl_algorithm fedgm --method GCond --num_clients 1` | Centralized condensation |
| SFGC | ✅ | (gfl chưa list; có thể có ở variant) | Structure-free condensation |
| FedC4 | ✅ | `--fl_algorithm fedc4` | Federated condensation |
| FedGVD | ✅ | `--fl_algorithm fedgvd` | Generative virtual nodes |
| FedGM | ✅ | `--fl_algorithm fedgm` | Our direct comparison |
| DANCE | ✅ | **chưa có** — phải tự code (xem Part X) | Direct comparison |
| **Ours (fedcond_qa)** | — | `--fl_algorithm fedcond_qa` | Đóng góp của chúng ta |

→ Trừ DANCE (cần tự implement theo Part X), tất cả còn lại có thể chạy với 1 dòng lệnh:

```bash
python main.py --fl_algorithm fedavg     --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedsage_plus --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedgta     --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedgm      --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedc4      --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedgvd     --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
python main.py --fl_algorithm fedcond_qa --dataset hotpot_trigraph --num_clients 5 --simulation_mode subgraph_fl_louvain
```

Để các baselines này chạy được trên Tri-Graph, cần một **dataset adapter** thoả mãn interface của `openfgl/task/subgraph_fl/`. Xem §50.4.

## 50. Integration Workflow

### 50.1 Vendor gfl

```bash
# Trong host repo
cd FedCondGraphRAG
git clone https://github.com/chuotchuilacduong/gfl.git /tmp/gfl
mkdir -p fedcond_grag/external/gfl
cp -r /tmp/gfl/{flcore,task,model,utils,config.py,main.py} fedcond_grag/external/gfl/
cp /tmp/gfl/requirements.txt requirements_gfl.txt
```

### 50.2 Rewrite imports

Như đã làm với LinearRAG (§29 Step 1), rewrite import path:

```bash
cd fedcond_grag/external/gfl
find . -name "*.py" -exec sed -i \
  -e 's|from flcore\.|from fedcond_grag.external.gfl.flcore.|g' \
  -e 's|from utils\.|from fedcond_grag.external.gfl.utils.|g' \
  -e 's|from model\.|from fedcond_grag.external.gfl.model.|g' \
  -e 's|from task\.|from fedcond_grag.external.gfl.task.|g' \
  -e 's|from config |from fedcond_grag.external.gfl.config |g' \
  {} +
```

### 50.3 Merge dependencies

`requirements_gfl.txt` của gfl:
```
torch==2.0.1+cu117
torch-geometric==2.6.1
torch-scatter==2.1.2+pt20cu117
torch-sparse==0.6.18+pt20cu117
scipy==1.14.0
numpy==1.26.4
ogb==1.3.6
```

→ Tương thích với G-Retriever (`torch==2.0.1`, `torch-geometric`). Khác nhỏ: `torch-scatter` và `torch-sparse` version. Chọn version higher; test compatibility.

### 50.4 Dataset adapter: Tri-Graph → gfl format

gfl/OpenFGL kỳ vọng dataset là PyG `Data` với `.x`, `.edge_index`, `.y`. Tri-Graph của chúng ta:

```python
# fedcond_grag/external/gfl/task/subgraph_fl/hotpot_trigraph.py
import torch
from torch_geometric.data import Data
from fedcond_grag.graph_building.trigraph_builder import build_trigraph_for_client

def load_hotpot_trigraph(args):
    """
    Load HotpotQA Tri-Graph và trả về list[Data] (1 per client).
    OpenFGL Louvain partitioning được skip vì đã có partition theo title hash.
    """
    clients_data = []
    for m in range(args.num_clients):
        passages = load_client_passages(m)
        tri_graph_pyg = build_trigraph_for_client(passages, cfg=...)
        # node_type as "y" để gfl không complain
        tri_graph_pyg.y = tri_graph_pyg.node_type
        tri_graph_pyg.num_classes = 3
        clients_data.append(tri_graph_pyg)
    return clients_data
```

Register vào gfl dataset loader (xem `openfgl/data/`).

### 50.5 Custom task: condensation_qa

gfl default tasks: `node_cls, link_pred, node_clust, graph_cls, graph_reg`. Thêm task `condensation_qa`:

```python
# fedcond_grag/external/gfl/task/condensation_qa.py
class CondensationQATask:
    """Task không có class labels; loss = surrogate (node-type + link)."""
    
    def __init__(self, args):
        self.args = args
    
    def loss(self, output, batch):
        # surrogate loss thay cho CE
        from fedcond_grag.server_condensation.fedcond_qa.surrogate_tasks import (
            node_type_ce, link_prediction_bce, kl_alignment
        )
        l = (self.args.surrogate_type_weight * node_type_ce(output, batch)
             + self.args.surrogate_link_weight * link_prediction_bce(output, batch)
             + self.args.surrogate_align_weight * kl_alignment(output, batch))
        return l
    
    def evaluate(self, output, batch):
        # Trả về dict metrics (accuracy của node-type prediction làm proxy)
        return {"type_acc": ..., "link_auc": ...}
```

Register vào `flcore/trainer.py` task selector.

### 50.6 Verification

Kịch bản verification gồm 3 bước:

```bash
# Step 1: gfl import works
python -c "from fedcond_grag.external.gfl.flcore.trainer import FGLTrainer; print('OK')"

# Step 2: FedAvg trên Tri-Graph chạy được (sanity baseline)
python -m fedcond_grag.external.gfl.main \
    --fl_algorithm fedavg \
    --dataset hotpot_trigraph \
    --num_clients 5 \
    --num_rounds 3 \
    --simulation_mode subgraph_fl_louvain \
    --task condensation_qa \
    --model gcn

# Step 3: fedcond_qa custom algorithm chạy được
python -m fedcond_grag.external.gfl.main \
    --fl_algorithm fedcond_qa \
    --dataset hotpot_trigraph \
    --num_clients 5 \
    --num_rounds 3 \
    --num_global_syn_nodes 1024 \
    --server_condense_iters 10 \
    --simulation_mode subgraph_fl_louvain
```

Nếu 3 step trên pass, stage B + C đã chạy được trong khung gfl. Stage D (dual-graph prompting với LLM) vẫn chạy tách rời qua `train.py` của G-Retriever (xem Part IX §29 Step 8).

## 51. Cập nhật bảng "thực sự phải viết mới" (đối chiếu Part IX §33)

Sau khi có gfl, ước tính LoC mới giảm thêm:

| Module | Trước khi có gfl | Sau khi có gfl |
|---|---|---|
| Federated partition | ~150 LoC | 0 (dùng Louvain của gfl) |
| FGL trainer skeleton | ~300 LoC | 0 (`FGLTrainer` sẵn) |
| `server_condensation/anchor_gradient.py` | ~200 LoC | ~50 LoC (subclass FedGM) |
| `server_condensation/synthetic_graph.py` | ~200 LoC | ~50 LoC (extend init logic) |
| `server_condensation/pge.py` | ~150 LoC | ~150 LoC (chưa có sẵn, vẫn phải viết) |
| `server_condensation/gradient_matching.py` | ~300 LoC | ~80 LoC (chỉ override loss function) |
| Baselines | ~2K LoC nếu viết 5 baselines | 0 (free từ gfl) |
| **Tổng** | ~2.5K–3.5K LoC mới | **~1K–1.5K LoC mới** |

→ Tiết kiệm thêm khoảng 60% so với plan Part IX. Tổng tiết kiệm so với viết-từ-đầu **80%**.

## 52. Cập nhật Appendix B — relationship matrix

Thêm gfl vào bảng:

| Source | Reused / adapted | Modified or replaced |
|---|---|---|
| G-Retriever | GNN encoders, projection MLP, LLM wrapper, training loop, `textualize_graph()` | Single-encoder → dual-encoder; PCST retrieval → LinearRAG + global graph |
| LinearRAG | Tri-Graph build, entity activation, PPR retrieval | Centralized → federated; wrapped trong `LinearRAGRetriever` façade |
| DANCE (paper, no code) | text bank caching, neighbor gating, chunk selection, self-expressive topology | Label-aware → query-agnostic; class CE → node-type CE + KL alignment; local training → upload C_m |
| **gfl / OpenFGL** | **FGLTrainer, partition utils, FedGM logic, 10+ baseline algorithms, DP infra, evaluation modes, comm-cost tracking** | **Custom algorithm `fedcond_qa` extends `fedgm`; custom task `condensation_qa`; custom dataset `hotpot_trigraph`** |

## 53. Risks specific to gfl integration

| Risk | Mitigation |
|---|---|
| **R8 — gfl không có "no-label" task** | Tự define `CondensationQATask`; node_type làm proxy label |
| **R9 — FedGM/FedC4 trên Tri-Graph mạnh hơn dự đoán** | Tốt cho positioning: nếu fedcond_qa thắng cả FedGM, kết quả mạnh; nếu thua, ablation cho thấy đóng góp của S-E-P motif + DANCE text condensation |
| **R10 — Louvain partition cho Tri-Graph có thể chia rời S-E-P motifs** | Test partition quality bằng motif coverage trước khi train; fallback: partition theo hash(title) như Section 9.2 |
| **R11 — gfl chưa có DANCE; phải tự implement (Part X) trong khung gfl** | Reuse FedGMClient subclass + custom local_train(); xem §48 |
| **R12 — Version conflict torch-scatter/sparse giữa G-Retriever và gfl** | Lock versions; build wheel cho cu118 từ `data.pyg.org` |
| **R13 — OpenFGL khá ít star (41), chưa nổi tiếng → có thể có bug** | Đọc kỹ commit history + issues; preserve `gfl` snapshot version trong submodule SHA |

## 54. Migration timeline cập nhật

Cập nhật Part IV §22 với gfl integration:

| Milestone | Output | Sử dụng gfl? |
|---|---|---|
| M1 — Data setup | HotpotQA processed | ❌ |
| M2 — Tri-Graph builder | Per-client `trigraph.pt` | wrap LinearRAG |
| M3 — Motif selection | Core graphs | ❌ |
| **M4a — gfl vendor + verify** | `fedavg` chạy được trên Tri-Graph | ✅ FREE BASELINE |
| M4 — Text condensation (DANCE) | `t_tilde_v`, `x_fused` | ❌ (theo Part X) |
| M5 — Topology recon (DANCE) | `A_hat` | ❌ (theo Part X) |
| **M6 — Server condensation (`fedcond_qa`)** | Anchor gradients, PGE, gradient matching | ✅ EXTEND FedGM |
| **M6b — Free baselines** | FedSage+, FedGTA, FedGM, FedC4, FedGVD results | ✅ 1 LINE EACH |
| M7 — Dual graph prompting (Stage D) | E2E QA pipeline | ❌ (G-Retriever) |
| M8 — Ablations + analysis | Full eval table | mix |

→ **M4a và M6b** là milestone mới chỉ tồn tại nhờ gfl, giúp có baselines sớm và **giảm risk** đáng kể (nếu fedcond_qa không thắng baseline đơn giản như fedavg, biết ngay từ đầu chứ không phải đến cuối project).

## 55. Tóm tắt 3-tier stack

```text
┌───────────────────────────────────────────────────────────────┐
│              FedCondGraphRAG (custom logic)                    │
│   - Tri-Graph builder, S-E-P motif, DANCE adaptation,          │
│   - Dual graph prompting, evidence/condensed encoders          │
│   ≈ 1K–1.5K LoC mới                                            │
└────────────┬──────────────────────┬────────────────────────────┘
             │                      │
             ▼                      ▼
┌─────────────────────┐  ┌─────────────────────────┐
│  G-Retriever        │  │  LinearRAG              │  ← Section 47 references both
│  - LLM + GNN soft   │  │  - Tri-Graph build       │
│    prompt           │  │  - 2-stage retrieval     │
│  - PCST (unused)    │  │  - PPR + sem bridging    │
│  - LoRA / freeze    │  │                          │
│  ≈ 5K LoC reused    │  │  ≈ 3K LoC reused         │
└─────────────────────┘  └─────────────────────────┘
             │                      
             ▼                      
┌──────────────────────────────────────────────────────┐
│  gfl / OpenFGL (federated infrastructure)             │
│  - FGLTrainer, round loop, client/server abstraction  │
│  - Partitioning (Louvain, Metis, Dirichlet)           │
│  - FedAvg, FedProx, FedSage+, FedGTA, FedGM, FedC4,   │
│    FedGVD, FedRGD (free baselines)                    │
│  - DP, comm cost, evaluation modes                    │
│  ≈ 10K LoC reused                                     │
└──────────────────────────────────────────────────────┘
```

Tổng: ≈ **18K LoC reused** vs **1K–1.5K LoC viết mới** = **92% reuse**.

---

*End of plan.*

---

## See also
- **Layer dưới (G-Retriever + LinearRAG)**: `09_INT_HOST_REPO.md`
- **Stage C spec mà fedcond_qa thực hiện**: `04_SERVER_CONDENSATION.md`
- **DANCE adaptation injected as `FedCondQAClient.local_train()`**: `10_DANCE_REFERENCE.md` (combined với §48 ở file này)
- **Base risks (R1-R7)**: `07_DEBUG_RISKS_ORDER.md`
- **Updated timeline với M4a/M6b**: §54 ở file này
