---
file: 09_INT_HOST_REPO.md
title: "Integration: G-Retriever (host repo) + LinearRAG (\"copy and fix\")"
load_priority: task-load
prerequisites: [01_OVERVIEW.md]
related: [02_DATA_AND_TRIGRAPH.md, 05_INFERENCE_PROMPTING.md, 11_INT_GFL.md]
covers_sections: "Part IX §28-35: Repo layout, step-by-step workflow, file mapping (LinearRAG→ours, G-Retriever→ours), patches, migration checkpoints"
project: FedCondGraphRAG
---

# Integration: G-Retriever (host repo) + LinearRAG ("copy and fix")

> **How to use this file.** Đây là **how-to-integrate** file. Load khi setup repo lần đầu, hoặc khi tham chiếu file mapping. Strategy: fork G-Retriever làm host, vendor LinearRAG vào `fedcond_grag/external/linearrag/`. **Quan trọng**: file này tập trung 2 codebase đầu; **tầng thứ 3 (gfl)** ở `11_INT_GFL.md` và cũng cần được vendor — đọc song song khi setup repo.

---

# Part IX — Code Integration Strategy ("copy and fix")

> **Nguyên tắc cốt lõi**: hạn chế tối đa việc viết lại từ đầu. Chiến lược là **fork G-Retriever làm host repo**, **vendor LinearRAG vào dưới dạng package con**, và chỉ viết module **DANCE-style + Server-side condensation** từ đầu (vì DANCE không có code chính thức và phần server là đóng góp gốc của project này).

## 28. Repository Layout sau khi tích hợp

Sau khi clone và tích hợp, repo của chúng ta trông như sau:

```text
FedCondGraphRAG/                             # = fork của G-Retriever
├── dataset/                                 # giữ nguyên (G-Retriever)
├── figs/                                    # giữ nguyên
├── src/                                     # giữ NGUYÊN của G-Retriever
│   ├── dataset/
│   │   ├── preprocess/
│   │   │   ├── expla_graphs.py              # giữ
│   │   │   ├── scene_graphs.py              # giữ
│   │   │   ├── webqsp.py                    # giữ
│   │   │   └── hotpot.py                    # MỚI (mục 9)
│   │   ├── utils/
│   │   │   └── retrieval.py                 # giữ (PCST – có thể bị thay cho fedcond pipeline)
│   │   ├── __init__.py                      # FIX: register hotpot_fedcond
│   │   ├── expla_graphs.py / scene_graphs.py / webqsp.py
│   │   └── hotpot_fedcond.py                # MỚI – Dataset class cho HotpotQA dual-graph
│   ├── model/
│   │   ├── __init__.py                      # FIX: register dual_graph_llm
│   │   ├── gnn.py                           # giữ (GraphTransformer/GAT/GCN); thêm GNN_c nhẹ
│   │   ├── graph_llm.py                     # giữ – là parent class
│   │   ├── dual_graph_llm.py                # MỚI – subclass với 2 encoder + late fusion
│   │   ├── pt_llm.py / llm.py / inference_llm.py    # giữ
│   ├── utils/                               # giữ toàn bộ (collate, evaluate, ckpt, seed, lr_schedule)
│   └── config.py                            # FIX nhẹ: thêm CLI args cho dual graph + server
├── train.py                                 # FIX nhẹ: thêm route train_server cho stage C
├── inference.py                             # FIX nhẹ: gọi dual_graph_llm
├── run.sh                                   # giữ; thêm scripts mới
│
├── fedcond_grag/                            # ============ PACKAGE MỚI ============
│   ├── external/
│   │   └── linearrag/                       # === vendor toàn bộ src/ của LinearRAG ===
│   │       ├── LinearRAG.py                 # copy nguyên
│   │       ├── config.py                    # copy nguyên (sẽ rename khi import)
│   │       ├── evaluate.py                  # copy (có thể không dùng)
│   │       └── utils.py                     # copy
│   │       └── __init__.py                  # MỚI: re-export public API
│   │
│   ├── data/
│   │   ├── hotpot_loader.py                 # MỚI (mục 9)
│   │   ├── federated_partition.py           # MỚI
│   │   └── corpus_index.py                  # MỚI
│   │
│   ├── graph_building/
│   │   ├── trigraph_builder.py              # WRAP linearrag.LinearRAG.index()
│   │   ├── entity_extractor.py              # WRAP spaCy NER của LinearRAG
│   │   ├── node_encoder.py                  # MỚI – dùng SentenceTransformer
│   │   └── graph_store.py                   # MỚI – save .pt
│   │
│   ├── graph_condensation/                  # === DANCE re-implementation từ paper ===
│   │   ├── motif_core_selector.py           # MỚI (mục 11)
│   │   ├── text_bank.py                     # MỚI (mục 12.1)
│   │   ├── neighbor_gating.py               # MỚI (mục 12.3)
│   │   ├── chunk_selection.py               # MỚI (mục 12.4)
│   │   ├── graph_text_fusion.py             # MỚI (mục 13)
│   │   ├── evidence_prior.py                # MỚI (mục 14.1)
│   │   ├── topology_reconstruction.py       # MỚI (mục 14)
│   │   └── client_condensor.py              # MỚI – orchestrator
│   │
│   ├── server_condensation/                 # === Đóng góp gốc ===
│   │   ├── anchor_gradient.py               # MỚI (mục 16)
│   │   ├── synthetic_graph.py               # MỚI (mục 17)
│   │   ├── pge.py                           # MỚI (mục 17.2)
│   │   └── gradient_matching.py             # MỚI (mục 18)
│   │
│   ├── retrieval/
│   │   ├── linearrag_retriever.py           # WRAP linearrag.LinearRAG.qa() pipeline
│   │   ├── evidence_graph_builder.py        # MỚI – build E_q từ activated entities
│   │   └── global_graph_retriever.py        # MỚI – top-r + 1-hop trên A_global
│   │
│   ├── training/
│   │   ├── train_client_condensation.py     # MỚI – stage B
│   │   ├── train_server_global_graph.py     # MỚI – stage C
│   │   └── (stage D dùng src/train.py + dual_graph_llm)
│   │
│   ├── configs/                             # MỚI – YAML configs
│   └── scripts/                             # MỚI – wrapper sh scripts
│
└── readme.md
```

## 29. Step-by-step "Copy and Fix" Workflow

### Step 0 — Khởi tạo

```bash
git clone https://github.com/XiaoxinHe/G-Retriever.git FedCondGraphRAG
cd FedCondGraphRAG
git checkout -b fedcond-grag

# Vendor LinearRAG
git clone https://github.com/DEEP-PolyU/LinearRAG.git /tmp/linearrag
mkdir -p fedcond_grag/external/linearrag
cp -r /tmp/linearrag/src/* fedcond_grag/external/linearrag/

# Vendor requirements
cat /tmp/linearrag/requirements.txt >> requirements_linearrag.txt
```

### Step 1 — Sửa import paths trong LinearRAG (search-and-replace)

Trong các file `fedcond_grag/external/linearrag/*.py`, sửa các absolute import:

```text
"from src.config import LinearRAGConfig"        →  "from .config import LinearRAGConfig"
"from src.LinearRAG import LinearRAG"           →  "from .LinearRAG import LinearRAG"
"from src.evaluate import Evaluator"            →  "from .evaluate import Evaluator"
"from src.utils import ..."                     →  "from .utils import ..."
```

Một dòng `sed`:

```bash
cd fedcond_grag/external/linearrag
find . -name "*.py" -exec sed -i \
  -e 's|from src\.|from .|g' \
  -e 's|import src\.|import .|g' \
  {} +
```

Tạo `fedcond_grag/external/linearrag/__init__.py`:

```python
from .config import LinearRAGConfig
from .LinearRAG import LinearRAG
from .utils import LLM_Model, setup_logging

__all__ = ["LinearRAG", "LinearRAGConfig", "LLM_Model", "setup_logging"]
```

### Step 2 — Hợp nhất env

LinearRAG dùng `sentence_transformers + spacy + en_core_web_trf`; G-Retriever dùng `torch + torch_geometric + transformers + peft`. Cài chung:

```bash
conda create -n fedcond python=3.9 -y
conda activate fedcond
# G-Retriever deps
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install peft pandas ogb transformers wandb sentencepiece torch_geometric \
  datasets pcst_fast gensim scipy==1.12 protobuf
# LinearRAG deps
pip install sentence-transformers spacy openai
python -m spacy download en_core_web_trf
```

### Step 3 — Smoke test (verify import works)

Trước khi viết logic, đảm bảo cả hai code base có thể được import song song:

```python
# scripts/smoke_test_imports.py
from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from fedcond_grag.external.linearrag import LinearRAG, LinearRAGConfig
print("Both codebases importable.")
```

### Step 4 — Wrap LinearRAG behind a thin façade

Trong `fedcond_grag/retrieval/linearrag_retriever.py`, không gọi class `LinearRAG` trực tiếp trong dataset class — bọc lại để dễ mock và test:

```python
# fedcond_grag/retrieval/linearrag_retriever.py
from fedcond_grag.external.linearrag import LinearRAG, LinearRAGConfig

class LinearRAGRetriever:
    """Thin wrapper exposing only what FedCondGraphRAG needs."""
    def __init__(self, passages, embedding_model, spacy_model, llm_model, **kw):
        cfg = LinearRAGConfig(
            embedding_model=embedding_model,
            spacy_model=spacy_model,
            llm_model=llm_model,
            **kw,
        )
        self.rag = LinearRAG(global_config=cfg)
        self.rag.index(passages)            # build Tri-Graph once

    def retrieve(self, question: str) -> dict:
        """Run two-stage retrieval; return passages + activated entities."""
        result = self.rag.qa([{"question": question}])
        return {
            "passages": result[0].get("retrieved_passages", []),
            "activated_entities": result[0].get("activated_entities", []),
            "tri_graph_view": self.rag.get_tri_graph_view(),   # may need to expose
        }
```

Nếu `LinearRAG.qa()` không trả về `activated_entities` ở top level, ta cần một **patch nhỏ** trong `fedcond_grag/external/linearrag/LinearRAG.py`: thêm `return` hoặc expose qua attribute (xem mục 32). Đây là kiểu "fix" duy nhất cần áp lên code đã copy.

### Step 5 — Build Tri-Graph re-export

`fedcond_grag/graph_building/trigraph_builder.py` không build lại Tri-Graph; nó **gọi** `LinearRAG.index()` rồi convert state thành format PyG:

```python
# fedcond_grag/graph_building/trigraph_builder.py
import torch
from torch_geometric.data import Data
from fedcond_grag.external.linearrag import LinearRAG, LinearRAGConfig

NODE_TYPE = {"entity": 0, "sentence": 1, "passage": 2}

def build_trigraph_for_client(passages, cfg) -> Data:
    rag = LinearRAG(global_config=cfg)
    rag.index(passages)
    # Trích entities, sentences, passages từ trạng thái LinearRAG.
    # (chính xác tên attribute lấy được sau khi inspect run.py output)
    entities  = rag.entity_list           # may be rag.graph.entities
    sentences = rag.sentence_list         # may be rag.graph.sentences
    passages_ = rag.passage_list          # rag.passages
    C = rag.contain_matrix                # |P| x |E| sparse  (Eq. 1)
    M = rag.mention_matrix                # |S| x |E| sparse  (Eq. 2)

    x_e = rag.entity_embeddings           # already cached by LinearRAG
    x_s = rag.sentence_embeddings
    x_p = rag.passage_embeddings
    x   = torch.cat([x_e, x_s, x_p], dim=0)

    node_type = torch.cat([
        torch.full((len(entities),),  NODE_TYPE["entity"]),
        torch.full((len(sentences),), NODE_TYPE["sentence"]),
        torch.full((len(passages_),), NODE_TYPE["passage"]),
    ])

    # S-E edges từ M, P-E edges từ C; offset index theo block
    se_edges = _coo_to_edge_index(M, offset_src=len(entities),
                                     offset_dst=0)
    pe_edges = _coo_to_edge_index(C, offset_src=len(entities)+len(sentences),
                                     offset_dst=0)
    edge_index = torch.cat([se_edges, pe_edges], dim=1)
    edge_type  = torch.cat([torch.zeros(se_edges.size(1)),
                            torch.ones (pe_edges.size(1))]).long()

    return Data(x=x, edge_index=edge_index, edge_type=edge_type,
                node_type=node_type)
```

> Lưu ý: nếu tên attribute `rag.entity_list`, `rag.contain_matrix`, ... không khớp với LinearRAG repo thực tế, đây là chỗ duy nhất cần **inspect và rename** sau khi clone. Coi đó là Task 1 sau khi `git clone`.

### Step 6 — Subclass `GraphLLM` thành `DualGraphLLM`

Đây là điểm "fix" sâu nhất trong G-Retriever. Thay vì viết lại `graph_llm.py`, ta **kế thừa**:

```python
# src/model/dual_graph_llm.py
import torch, torch.nn as nn
from src.model.graph_llm import GraphLLM    # giữ nguyên parent

class DualGraphLLM(GraphLLM):
    """G-Retriever với 2 graph encoders (evidence + condensed) + late fusion."""

    def __init__(self, args, **kw):
        super().__init__(args, **kw)
        # Reuse parent.graph_encoder làm evidence encoder (GNN_e)
        # Thêm GNN_c và projection riêng
        self.condensed_encoder = self._build_gnn(
            in_dim=args.gnn_in_dim,
            hidden_dim=args.gnn_hidden_dim,
            out_dim=args.gnn_out_dim,
            num_layers=args.gnn_num_layers_c,
            gnn_type=args.gnn_model_name_c,
        )
        self.projector_c = nn.Sequential(
            nn.Linear(args.gnn_out_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.llm.config.hidden_size),
        )

    def encode_graphs(self, batch):
        # parent.encode_graphs trả về z_e (graph token cho evidence graph E_q)
        z_e = super().encode_graphs(batch["evidence_graph"])
        # encode global condensed subgraph G_global(q)
        z_c_raw = self.condensed_encoder(batch["condensed_graph"])
        z_c = self.projector_c(z_c_raw.mean(0))
        return z_e, z_c

    def forward(self, samples):
        z_e, z_c = self.encode_graphs(samples)
        # build LLM input: [z_e ; z_c ; textualized_E_q ; passages ; question]
        ...   # 90% logic copy từ parent.forward, chỉ đổi prefix tokens
```

`src/model/__init__.py` thêm:

```python
from src.model.dual_graph_llm import DualGraphLLM
load_model["dual_graph_llm"] = DualGraphLLM
```

### Step 7 — Dataset class

`src/dataset/hotpot_fedcond.py` follow đúng pattern của `src/dataset/webqsp.py`:

```python
# src/dataset/hotpot_fedcond.py
import torch
from torch.utils.data import Dataset

from fedcond_grag.retrieval.linearrag_retriever import LinearRAGRetriever
from fedcond_grag.retrieval.global_graph_retriever import GlobalGraphRetriever
from fedcond_grag.retrieval.evidence_graph_builder import build_evidence_graph

class HotpotFedCondDataset(Dataset):
    def __init__(self, split="train"):
        self.questions = load_hotpot_split(split)
        self.lr   = LinearRAGRetriever.load_cached()
        self.ggr  = GlobalGraphRetriever.load_cached()
        self.prompt = "Answer the question using the given graphs and passages."
        self.graph_type = "tri_graph"

    def get_idx_split(self):
        return {"train": [...], "val": [...], "test": [...]}

    def __getitem__(self, idx):
        q = self.questions[idx]
        out = self.lr.retrieve(q["question"])
        E_q = build_evidence_graph(out)
        G_global_q = self.ggr.retrieve(q["question"])
        return dict(
            id=q["question_id"],
            question=q["question"],
            label=q["answer"],
            desc=textualize(E_q),                     # tận dụng utility của G-Retriever
            evidence_graph=E_q,
            condensed_graph=G_global_q,
            retrieved_passages=out["passages"],
        )
```

`src/dataset/__init__.py` thêm `load_dataset["hotpot_fedcond"] = HotpotFedCondDataset`.

`src/utils/collate.py` chỉ cần FIX nhẹ: hỗ trợ batch 2 graphs thay vì 1 (cộng thêm 4 dòng tạo `Batch` cho `condensed_graph`).

### Step 8 — Train script reuse

Stage D (dual graph prompting) chạy thẳng `train.py` đã có sẵn:

```bash
python train.py --dataset hotpot_fedcond --model_name dual_graph_llm \
                --gnn_model_name gt --gnn_model_name_c gat \
                --llm_model_name 7b_chat --llm_frozen True
```

Vì cả dataset và model đều đã được register qua registry pattern của G-Retriever, không cần đụng `train.py` (trừ optional CLI args).

Stage B (client condensation) và Stage C (server global) có script **riêng** vì pipeline khác (không có LLM trong vòng lặp):

```bash
python -m fedcond_grag.training.train_client_condensation --config configs/client_cond.yaml
python -m fedcond_grag.training.train_server_global_graph --config configs/server_pge.yaml
```

## 30. File-by-file Mapping: LinearRAG → FedCondGraphRAG

| LinearRAG file | FedCondGraphRAG target | Action |
|---|---|---|
| `src/LinearRAG.py` | `fedcond_grag/external/linearrag/LinearRAG.py` | **Copy nguyên**, sửa import path. Có thể cần patch nhỏ để expose `activated_entities`, `entity_embeddings`, `contain_matrix`, `mention_matrix` qua attributes hoặc method getter. |
| `src/config.py` | `fedcond_grag/external/linearrag/config.py` | **Copy nguyên**. |
| `src/evaluate.py` | `fedcond_grag/external/linearrag/evaluate.py` | **Copy nguyên** (có thể dùng để so sánh QA metrics nếu format predictions tương thích). |
| `src/utils.py` (LLM_Model, setup_logging) | `fedcond_grag/external/linearrag/utils.py` | **Copy nguyên**. `LLM_Model` chỉ dùng cho stage A baseline; stage D dùng LLM của G-Retriever. |
| `run.py` | (không copy) | Logic của `run.py` chia đôi: indexing → `trigraph_builder.build_trigraph_for_client`; QA → `linearrag_retriever.retrieve`. |
| `requirements.txt` | append vào root `requirements.txt` | Merge deps. |
| `scripts/` | tham khảo | Không copy; viết scripts mới trong `fedcond_grag/scripts/`. |
| `dataset/` (HuggingFace data) | (không cần) | Project dùng HotpotQA raw từ official source, không cần data đã preprocess của LinearRAG. |

## 31. File-by-file Mapping: G-Retriever → FedCondGraphRAG

| G-Retriever file | Action | Lý do |
|---|---|---|
| `src/model/graph_llm.py` | **Giữ nguyên** | Là parent class cho `DualGraphLLM`. |
| `src/model/gnn.py` | **Giữ + thêm import** | Reuse `GraphTransformer`, `GAT`, `GCN`. Thêm helper `_build_gnn(type='gt'/'gat'/'gcn', ...)` nếu chưa có. |
| `src/model/llm.py` | **Giữ nguyên** | LLM wrapper với LoRA support. |
| `src/model/pt_llm.py` | **Giữ nguyên** | Baseline B3 (prompt tuning). |
| `src/model/inference_llm.py` | **Giữ nguyên** | Baseline B1 (zero-shot). |
| `src/model/__init__.py` | **FIX**: register `DualGraphLLM` | 2 dòng `import` + 1 dòng `load_model[...]`. |
| `src/dataset/utils/retrieval.py` (PCST) | **Giữ; KHÔNG dùng cho hotpot_fedcond** | Chỉ giữ cho các dataset gốc của G-Retriever. Pipeline mới dùng LinearRAG retrieval. |
| `src/dataset/webqsp.py` | **Giữ** | Là template cho `hotpot_fedcond.py`. |
| `src/dataset/__init__.py` | **FIX**: register `hotpot_fedcond` | 2 dòng. |
| `src/utils/collate.py` | **FIX nhẹ**: hỗ trợ 2 graphs trong batch | ~4 dòng để batch condensed_graph song song với evidence_graph. |
| `src/utils/evaluate.py` | **FIX**: thêm `eval_funcs["hotpot_fedcond"]` | Implement EM/F1 cho HotpotQA. |
| `src/utils/{ckpt, seed, lr_schedule}.py` | **Giữ nguyên** | Generic. |
| `src/config.py` | **FIX nhẹ**: thêm CLI args (`--gnn_model_name_c`, `--global_top_r`, ...) | ~10 dòng. |
| `train.py`, `inference.py` | **Giữ nguyên** | Vì đã dùng registry pattern. |

## 32. Patches có thể cần áp vào LinearRAG sau khi copy

Đây là các điểm có thể (chưa chắc) cần fix sau khi inspect code thực tế:

1. **Expose internal state**: `LinearRAG.qa()` hiện trả về answer + passages. Ta cần thêm activated entities và evidence graph view. Patch: thêm method `get_activated_entities(question_id)` và `get_evidence_subgraph(question_id)`.

2. **Decouple LLM call**: `LinearRAG.qa()` gọi LLM của LinearRAG để sinh answer. Ta muốn **chỉ retrieve**, không sinh answer. Patch: thêm flag `do_generate=False` để skip phần generation, chỉ trả về retrieved structure.

3. **Per-client indexing**: `LinearRAG.index(passages)` hoạt động trên 1 corpus. Cho federated setting, ta tạo N instance, mỗi instance index riêng. Không patch cần thiết, chỉ wrap.

4. **Embedding cache I/O**: thêm `save_state(path)` / `load_state(path)` để Tri-Graph có thể được build offline (mất ~vài giờ cho HotpotQA), save, rồi load nhanh ở mỗi run training.

5. **Tensor dtype consistency**: LinearRAG dùng numpy/scipy sparse. Khi convert sang PyG `Data`, ép dtype về `float32` + `int64` cho edge_index để tương thích với GPU GNN forward.

## 33. Cái gì THỰC SỰ phải viết từ đầu

Sau khi tận dụng tối đa 2 repo trên, danh sách module phải code mới:

```text
fedcond_grag/data/hotpot_loader.py
fedcond_grag/data/federated_partition.py
fedcond_grag/data/corpus_index.py

fedcond_grag/graph_condensation/   (toàn bộ thư mục - DANCE không có code chính thức)
fedcond_grag/server_condensation/  (toàn bộ - đóng góp gốc)

fedcond_grag/retrieval/evidence_graph_builder.py
fedcond_grag/retrieval/global_graph_retriever.py

fedcond_grag/training/train_client_condensation.py
fedcond_grag/training/train_server_global_graph.py

src/model/dual_graph_llm.py        (~150 dòng subclass)
src/dataset/hotpot_fedcond.py      (~120 dòng theo template webqsp.py)
src/dataset/preprocess/hotpot.py   (~80 dòng)
```

Ước tính tổng dòng code phải viết mới: **~2.5K–3.5K LoC**, so với nếu viết lại cả Tri-Graph + LinearRAG retrieval + LLM wrapper từ đầu sẽ là **8K–10K LoC**.

## 34. Tham chiếu API ngoài cần verify khi clone

Khi clone về máy lần đầu, **chạy 3 verification scripts** trước khi viết logic mới, để biết exact API surface của LinearRAG (vì README không document đầy đủ):

```python
# scripts/verify_linearrag_api.py
import inspect
from fedcond_grag.external.linearrag import LinearRAG

# 1) Liệt kê attributes/methods của LinearRAG sau khi index()
rag = LinearRAG(global_config=...)
rag.index(["Some sample passage about Einstein."])
print([a for a in dir(rag) if not a.startswith("_")])

# 2) Inspect class config
print(inspect.getsource(type(rag.config)))

# 3) Chạy qa() trên 1 question, in toàn bộ return structure
out = rag.qa([{"question": "Who is Einstein?"}])
print(out)
```

Output của 3 scripts trên xác định chính xác tên attribute (`rag.entity_list` vs `rag.graph.entities` vs `rag._entities`) ở `trigraph_builder.py` mục 29 Step 5.

## 35. Migration Checkpoints (theo thứ tự thời gian)

Sau mỗi step ở §29, chạy checkpoint nhỏ trước khi tiếp:

| Step | Checkpoint command | Pass criteria |
|---|---|---|
| Step 0 | `ls fedcond_grag/external/linearrag/*.py` | ≥ 4 files |
| Step 1 | `python -c "from fedcond_grag.external.linearrag import LinearRAG"` | Không ImportError |
| Step 2 | `python -c "import spacy; spacy.load('en_core_web_trf')"` | Load ok |
| Step 3 | `python scripts/smoke_test_imports.py` | "Both codebases importable." |
| Step 4 | `python scripts/test_linearrag_wrapper.py` | retrieve(q) returns dict |
| Step 5 | `python scripts/test_trigraph_to_pyg.py` | PyG Data với 3 node_type values |
| Step 6 | `python -c "from src.model.dual_graph_llm import DualGraphLLM"` + dummy forward | Forward không crash |
| Step 7 | `python -c "from src.dataset import load_dataset; ds=load_dataset['hotpot_fedcond'](); print(ds[0].keys())"` | dict có evidence_graph + condensed_graph |
| Step 8 | `python train.py --dataset hotpot_fedcond --model_name dual_graph_llm --num_epochs 1` | Train 1 epoch không OOM, val loss hữu hạn |

---

---

## See also
- **Tầng thứ 3 (gfl/OpenFGL infrastructure)**: `11_INT_GFL.md` toàn bộ (đặc biệt §50 vendor workflow)
- **Updated folder layout với gfl thêm vào**: `11_INT_GFL.md` §46
- **Updated "what to truly write from scratch" với gfl giảm thêm**: `11_INT_GFL.md` §51
- **DANCE re-implementation (vì không có code)**: `10_DANCE_REFERENCE.md`
- **Module-level spec mà các file `fedcond_grag/...` này phải thoả mãn**: `02_*` đến `05_*`
