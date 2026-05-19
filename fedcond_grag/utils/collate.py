from torch_geometric.data import Batch


def collate_fn(original_batch):
    batch = {}
    all_keys = set().union(*(d.keys() for d in original_batch))
    for k in all_keys:
        batch[k] = [d.get(k) for d in original_batch]
    for graph_key in ("graph", "evidence_graph", "condensed_graph"):
        if graph_key in batch and any(v is not None for v in batch[graph_key]):
            non_null = [v for v in batch[graph_key] if v is not None]
            if len(non_null) == len(batch[graph_key]):
                batch[graph_key] = Batch.from_data_list(batch[graph_key])
            else:
                # mixed None — drop the key so model uses zeros fallback
                del batch[graph_key]
    return batch
