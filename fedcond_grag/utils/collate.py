from torch_geometric.data import Batch


def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    for graph_key in ("graph", "evidence_graph", "condensed_graph"):
        if graph_key in batch:
            batch[graph_key] = Batch.from_data_list(batch[graph_key])
    return batch
