"""
dataset.py - Data loading and leakage-free splitting for the Facebook
Large Page-Page Network (Rozemberczki et al., 2021).

Dataset: the modified Gayanku `facebook.npz`
(https://github.com/gayanku/facebook_page-page_dataset), which contains
22,470 Facebook pages (nodes), 171,002 mutual-like links (stored as
342,004 directed edges), 128-dim node features, and 4 page categories:
0 = TV show, 1 = government, 2 = company, 3 = politician.

Setting: TRANSDUCTIVE semi-supervised node classification.
    - The model sees the WHOLE graph (all nodes, all edges, all features)
      during training, because message passing needs neighbours.
    - The loss is computed ONLY on training-node labels.
    - Validation/test labels are never used for training or tuning;
      they are hidden behind boolean masks.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

CLASS_NAMES = ["tvshow", "government", "company", "politician"]


def load_facebook(path="data/facebook.npz"):
    """Load the .npz file and convert it into a PyG Data object.

    Returns:
        data (Data): x [N, 128] float, edge_index [2, E] long, y [N] long.
        page_names (list[str]): human-readable page names (for the failure autopsy).
    """
    # allow_pickle=True is needed because page_name holds Python strings
    raw = np.load(path, allow_pickle=True)

    x = torch.tensor(raw["X"], dtype=torch.float)
    y = torch.tensor(raw["y"], dtype=torch.long)

    # edges are stored as two parallel arrays (source, target);
    # stack them into the [2, E] layout PyG expects
    edge_index = torch.tensor(np.vstack([raw["edges_x"], raw["edges_y"]]),
                              dtype=torch.long)

    page_names = [str(n) for n in raw["page_name"]]
    return Data(x=x, edge_index=edge_index, y=y), page_names


def make_stratified_masks(y, train_frac=0.70, val_frac=0.15, seed=42):
    """Split node indices into train/val/test, stratified by class.

    Stratified = each class is split 70/15/15 separately, so the small
    politician class (~15% of nodes) keeps the same share in every split.
    A fixed seed makes the split identical on every run (reproducibility),
    so the MLP baseline and the GCN are compared on exactly the same nodes.

    Returns:
        train_mask, val_mask, test_mask: boolean tensors of shape [N].
    """
    gen = torch.Generator().manual_seed(seed)
    n = y.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for c in torch.unique(y):
        # all node indices belonging to class c, shuffled
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=gen)]

        n_train = int(round(train_frac * idx.numel()))
        n_val = int(round(val_frac * idx.numel()))

        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def check_masks(train_mask, val_mask, test_mask):
    """Leakage guard: every node must be in exactly ONE split."""
    overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
    assert not overlap.any(), "Leakage: a node appears in more than one split"
    assert (train_mask | val_mask | test_mask).all(), "Some nodes are in no split"


def get_data(path="data/facebook.npz", seed=42):
    """One-call helper used by train.py and predict.py."""
    data, page_names = load_facebook(path)
    data.train_mask, data.val_mask, data.test_mask = make_stratified_masks(data.y, seed=seed)
    check_masks(data.train_mask, data.val_mask, data.test_mask)
    return data, page_names


if __name__ == "__main__":
    # Data audit: run `python dataset.py` to print dataset statistics.
    data, names = get_data()
    src, dst = data.edge_index

    print(f"Nodes: {data.num_nodes} | Directed edges: {data.num_edges} "
          f"| Features: {data.num_node_features} | Classes: {len(CLASS_NAMES)}")

    counts = torch.bincount(data.y)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<11} {counts[i].item():>6}  ({100 * counts[i].item() / data.num_nodes:.1f}%)")

    print(f"Split sizes -> train {int(data.train_mask.sum())} | "
          f"val {int(data.val_mask.sum())} | test {int(data.test_mask.sum())}")

    # Edge homophily: fraction of edges joining two pages of the SAME class.
    # High homophily is the reason a GNN should beat a features-only MLP.
    homophily = (data.y[src] == data.y[dst]).float().mean().item()
    print(f"Edge homophily: {homophily:.3f}")

    deg = degree(src, num_nodes=data.num_nodes)
    print(f"Degree -> mean {deg.mean():.1f} | median {deg.median():.0f} | "
          f"max {int(deg.max())} | isolated nodes {int((deg == 0).sum())}")