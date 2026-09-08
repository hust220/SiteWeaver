import torch
from typing import List


class Graph:
    """Small graph container adapted from YuelPocket's DGL replacement."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        if edge_index.size(0) != 2:
            raise ValueError(f"edge_index should have shape [2, E], got {edge_index.shape}")
        if edge_index.numel() > 0:
            if edge_index.min() < 0 or edge_index.max() >= num_nodes:
                raise ValueError(f"Edge indices must be in range [0, {num_nodes})")

        self.num_nodes = int(num_nodes)
        self.num_edges = int(edge_index.size(1))
        self.edge_index = edge_index.clone().long()
        self.ndata = {}
        self.edata = {}
        self.batch_size = 1
        self.batch_num_nodes = [self.num_nodes]
        self.batch_num_edges = [self.num_edges]

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        for key in self.ndata:
            self.ndata[key] = self.ndata[key].to(device)
        for key in self.edata:
            self.edata[key] = self.edata[key].to(device)
        return self

    def clone(self):
        graph = Graph(self.edge_index.clone(), self.num_nodes)
        graph.ndata = {k: v.clone() for k, v in self.ndata.items()}
        graph.edata = {k: v.clone() for k, v in self.edata.items()}
        graph.meta = dict(getattr(self, "meta", {}))
        graph.batch_size = self.batch_size
        graph.batch_num_nodes = list(self.batch_num_nodes)
        graph.batch_num_edges = list(self.batch_num_edges)
        return graph

    def get_batch_masks(self) -> List[torch.Tensor]:
        masks = []
        start = 0
        device = self.edge_index.device
        for n_nodes in self.batch_num_nodes:
            mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
            mask[start:start + n_nodes] = True
            masks.append(mask)
            start += n_nodes
        return masks


def batch(graphs: List[Graph]) -> Graph:
    if not graphs:
        raise ValueError("Cannot batch empty graph list")
    if len(graphs) == 1:
        return graphs[0].clone()

    graphs = [g.clone() for g in graphs]
    node_offsets = [0]
    for graph in graphs:
        node_offsets.append(node_offsets[-1] + graph.num_nodes)

    edge_parts = []
    for graph, offset in zip(graphs, node_offsets):
        if graph.num_edges:
            edge_parts.append(graph.edge_index + offset)
    if edge_parts:
        edge_index = torch.cat(edge_parts, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    out = Graph(edge_index, sum(g.num_nodes for g in graphs))
    out.batch_size = len(graphs)
    out.batch_num_nodes = [g.num_nodes for g in graphs]
    out.batch_num_edges = [g.num_edges for g in graphs]

    node_keys = set().union(*(g.ndata.keys() for g in graphs))
    for key in node_keys:
        out.ndata[key] = torch.cat([g.ndata[key] for g in graphs], dim=0)

    edge_keys = set().union(*(g.edata.keys() for g in graphs))
    for key in edge_keys:
        out.edata[key] = torch.cat([g.edata[key] for g in graphs], dim=0)
    return out
