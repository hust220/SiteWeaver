import torch
import torch.nn as nn


class GCL(nn.Module):
    """YuelPocket graph convolution layer."""

    def __init__(self, c_h, bi_directional=True):
        super().__init__()
        self.bi_directional = bi_directional
        self.layer_norm_h = nn.LayerNorm(c_h)
        self.layer_norm_e = nn.LayerNorm(c_h)
        self.gate_node = nn.Linear(c_h, c_h)
        self.proj_src = nn.Linear(c_h, c_h)
        self.proj_dst = nn.Linear(c_h, c_h)
        self.proj_edge = nn.Linear(c_h, c_h)
        self.edge_mlp = nn.Sequential(nn.Linear(c_h, c_h), nn.SiLU(), nn.Linear(c_h, c_h))
        self.edge2node = nn.Linear(c_h, c_h)
        self.gate_edge = nn.Linear(c_h, c_h)
        self.proj_node = nn.Linear(c_h, c_h)
        self.node_mlp = nn.Sequential(nn.Linear(c_h, c_h), nn.SiLU(), nn.Linear(c_h, c_h))

    def forward(self, h, edges, e):
        h = self.layer_norm_h(h)
        e = self.layer_norm_e(e)
        src, dst = edges

        gate = self.gate_node(h)
        h_src = (self.proj_src(h) * gate)[src]
        h_dst = (self.proj_dst(h) * gate)[dst]
        e = e + self.edge_mlp(self.proj_edge(e) * (h_src + h_dst) / 2.0)

        edge_msg = self.edge2node(e) * self.gate_edge(e)
        shape = (h.size(0), edge_msg.size(1))
        agg = edge_msg.new_zeros(shape)
        dst_idx = dst[:, None].expand(-1, edge_msg.size(1))
        agg.scatter_add_(0, dst_idx, edge_msg)
        if self.bi_directional:
            src_idx = src[:, None].expand(-1, edge_msg.size(1))
            agg.scatter_add_(0, src_idx, edge_msg)

        norm = edge_msg.new_zeros(shape)
        ones = edge_msg.new_ones(edge_msg.shape)
        norm.scatter_add_(0, dst_idx, ones)
        if self.bi_directional:
            norm.scatter_add_(0, src_idx, ones)
        norm[norm == 0] = 1
        agg = agg / (norm + 1e-6)

        h = h + self.node_mlp(self.proj_node(h) * agg)
        return h, e


class GNN(nn.Module):
    """YuelPocket GNN backbone."""

    def __init__(self, c_h, n_layers, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, bi_directional=True):
        super().__init__()
        self.emb_node = nn.Sequential(nn.Linear(in_node_dim, c_h * 2), nn.SiLU(), nn.Linear(c_h * 2, c_h))
        self.emb_edge = nn.Sequential(nn.Linear(in_edge_dim, c_h * 2), nn.SiLU(), nn.Linear(c_h * 2, c_h))
        self.layers = nn.ModuleList([GCL(c_h=c_h, bi_directional=bi_directional) for _ in range(n_layers)])
        self.out_node = nn.Linear(c_h, out_node_dim)
        self.out_edge = nn.Linear(c_h, out_edge_dim)

    def forward(self, h, edges, e):
        h = self.emb_node(h)
        e = self.emb_edge(e)
        for layer in self.layers:
            h, e = layer(h, edges, e)
        return self.out_node(h), self.out_edge(e)

