from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .egnn import GNN as ReferenceGNN


def _copy_linear_stack(destination: nn.Linear, sources: tuple[nn.Linear, ...]) -> None:
    """Pack independent same-input projections into one output projection."""
    with torch.no_grad():
        destination.weight.copy_(torch.cat([source.weight for source in sources], dim=0))
        if destination.bias is not None:
            destination.bias.copy_(torch.cat([source.bias for source in sources], dim=0))


class FusedGCL(nn.Module):
    """YuelPocket GCL with fewer projection and aggregation launches.

    The equations match ``egnn.GCL``. The only intentional numerical change is
    the order in which equivalent floating-point reductions are performed.
    """

    def __init__(self, c_h: int, bi_directional: bool = True):
        super().__init__()
        self.bi_directional = bool(bi_directional)
        self.layer_norm_h = nn.LayerNorm(c_h)
        self.layer_norm_e = nn.LayerNorm(c_h)

        # These projections consume the same h and are packed into one GEMM.
        self.node_projections = nn.Linear(c_h, 4 * c_h)
        self.proj_edge = nn.Linear(c_h, c_h)
        self.edge_mlp = nn.Sequential(
            nn.Linear(c_h, c_h),
            nn.SiLU(),
            nn.Linear(c_h, c_h),
        )

        self.edge_message_projections = nn.Linear(c_h, 2 * c_h)
        self.node_mlp = nn.Sequential(
            nn.Linear(c_h, c_h),
            nn.SiLU(),
            nn.Linear(c_h, c_h),
        )

    @classmethod
    def from_reference(cls, reference_layer: nn.Module) -> "FusedGCL":
        fused = cls(
            c_h=int(reference_layer.layer_norm_h.normalized_shape[0]),
            bi_directional=bool(reference_layer.bi_directional),
        )
        fused.layer_norm_h.load_state_dict(reference_layer.layer_norm_h.state_dict())
        fused.layer_norm_e.load_state_dict(reference_layer.layer_norm_e.state_dict())
        _copy_linear_stack(
            fused.node_projections,
            (
                reference_layer.gate_node,
                reference_layer.proj_src,
                reference_layer.proj_dst,
                reference_layer.proj_node,
            ),
        )
        fused.proj_edge.load_state_dict(reference_layer.proj_edge.state_dict())
        fused.edge_mlp.load_state_dict(reference_layer.edge_mlp.state_dict())
        _copy_linear_stack(
            fused.edge_message_projections,
            (reference_layer.edge2node, reference_layer.gate_edge),
        )
        fused.node_mlp.load_state_dict(reference_layer.node_mlp.state_dict())
        return fused

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        e: torch.Tensor,
        degree: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layer_norm_h(h)
        e = self.layer_norm_e(e)

        gate, proj_src, proj_dst, proj_node = self.node_projections(h).chunk(4, dim=-1)
        h_src = (proj_src * gate)[src]
        h_dst = (proj_dst * gate)[dst]
        e = e + self.edge_mlp(self.proj_edge(e) * (h_src + h_dst) / 2.0)

        edge2node, gate_edge = self.edge_message_projections(e).chunk(2, dim=-1)
        edge_msg = edge2node * gate_edge
        agg = edge_msg.new_zeros((h.size(0), edge_msg.size(1)))
        agg.scatter_add_(0, dst_idx, edge_msg)
        if self.bi_directional:
            agg.scatter_add_(0, src_idx, edge_msg)

        # degree is computed once per graph batch and reused by every layer.
        agg = agg / (degree + 1e-6)
        h = h + self.node_mlp(proj_node * agg)
        return h, e


class FusedGNN(nn.Module):
    """Reference-compatible YuelPocket GNN with a lower launch count."""

    def __init__(
        self,
        c_h: int,
        n_layers: int,
        in_node_dim: int,
        in_edge_dim: int,
        out_node_dim: int,
        out_edge_dim: int,
        bi_directional: bool = True,
    ):
        super().__init__()
        self.bi_directional = bool(bi_directional)
        self.emb_node = nn.Sequential(
            nn.Linear(in_node_dim, c_h * 2),
            nn.SiLU(),
            nn.Linear(c_h * 2, c_h),
        )
        self.emb_edge = nn.Sequential(
            nn.Linear(in_edge_dim, c_h * 2),
            nn.SiLU(),
            nn.Linear(c_h * 2, c_h),
        )
        self.layers = nn.ModuleList(
            [FusedGCL(c_h=c_h, bi_directional=bi_directional) for _ in range(n_layers)]
        )
        self.out_node = nn.Linear(c_h, out_node_dim)
        self.out_edge = nn.Linear(c_h, out_edge_dim)

    @classmethod
    def from_reference(cls, reference_gnn: ReferenceGNN) -> "FusedGNN":
        if not reference_gnn.layers:
            raise ValueError("Cannot convert a reference GNN without layers")
        first_layer = reference_gnn.layers[0]
        fused = cls(
            c_h=int(first_layer.layer_norm_h.normalized_shape[0]),
            n_layers=len(reference_gnn.layers),
            in_node_dim=int(reference_gnn.emb_node[0].in_features),
            in_edge_dim=int(reference_gnn.emb_edge[0].in_features),
            out_node_dim=int(reference_gnn.out_node.out_features),
            out_edge_dim=int(reference_gnn.out_edge.out_features),
            bi_directional=bool(first_layer.bi_directional),
        )
        fused.emb_node.load_state_dict(reference_gnn.emb_node.state_dict())
        fused.emb_edge.load_state_dict(reference_gnn.emb_edge.state_dict())
        for fused_layer, reference_layer in zip(fused.layers, reference_gnn.layers):
            converted = FusedGCL.from_reference(reference_layer)
            fused_layer.load_state_dict(converted.state_dict())
        fused.out_node.load_state_dict(reference_gnn.out_node.state_dict())
        fused.out_edge.load_state_dict(reference_gnn.out_edge.state_dict())
        return fused

    def copy_from_reference(self, reference_gnn: ReferenceGNN) -> None:
        converted = type(self).from_reference(reference_gnn)
        self.load_state_dict(converted.state_dict())

    def forward(
        self,
        h: torch.Tensor,
        edges: torch.Tensor,
        e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.emb_node(h)
        e = self.emb_edge(e)
        src, dst = edges

        if self.bi_directional:
            degree_index = torch.cat((dst, src), dim=0)
        else:
            degree_index = dst
        degree = torch.bincount(degree_index, minlength=h.size(0)).to(dtype=e.dtype).unsqueeze(-1)
        degree.clamp_min_(1.0)
        dst_idx = dst[:, None].expand(-1, e.size(-1))
        src_idx = src[:, None].expand(-1, e.size(-1))

        for layer in self.layers:
            h, e = layer(h, src, dst, e, degree, src_idx, dst_idx)
        return self.out_node(h), self.out_edge(e)
