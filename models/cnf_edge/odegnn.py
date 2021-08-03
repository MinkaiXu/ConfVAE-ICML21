import torch
from torch import Tensor
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from .odemlp import *


class GINEConv(MessagePassing):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super(GINEConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, t, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(t, out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.softplus(x_j + edge_attr)


class ODEgnn(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.act = F.softplus
        self.d_fc1 = ConcatSquashLinear(1, hidden_dim, dim_c=0)
        # self.bn_d1 = torch.nn.BatchNorm1d(hidden_dim)
        self.d_fc2 = ConcatSquashLinear(hidden_dim, hidden_dim, dim_c=0)
        # self.bn_d2 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv1 = GINEConv(ODEmlp((hidden_dim, ), (hidden_dim, )))
        # self.bn_conv1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GINEConv(ODEmlp((hidden_dim, ), (hidden_dim, )))
        # self.bn_conv2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GINEConv(ODEmlp((hidden_dim, ), (hidden_dim, )))
        # self.bn_conv3 = torch.nn.BatchNorm1d(hidden_dim)

        self.out_fc1 = ConcatSquashLinear(2 * hidden_dim, hidden_dim, dim_c=0)
        # self.bn_out1 = torch.nn.BatchNorm1d(hidden_dim)
        self.out_fc2 = ConcatSquashLinear(hidden_dim, hidden_dim // 2, dim_c=0)
        # self.bn_out2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.out_fc3 = ConcatSquashLinear(hidden_dim // 2, 1, dim_c=0)

        self.edge_index = None


    def forward(self, t, x, node_attr, edge_attr):
        assert self.edge_index is not None, '`edge_index` is not prepared.'
        edge_index = self.edge_index

        d_emb = self.act(self.d_fc1(t, x))
        d_emb = self.d_fc2(t, d_emb)   # Embedings for edge lengths `x`
        edge_attr = d_emb*edge_attr

        t_node = torch.ones_like(node_attr)[0, :1] * t.mean()
        h = node_attr
        h = self.act(self.conv1(t_node, h, edge_index, edge_attr))
        h = self.act(self.conv2(t_node, h, edge_index, edge_attr))
        h = self.conv3(t_node, h, edge_index, edge_attr)

        h_row, h_col = h[edge_index[0]], h[edge_index[1]]
        pair_feat = torch.cat([h_row*h_col, edge_attr], dim=-1)
        # pair_feat = h_row*h_col
        pair_feat = self.act(self.out_fc1(t, pair_feat))
        pair_feat = self.act(self.out_fc2(t, pair_feat))
        out = self.out_fc3(t, pair_feat)

        return out

