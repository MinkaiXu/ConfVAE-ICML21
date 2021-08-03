import copy
import torch
from torch_sparse import spspmm, coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.transforms import Compose

from .chem import *


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj, order):
    """
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    """
    adj_mats = [torch.eye(adj.size(0)).long(), binarize(adj + torch.eye(adj.size(0)).long())]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    # print(adj_mats)

    order_mat = torch.zeros_like(adj)
    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return order_mat


class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def __call__(self, data:Data):
        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types+adj_order-1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N)
        assert (data.edge_index == edge_index_1).all()

        return data


class AddEdgeLength(object):

    def __call__(self, data:Data):
        pos = data.pos
        row, col = data.edge_index
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        data.edge_length = dist
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(BOND_NAMES)
        self.bonds[len(BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data:Data):
        data.edge_name = []
        for i in range(data.edge_index.size(1)):
            tail = data.edge_index[0, i]
            head = data.edge_index[1, i]
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = get_atom_symbol(data.node_type[tail].item())
            head_name = get_atom_symbol(data.node_type[head].item())
            name = '%s_%s_%s_%d_%d' % (
                self.bonds[data.edge_type[i].item()] if data.edge_type[i].item() in self.bonds else 'E'+str(data.edge_type[i].item()),
                tail_name,
                head_name,
                tail,
                head,
            )
            if hasattr(data, 'edge_length'):
                name += '_%.3f' % (data.edge_length[i].item())
            data.edge_name.append(name)
        return data


class AddIsBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data:Data):
        is_bond = []
        E = data.edge_index.size(1)
        for i in range(E):
            if data.edge_type[i].item() < len(BOND_TYPES):
                is_bond.append(True)
            else:
                is_bond.append(False)
        data.is_bond = is_bond
        return data


def get_standard_transforms(order=3, asym_name=True):
    tf = Compose([
        AddHigherOrderEdges(order=order),
        AddIsBond(),
        AddEdgeLength(),
        AddEdgeName(asymmetric=asym_name),
    ])
    return tf


if __name__ == '__main__':
    import numpy as np
    import networkx as nx
    from tqdm.auto import tqdm
    from datasets.molecule import *
    def bin(mat):
        return np.where(mat > 0, np.ones_like(mat), np.zeros_like(mat))
    print('Testing AddHigherOrderEdges...')
    t = AddHigherOrderEdges(order=3)
    dset = MoleculeDataset('./data/qm9/QM9_train.pkl')
    for idx in tqdm(range(100)):
        data = dset[idx]
        data_t = t(dset[idx])
        G = data.nx

        I = np.eye(len(G.nodes), dtype=np.int)
        A = nx.adjacency_matrix(G).todense()
        A1 = bin(A + I)
        A2 = bin(A1 @ A1)
        A3 = bin(A2 @ A1)
        A_angle = A2 - A1
        A_dihed = A3 - A2
        hop_mat = A + 2*A_angle + 3*A_dihed

        for x, (i, j) in enumerate(data_t.edge_index.t()):
            if hop_mat[i,j] == 0:
                assert False
            elif hop_mat[i,j] == 1:
                # print(idx, x, i, j, data_t.edge_type[x])
                assert data_t.edge_type[x] < len(BOND_TYPES)
            elif hop_mat[i,j] == 2:
                assert data_t.edge_type[x] == len(BOND_TYPES) + 1
            elif hop_mat[i,j] == 3:
                assert data_t.edge_type[x] == len(BOND_TYPES) + 2
