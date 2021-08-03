import torch
import numpy as np
from tqdm.auto import tqdm


def remove_outliers(data_list, std_range=2.5, mode='bond'):
    assert mode in ('bond', 'all')
    sample = data_list[0]
    smiles = sample.smiles
    for data in data_list:
        assert data.smiles == smiles

    if mode == 'bond':
        edge_index = sample.bond_edge_index
    else:
        edge_index = sample.edge_index
    node_type  = sample.node_type
    lengths = []    # list, (N, E)
    for data in tqdm(data_list, desc='Length'):
        lengths.append((data.pos[edge_index[0]] - data.pos[edge_index[1]]).norm(dim=1).tolist())
    lengths = np.array(lengths)

    mean, std = np.mean(lengths, axis=0, keepdims=True), np.std(lengths, axis=0, keepdims=True)
    range_min = mean - std * std_range
    range_max = mean + std * std_range
    filt = np.logical_and(lengths > range_min, lengths < range_max)
    filt = filt.sum(axis=1) == lengths.shape[1]
    filt_data = [data_list[i] for i, b in enumerate(filt) if b == True]
    return filt_data
