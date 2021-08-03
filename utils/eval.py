import os
from numpy.lib.function_base import append
import torch
from torch_geometric.data import Data
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.AllChem import AlignMol
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdMolAlign import GetBestRMS
from functools import partial 

from .chem import get_atom_symbol, set_rdmol_positions
from .transforms import get_standard_transforms
from .metrics import compute_mmd
from .rmsd import GetBestRMSD
from .misc import split_dataset_by_smiles


def GetBestRMSD(probe, ref):
    # rmsd = AlignMol(probe, ref)
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = GetBestRMS(probe, ref)
    return rmsd


def evaluate_conf(rdmol, pos_ref, pos_gen, useFF=False, thresholds=[0.5]):
    """
    Args:
        rdmol:  An RDKit molecule object.
        pos_ref:  (num_refs, num_atoms, 3)
        pos_gen:  (num_gens, num_atoms, 3)
        thresholds:  A list of thresholds.
    Returns:
        coverages:  A list of coverage scores at different thresholds.
        rmsd_ref_min:  (num_ref, )
    """
    num_gen = pos_gen.shape[0]
    num_ref = pos_ref.shape[0]

    # row: ref, col, gen
    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen],dtype=np.float)
    
    for i in range(num_gen):
        gen_mol = set_rdmol_positions(rdmol, pos_gen[i])
        if useFF:
            #print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        for j in range(num_ref):
            ref_mol = set_rdmol_positions(rdmol, pos_ref[j])
            
            rmsd_confusion_mat[j,i] = GetBestRMSD(gen_mol, ref_mol)
    
    rmsd_ref_min = rmsd_confusion_mat.min(-1)   # (num_ref, )

    coverages = []
    for t in thresholds:
        coverages.append((rmsd_ref_min <= t).mean())

    return coverages, rmsd_ref_min.mean(), rmsd_ref_min


def evaluate_distance(pos_ref, pos_gen, edge_index, atom_type, ignore_H=True):

    # compute generated length and ref length 
    ref_lengths = (pos_ref[:, edge_index[0]] - pos_ref[:, edge_index[1]]).norm(dim=-1) # (N, num_edge)
    gen_lengths = (pos_gen[:, edge_index[0]] - pos_gen[:, edge_index[1]]).norm(dim=-1) # (M, num_edge)
    # print(ref_lengths.shape, gen_lengths.shape)

    stats_single = []
    first = 1
    for i, (row, col) in enumerate(tqdm(edge_index.t())):
        if row >= col: 
            continue
        if ignore_H and 1 in (atom_type[row].item(), atom_type[col].item()): 
            continue
        gen_l = gen_lengths[:, i]
        ref_l = ref_lengths[:, i]
        if first:
            # print(gen_l.shape, ref_l.shape)
            first = 0
        mmd = compute_mmd(gen_l.view(-1, 1).cuda(), ref_l.view(-1, 1).cuda()).item()
        stats_single.append({
            'edge_id': i,
            'elems': '%s - %s' % (get_atom_symbol(atom_type[row].item()), get_atom_symbol(atom_type[col].item())),
            'nodes': (row.item(), col.item()),
            'gen_lengths': gen_l.cpu(),
            'ref_lengths': ref_l.cpu(),
            'mmd': mmd
        })

    first = 1
    stats_pair = []
    for i, (row_i, col_i) in enumerate(tqdm(edge_index.t())):
        if row_i >= col_i: 
            continue
        if ignore_H and 1 in (atom_type[row_i].item(), atom_type[col_i].item()): 
            continue
        for j, (row_j, col_j) in enumerate(edge_index.t()):
            if (row_i >= row_j) or (row_j >= col_j): 
                continue
            if ignore_H and 1 in (atom_type[row_j].item(), atom_type[col_j].item()): 
                continue

            gen_L = gen_lengths[:, (i,j)]   # (N, 2)
            ref_L = ref_lengths[:, (i,j)]   # (M, 2)
            if first:
                # print(gen_L.shape, ref_L.shape)
                first = 0
            mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

            stats_pair.append({
                'edge_id': (i, j),
                'elems': (
                    '%s - %s' % (get_atom_symbol(atom_type[row_i].item()), get_atom_symbol(atom_type[col_i].item())), 
                    '%s - %s' % (get_atom_symbol(atom_type[row_j].item()), get_atom_symbol(atom_type[col_j].item())),                         
                ),
                'nodes': (
                    (row_i.item(), col_i.item()),
                    (row_j.item(), col_j.item()),
                ),
                'gen_lengths': gen_L.cpu(),
                'ref_lengths': ref_L.cpu(),
                'mmd': mmd
            })    

    edge_filter = edge_index[0] < edge_index[1]
    if ignore_H:
        for i, (row, col) in enumerate(edge_index.t()): 
            if 1 in (atom_type[row].item(), atom_type[col].item()):
                edge_filter[i] = False

    gen_L = gen_lengths[:, edge_filter]    # (N, Ef)
    ref_L = ref_lengths[:, edge_filter]    # (M, Ef)
    # print(gen_L.shape, ref_L.shape)
    mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

    stats_all = {
        'gen_lengths': gen_L.cpu(),
        'ref_lengths': ref_L.cpu(),
        'mmd': mmd
    }
    return stats_single, stats_pair, stats_all


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Params:
	    source: n * len(x)
	    target: m * len(y)
	Return:
		sum(kernel_val): Sum of various kernel matrices
    '''
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    total1 = total.unsqueeze(1).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)
 
def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Params:
	    source: (N, D)
	    target: (M, D)
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

    return loss


def _evaluate_conf(data, useFF, thresholds):
    return evaluate_conf(data[0], data[1], data[2], useFF=useFF, thresholds=thresholds)


class CovMatEvaluator(object):

    def __init__(self, num_workers=16, use_FF=False, thresholds=[0.5, 1.25]):
        super().__init__()
        self.pool = mp.Pool(num_workers)
        self.func = partial(_evaluate_conf, useFF=use_FF, thresholds=thresholds)

    def __call__(self, ref_dset, gen_dset):
        ref_grouped = split_dataset_by_smiles(ref_dset)
        gen_grouped = split_dataset_by_smiles(gen_dset)

        rdmols = []
        pos_refs = []
        pos_gens = []

        for smiles, gen_mols in gen_grouped.items():
            if smiles not in ref_grouped:
                continue
            ref_mols = ref_grouped[smiles]
            rdmols.append(gen_mols[0].rdmol)

            p_ref = []
            p_gen = []
            for mol in ref_mols:
                p_ref.append(mol.pos.cpu().numpy().reshape(1, -1, 3))
            for mol in gen_mols:
                p_gen.append(mol.pos.cpu().numpy().reshape(1, -1, 3))
            pos_refs.append(np.vstack(p_ref))
            pos_gens.append(np.vstack(p_gen))
        
        # return rdmols, pos_refs, pos_gens
        return self._run(rdmols, pos_refs, pos_gens)

    def _run(self, rdmols, pos_refs, pos_gens):
        """
        Args:
            rdmols:  A list of rdkit molecules.
            pos_refs:  A list of numpy tensors of shape (num_refs, num_atoms, 3)
            pos_gens:  A list of numpy tensors of shape (num_gens, num_atoms, 3)
        """
        covs = []
        mats = []
        for result in tqdm(self.pool.imap(self.func, zip(rdmols, pos_refs, pos_gens)), total=len(rdmols)):
            covs.append(result[0])
            mats.append(result[1])
        return covs, mats



def _evaluate_distance(data, ignore_H):
    return evaluate_distance(data[0], data[1], data[2], data[3], ignore_H=ignore_H)

class DistEvaluator(object):
    
    def __init__(self, ignore_H=False, device='cuda'):
        super().__init__()
        self.device = device
        self.func = partial(_evaluate_distance, ignore_H=ignore_H)


    def __call__(self, ref_dset, gen_dset):
        ref_grouped = split_dataset_by_smiles(ref_dset)
        gen_grouped = split_dataset_by_smiles(gen_dset)

        pos_refs = []
        pos_gens = []
        edge_indexs = []
        atom_types = []

        for smiles, gen_mols in gen_grouped.items():
            if smiles not in ref_grouped:
                continue
            edge_indexs.append(gen_mols[0].edge_index)
            atom_types.append(gen_mols[0].node_type)
            ref_mols = ref_grouped[smiles]
            
            p_ref = []
            p_gen = []
            for mol in ref_mols:
                p_ref.append(mol.pos.reshape(1, -1, 3).to(self.device))
            for mol in gen_mols:
                p_gen.append(mol.pos.reshape(1, -1, 3).to(self.device))
            pos_refs.append(torch.cat(p_ref, dim=0))
            pos_gens.append(torch.cat(p_gen, dim=0))
        
        return self._run(pos_refs, pos_gens, edge_indexs, atom_types)
    
    def _run(self, pos_refs, pos_gens, edge_indexs, atom_types):
        """
        Args:
            pos_refs:  A list of numpy tensors of shape (num_refs, num_atoms, 3)
            pos_gens:  A list of numpy tensors of shape (num_gens, num_atoms, 3)
            edge_indexs:  A list of LongTensor(E, 2)
            atom_types:   A list of LongTensor(N, )
        """
        s_mmd_all = []
        p_mmd_all = []
        a_mmd_all = []

        for data in tqdm(zip(pos_refs, pos_gens, edge_indexs, atom_types), total=len(pos_refs)):
            stats_single, stats_pair, stats_all = self.func(data)
            s_mmd_all += [e['mmd'] for e in stats_single]
            p_mmd_all += [e['mmd'] for e in stats_pair]
            a_mmd_all.append(stats_all['mmd'])            


        # for result in tqdm(self.pool.imap(self.func, zip(pos_refs, pos_gens, edge_indexs, atom_types)), total=len(pos_refs)):
        #     stats_single, stats_pair, stats_all = result
        #     s_mmd_all += [e['mmd'] for e in stats_single]
        #     p_mmd_all += [e['mmd'] for e in stats_pair]
        #     a_mmd_all.append(stats_all['mmd'])

        return s_mmd_all, p_mmd_all, a_mmd_all
