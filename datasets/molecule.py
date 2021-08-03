import os
import pickle
import copy
import torch
from torch.utils.data import Dataset
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm.auto import tqdm
RDLogger.DisableLog('rdApp.*')

from utils.chem import BOND_TYPES, BOND_NAMES


def rdmol_to_data(mol:Mol):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    smiles = Chem.MolToSmiles(mol)

    data = Data(node_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    data.nx = to_networkx(data, to_undirected=True)

    return data


def enumerate_conformers(mol):
    num_confs = mol.GetNumConformers()
    if num_confs == 1:
        yield mol
        return
    mol_templ = copy.deepcopy(mol)
    mol_templ.RemoveAllConformers()
    for conf_id in tqdm(range(num_confs), desc='Conformer'):
        conf = mol.GetConformer(conf_id)
        conf.SetId(0)
        mol_conf = copy.deepcopy(mol_templ)
        conf_id = mol_conf.AddConformer(conf)
        yield mol_conf


class MoleculeDataset(Dataset):

    def __init__(self, raw_path, force_reload=False, transform=None):
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = raw_path + '.processed'
        self.transform = transform

        _, extname = os.path.splitext(raw_path)
        assert extname in ('.sdf', '.pkl'), 'Only supports .sdf and .pkl files'

        self.dataset = None
        if force_reload or not os.path.exists(self.processed_path):
            if extname == '.sdf':
                self.process_sdf()
            elif extname == '.pkl':
                self.process_pickle()
        else:
            self.load_processed()

    def load_processed(self):
        self.dataset = torch.load(self.processed_path)

    def process_sdf(self):
        self.dataset = []
        suppl = Chem.SDMolSupplier(self.raw_path, removeHs=False, sanitize=True)
        for mol in tqdm(suppl):
            if mol is None:
                continue
            for conf in enumerate_conformers(mol):
                self.dataset.append(rdmol_to_data(conf))
        torch.save(self.dataset, self.processed_path)

    def process_pickle(self):
        self.dataset = []
        with open(self.raw_path, 'rb') as f:
            mols = pickle.load(f)
            for mol in tqdm(mols):
                for conf in enumerate_conformers(mol):
                    self.dataset.append(rdmol_to_data(conf))
            torch.save(self.dataset, self.processed_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data
