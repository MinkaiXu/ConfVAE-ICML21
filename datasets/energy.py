import random
import torch
import numpy as np
from .molecule import *


class GEOMDataset(Dataset):

    def __init__(self, raw_path, force_reload=False, transform=None, size_limit=100000):
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = raw_path + '.processed'
        self.transform = transform
        self.size_limit = size_limit

        _, extname = os.path.splitext(raw_path)
        assert extname == '.pkl', 'Only .pkl files are supported.'

        self.dataset = None
        if force_reload or not os.path.exists(self.processed_path):
            self.process()
        else:
            self.load_processed()

    def load_processed(self):
        saved = torch.load(self.processed_path)
        self.dataset = saved['dataset']
        self.stats = saved['stats']

    def process(self):
        self.dataset = []
        with open(self.raw_path, 'rb') as f:
            mols_db = pickle.load(f)

        # Statistics buffer
        energies = []
        lowesteners = []
        ensembleeners = []
        
        for mol_meta in tqdm(mols_db):
            for conf_meta in mol_meta['conformers']:
                data = rdmol_to_data(conf_meta['rd_mol'])
                labels = {
                    'ensembleenergy': mol_meta['ensembleenergy'],
                    'ensembleentropy': mol_meta['ensembleentropy'],
                    'ensemblefreeenergy': mol_meta['ensemblefreeenergy'],
                    'lowestenergy': mol_meta['lowestenergy'],
                    'totalenergy': conf_meta['totalenergy'],
                    'degeneracy': conf_meta['degeneracy'],
                    'relativeenergy': conf_meta['relativeenergy'],
                    'boltzmannweight': conf_meta['boltzmannweight'],
                    'conformerweight': conf_meta['conformerweights'][0],
                }
                for k, v in labels.items():
                    data[k] = torch.FloatTensor([v])
                self.dataset.append(data)
                energies.append(conf_meta['totalenergy'])
            # end for
            lowesteners.append(mol_meta['lowestenergy'])
            ensembleeners.append(mol_meta['ensembleenergy'])

        if self.size_limit > 0:
            random.Random(2020).shuffle(self.dataset)
            self.dataset = self.dataset[:self.size_limit]

        # Compute statistics
        self.stats = {
            'totalenergy': {'mean': np.mean(energies), 'std': np.std(energies)},
            'lowestenergy': {'mean': np.mean(lowesteners), 'std': np.std(lowesteners)},
            'ensembleenergy': {'mean': np.mean(ensembleeners), 'std': np.std(ensembleeners)},
        }

        torch.save({
            'dataset': self.dataset,
            'stats': self.stats
        }, self.processed_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data
