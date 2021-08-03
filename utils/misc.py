import os
import time
import random
import logging
import torch
import rdkit
import numpy as np
from logging import Logger
from tqdm.auto import tqdm

from datasets import MoleculeDataset, GEOMDataset


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, best_k=5, logger=BlackHole(), device='cuda'):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.best_k = best_k
        self.ckpts = []
        self.logger = logger
        self.device = device

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

        for _ in range(max(self.best_k - len(self.ckpts), 0)):
            self.ckpts.append({
                'score': torch.tensor(float('inf')),
                'file': None,
                'iteration': -1,
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = torch.tensor(float('-inf'))
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = torch.tensor(float('inf'))
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def save(self, model, args, score, step=None):
        idx = self.get_worst_ckpt_idx()
        if idx is None:
            return False

        if step is None:
            fname = 'ckpt_%.6f_.pt' % float(score)
        else:
            fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
        }, path)

        self.ckpts[idx] = {
            'score': score,
            'file': fname
        }

        return True

    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')

        self.logger.info(repr(self.ckpts[idx]))
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']), map_location=self.device)
        ckpt['iteration'] = self.ckpts[idx]['iteration']
        return ckpt

    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        self.logger.info(repr(self.ckpts[idx]))
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']), map_location=self.device)
        ckpt['iteration'] = self.ckpts[idx]['iteration']
        return ckpt

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def get_data_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def split_dataset_by_smiles(dataset):
    split = {}

    if isinstance(dataset, MoleculeDataset) or isinstance(dataset, GEOMDataset):
        dset = dataset.dataset
    else:
        dset = dataset
    for i, data in enumerate(tqdm(dset)):
        smiles = data.smiles
        if smiles in split:
            split[smiles].append(i)
        else:
            split[smiles] = [i]

    split = {k:torch.utils.data.Subset(dataset, v) for k, v in split.items()}
    return split


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


class VirtualDataset(torch.utils.data.Dataset):
    
    def __init__(self, grouped, num_samples):
        super().__init__()
        self.grouped = [subset for _, subset in grouped.items()]
        assert isinstance(num_samples, int) and num_samples != 0
        self.num_samples = num_samples
        if num_samples < 0:
            self.multiple = -num_samples
            self.splits = [0]
            for g in self.grouped:
                self.splits.append(self.splits[len(self.splits)-1] + len(g) * self.multiple)
            self.splits.reverse()

    def __len__(self):
        if self.num_samples > 0:
            return len(self.grouped) * self.num_samples
        else:
            return self.splits[0]

    def __getitem__(self, idx):
        if self.num_samples > 0:
            gid = idx // self.num_samples
            return self.grouped[gid][0]
        else:
            for gid, offset in enumerate(self.splits):
                if idx >= offset:
                    return self.grouped[len(self.grouped) - gid][0]
