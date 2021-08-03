import os
import time
import argparse
import torch
import pickle
import pandas as pd
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from models.edgecnf import *
from models.vae import *
from datasets import *
from utils.chem import *
from utils.misc import *
from utils.transforms import *
from utils.rmoutlier import *
from utils.evaluation import EvaluationSession
from utils.eval import CovMatEvaluator, DistEvaluator

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_important/ECNF_2020_08_21__13_31_32_B128N0.1_QM9')
parser.add_argument('--dataset', type=str, default='./data/ISO17Conf/iso17_split-0_test.pkl')
parser.add_argument('--out', type=str, default='./output')
parser.add_argument('--prefix', type=str)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--em_steps', type=int, default=0)
# parser.add_argument('--eval_match', type=eval, required=True, choices=[True, False])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--emb_step_size', type=float, default=3.0) # 3.0 for QM9, 5.0 for ISO17
parser.add_argument('--emb_num_steps', type=int, default=1000)
parser.add_argument('--emb_optim', type=str, default='Adam')
parser.add_argument('--rmoutlier', type=eval, default=False)
parser.add_argument('--num_samples_real', type=int, default=None)
parser.add_argument('--outlier_std', type=float, default=2.5)
parser.add_argument('--mmff', type=eval, default=False)
parser.add_argument('--deterministic_sampling', action='store_true', default=False,
                    help='Whether to use a deterministic sampling procedure.')
parser.add_argument('--eval_only', action='store_true', default=False)
args = parser.parse_args()

if args.eval_only:
    save_path = os.path.join(args.out, 'mols.pkl')
    logger = get_logger('eval', log_dir=args.out)
    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))
else:
    # Output and Logging
    model_name = 'VAE'
    if args.mmff:
        model_name += 'mmff'
    out_dir = os.path.join(args.out, '%s_%s_%d%s' % (
        args.prefix, model_name, int(time.time()), ('_' if len(args.tag) > 0 else '') + args.tag
    ))
    os.makedirs(out_dir, exist_ok=False)
    logger = get_logger('gen', log_dir=out_dir)
    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))

    # Model
    logger.info('Loading VAE...')
    ckpt = CheckpointManager(args.ckpt).load_latest()
    args_old = ckpt['args']
    # args_old.implicit_weight = 10.
    model = ImplicitVAE(args_old).to(args.device)
    if ckpt['args'].spectral_norm:
        add_spectral_norm(model.decoder)
    model.load_state_dict(ckpt['state_dict'])
    if args.deterministic_sampling:
        model.use_deterministic_encoder = True
        model.decoder.use_deterministic_encoder = False

    # Test Dataset
    logger.info('Loading test-set: %s' % args.dataset)
    tf = get_standard_transforms(ckpt['args'].aux_edge_order)
    test_dset = MoleculeDataset(args.dataset, transform=tf)
    grouped = split_dataset_by_smiles(test_dset)
    loader = DataLoader(VirtualDataset(grouped, args.num_samples), batch_size=args.batch_size, shuffle=False)

    # Output buffer
    gen_rdmols = []

    # DistGeom Embedder
    embedder = Embed3D(step_size=args.emb_step_size, num_steps=args.emb_num_steps)

    # Generate
    all_data_list = []
    for batch in tqdm(loader):
        batch = batch.to(args.device)
        pos_s = em_generate_batch(
                        model, 
                        batch, 
                        num_samples=1, 
                        embedder=embedder, 
                        em_steps=args.em_steps)[0]  # (1, BN, 3)
        batch.pos = pos_s[0]
        batch.to('cpu')
        batch_list = batch.to_data_list()
        all_data_list += batch_list

    grouped_data = split_dataset_by_smiles(all_data_list)
    for smiles in tqdm(grouped_data, 'RmOutliers'):
        if args.rmoutlier:
            grouped_data[smiles] = remove_outliers(grouped_data[smiles], args.outlier_std)
            if args.num_samples_real is not None:
                if args.num_samples_real > 0:
                    nsample_real = args.num_samples_real
                else:
                    nsample_real = -1 * args.num_samples_real * len(grouped[smiles])
                grouped_data[smiles] = grouped_data[smiles][:nsample_real]
        for data in grouped_data[smiles]:
            rdmol = data['rdmol']
            rdmol = set_rdmol_positions_(rdmol, data.pos.cpu())
            gen_rdmols.append(rdmol)


    # Optimize using MMFF
    opt_rdmols = []
    if args.mmff:
        for mol in tqdm(gen_rdmols, desc='MMFF'):
            opt_mol = deepcopy(mol)
            MMFFOptimizeMolecule(opt_mol)
            opt_rdmols.append(opt_mol)
        gen_rdmols = opt_rdmols

    # Save
    save_path = os.path.join(out_dir, 'mols.pkl')
    logger.info('Saving to: %s' % save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(gen_rdmols, f)

# Evaluate
gen_dset = MoleculeDataset(save_path)
ref_dset = MoleculeDataset(args.dataset)


# MAT/COV
thresholds = [0.5, 1.25]
evaluator = CovMatEvaluator(thresholds=thresholds)
# Run evaluation
results = evaluator(ref_dset, gen_dset)
covs = np.asarray(results[0])
mats = np.asarray(results[1])
for i in range(len(thresholds)):
    logger.info('Threshold %.6f: COV(Mean) %.6f, COV(Median) %.6f' % (
            thresholds[i], 
            covs.mean(axis=0)[i],
            np.median(covs, axis=0)[i],
        ))
logger.info('MAT(Mean) %.6f, MAT(Median) %.6f' % (
        mats.mean(axis=0),
        np.median(mats, axis=0),
    ))


# # MAT
# thresholds = np.linspace(0, 2, 41)
# evaluator = CovMatEvaluator(thresholds=thresholds)
# # Run evaluation
# results = evaluator(ref_dset, gen_dset)
# covs = np.asarray(results[0])

# gen_grouped = list(split_dataset_by_smiles(gen_dset).items())

# all_cov_thr = {}
# for i in range(min(len(gen_grouped), len(covs))):
#     smiles, gen_mols = gen_grouped[i]
#     cur_cov_thr = {}
#     for j in range(len(thresholds)):
#         threshold = thresholds[j]
#         cur_cov_thr[threshold] = covs[i][j]
#     all_cov_thr[smiles] = cur_cov_thr
# all_cov_thr_pd = pd.DataFrame(all_cov_thr)

# save_path = os.path.join(args.out, 'report.pkl')
# logger.info('Saving results to %s' % save_path)
# with open(save_path, 'wb') as f:
#     pickle.dump(all_cov_thr_pd, f)


# # Dist
# evaluator = DistEvaluator(ignore_H=True)
# # Run evaluation
# results = evaluator(ref_dset, gen_dset)
# s_mmd_all = np.asarray(results[0])
# p_mmd_all = np.asarray(results[1])
# a_mmd_all = np.asarray(results[2])
# logger.info('single(Mean) %.6f, single(Median) %.6f' % (
#         np.mean(s_mmd_all, axis=0),
#         np.median(s_mmd_all, axis=0),
#     ))
# logger.info('pair(Mean) %.6f, pair(Median) %.6f' % (
#         np.mean(p_mmd_all, axis=0),
#         np.median(p_mmd_all, axis=0),
#     ))
# logger.info('all(Mean) %.6f, all(Median) %.6f' % (
#         np.mean(a_mmd_all, axis=0),
#         np.median(a_mmd_all, axis=0),
#     ))
