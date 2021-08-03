import os
import torch
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from .chem import get_atom_symbol
from .transforms import get_standard_transforms
from .metrics import compute_mmd
from .rmsd import GetBestRMSD
from .misc import split_dataset_by_smiles


def compute_pair_rmsds(args, worker_id):
    rets = []
    # print('Chunk size: %d | %s' % (len(args), repr(args[0])))
    if worker_id == 0:
        for idx, probe, ref in tqdm(args, desc='Worker#0'):
            rms, rmsd = GetBestRMSD(probe, ref)
            rets.append((idx, rms, rmsd))
    else:
        for idx, probe, ref in args:
            rms, rmsd = GetBestRMSD(probe, ref)
            rets.append((idx, rms, rmsd))
    return rets


class Evaluator(object):

    def __init__(self, gen, ref, num_workers=64, use_FF=False):
        super().__init__()
        try:
            if isinstance(gen, torch.utils.data.Subset): assert gen.dataset.transform is None
            else: assert gen.transform is None
            if isinstance(ref, torch.utils.data.Subset): assert ref.dataset.transform is None
            else: assert ref.transform is None
        except AssertionError:
            assert False, 'Please use datasets without transforms for the sake of performance. '
        except AttributeError:
            pass

        self.gen = gen
        self.ref = ref
        self.sample = get_standard_transforms(order=3)(self.ref[0])
        smiles = self.sample['smiles']
        for data in tqdm(gen, desc='Check-GEN'): assert data['smiles'] == smiles
        for data in tqdm(ref, desc='Check-REF'): assert data['smiles'] == smiles

        # Compute distances
        edge_index = self.sample.edge_index
        node_type  = self.sample.node_type
        gen_lengths = []
        ref_lengths = []
        for data in tqdm(self.gen, desc='Length-GEN'):
            gen_lengths.append((data.pos[edge_index[0]] - data.pos[edge_index[1]]).norm(dim=1).tolist())
        for data in tqdm(self.ref, desc='Length-REF'):
            ref_lengths.append((data.pos[edge_index[0]] - data.pos[edge_index[1]]).norm(dim=1).tolist())
        gen_lengths = torch.FloatTensor(gen_lengths)
        ref_lengths = torch.FloatTensor(ref_lengths)
        self.gen_lengths = gen_lengths
        self.ref_lengths = ref_lengths

        self.rms_mat = None
        self.rmsd_mat = None

        self.num_workers = num_workers
        self.use_FF = use_FF

    def precompute_rms_rmsd(self):
        if self.rms_mat is not None and self.rmsd_mat is not None:
            return
        # Compute pairwise RMS/RMSD
        num_workers = self.num_workers
        self.rms_mat = np.zeros([len(self.gen), len(self.ref)])
        self.rmsd_mat = np.zeros([len(self.gen), len(self.ref)])
        rmsd_args = []
        for i_gen, data_gen in enumerate(self.gen):
            gen_rdmol = data_gen.rdmol
            if self.use_FF:
                print('Applying FF on generated molecules...')
                MMFFOptimizeMolecule(gen_rdmol)
            for i_ref, data_ref in enumerate(self.ref):

                rmsd_args.append(((i_gen, i_ref), gen_rdmol, data_ref.rdmol))

        # Multi-processing
        chunk_size = len(rmsd_args) // num_workers
        if chunk_size == 0:
            chunk_size = 1
        chunks = [rmsd_args[x:x+chunk_size] for x in range(0, len(rmsd_args), chunk_size)]
        rmsd_tasks = []
        rmsd_pbar = tqdm(total=len(rmsd_args))
        rmsd_pool = mp.Pool(processes=num_workers)
        def pool_callback(retval):
            for idx, rms, rmsd in retval:
                self.rms_mat[idx] = rms
                self.rmsd_mat[idx] = rmsd
            rmsd_pbar.update(len(retval))
        for i, chunk in enumerate(chunks):
            rmsd_tasks.append(rmsd_pool.apply_async(compute_pair_rmsds, (chunk, i), callback=pool_callback))
        for task in rmsd_tasks:
            task.wait()
        rmsd_pbar.close()
        rmsd_pool.close()
        

    def single_dist_distrib(self, ignore_H=True):
        edge_index = self.sample.edge_index
        node_type  = self.sample.node_type
        gen_lengths = self.gen_lengths.cuda()
        ref_lengths = self.ref_lengths.cuda()
        
        stats = []
        for i, (row, col) in enumerate(tqdm(edge_index.t())):
            if row >= col: continue
            if ignore_H and 1 in (node_type[row].item(), node_type[col].item()): continue
            gen_l = gen_lengths[:, i]
            ref_l = ref_lengths[:, i]
            mmd = compute_mmd(gen_l.view(-1, 1), ref_l.view(-1, 1)).item()
            stats.append({
                'edge_id': i,
                'elems': '%s - %s' % (get_atom_symbol(node_type[row].item()), get_atom_symbol(node_type[col].item())),
                'nodes': (row.item(), col.item()),
                'gen_lengths': gen_l.cpu(),
                'ref_lengths': ref_l.cpu(),
                'mmd': mmd
            })

        return stats

    def pair_dist_distrib(self, ignore_H=True):
        edge_index = self.sample.edge_index
        node_type  = self.sample.node_type
        gen_lengths = self.gen_lengths.cuda()
        ref_lengths = self.ref_lengths.cuda()

        stats = []
        for i, (row_i, col_i) in enumerate(tqdm(edge_index.t())):
            if row_i >= col_i: continue
            if ignore_H and 1 in (node_type[row_i].item(), node_type[col_i].item()): continue
            for j, (row_j, col_j) in enumerate(edge_index.t()):
                if (row_i >= row_j) or (row_j >= col_j): continue
                if ignore_H and 1 in (node_type[row_j].item(), node_type[col_j].item()): continue

                gen_L = gen_lengths[:, (i,j)]   # (N, 2)
                ref_L = ref_lengths[:, (i,j)]   # (M, 2)
                mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

                stats.append({
                    'edge_id': (i, j),
                    'elems': (
                        '%s - %s' % (get_atom_symbol(node_type[row_i].item()), get_atom_symbol(node_type[col_i].item())), 
                        '%s - %s' % (get_atom_symbol(node_type[row_j].item()), get_atom_symbol(node_type[col_j].item())),                         
                    ),
                    'nodes': (
                        (row_i.item(), col_i.item()),
                        (row_j.item(), col_j.item()),
                    ),
                    'gen_lengths': gen_L.cpu(),
                    'ref_lengths': ref_L.cpu(),
                    'mmd': mmd
                })
        return stats

    def all_dist_distrib(self, ignore_H=True):
        edge_index = self.sample.edge_index
        node_type  = self.sample.node_type
        gen_lengths = self.gen_lengths.cuda()
        ref_lengths = self.ref_lengths.cuda()

        edge_filter = edge_index[0] < edge_index[1]
        if ignore_H:
            for i, (row, col) in enumerate(edge_index.t()): 
                if 1 in (node_type[row].item(), node_type[col].item()):
                    edge_filter[i] = False

        gen_L = gen_lengths[:, edge_filter]    # (N, Ef)
        ref_L = ref_lengths[:, edge_filter]    # (M, Ef)
        mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

        return {
            'gen_lengths': gen_L.cpu(),
            'ref_lengths': ref_L.cpu(),
            'mmd': mmd
        }

    def min_match_dist(self):
        self.precompute_rms_rmsd()
        mm_rms, mm_rmsd = [], []

        for i_ref in tqdm(range(len(self.ref))):
            min_rms, min_rmsd = float("inf"), float("inf")
            for i_gen in range(len(self.gen)):
                min_rms = min(min_rms, self.rms_mat[(i_gen, i_ref)])
                min_rmsd = min(min_rmsd, self.rmsd_mat[(i_gen, i_ref)])
            mm_rms.append(min_rms)
            mm_rmsd.append(min_rmsd)
        return mm_rms, mm_rmsd


    def coverage(self):
        self.precompute_rms_rmsd()
        covered_rms = set()
        covered_rmsd = set()
        
        for i_gen in range(len(self.gen)):
            min_rms, min_rmsd = float("inf"), float("inf")
            min_rms_id, min_rmsd_id = -1, -1
            for i_ref in range(len(self.ref)):
                rms, rmsd = self.rms_mat[(i_gen, i_ref)], self.rmsd_mat[(i_gen, i_ref)]
                if rms < min_rms:
                    min_rms, min_rms_id = rms, i_ref
                if rmsd < min_rmsd:
                    min_rmsd, min_rmsd_id = rmsd, i_ref
            covered_rms.add(min_rms_id)
            covered_rmsd.add(min_rmsd_id)
        
        cov_rms = len(covered_rms) / len(self.ref)
        cov_rmsd = len(covered_rmsd) / len(self.ref)
        return cov_rms, cov_rmsd

    def one_nna(self):
        pass


class EvaluationSession(object):

    def __init__(self, gen_dset, ref_dset, out_dir, logger, ignore_H=True, eval_match=True, gen_limit=0, use_FF=False):
        super().__init__()
        self.gen_dset = gen_dset
        self.ref_dset = ref_dset
        self.out_dir = out_dir
        self.logger = logger
        self.ignore_H = ignore_H
        self.eval_match = eval_match
        self.use_FF = use_FF

        self.ref_grouped = split_dataset_by_smiles(ref_dset)
        self.gen_grouped = split_dataset_by_smiles(gen_dset)
        if gen_limit < 0:
            gen_mult = -1 * gen_limit
            for smiles in self.gen_grouped:
                subset_new = []
                if smiles not in self.ref_grouped:
                    continue
                for i in range(min(int(gen_mult * len(self.ref_grouped[smiles])), len(self.gen_grouped[smiles]))):
                    subset_new.append(self.gen_grouped[smiles][i])
                logger.info('Truncate: %d -> %d' % (len(self.gen_grouped[smiles]), len(subset_new)))
                self.gen_grouped[smiles] = subset_new

        count_gen_conf = sum([len(g) for _, g in self.gen_grouped.items()])

        logger.info('Ref %d items | Gen %d items' % (len(self.ref_dset), count_gen_conf))
        logger.info('Ref %d mols | Gen %d mols' % (len(self.ref_grouped), len(self.gen_grouped)))

    def print_stats(self, title, data):
        stat_funcs = (np.mean, np.median, np.min, np.max)
        self.logger.info('[' + title + '] Mean %.4f | Median %.4f | Min %.4f | Max %.4f' % tuple([
            f(data) for f in stat_funcs
        ]))

    def run(self):
        # shortcuts
        logger = self.logger
        gen_grouped = self.gen_grouped
        ref_grouped = self.ref_grouped

        logger.info('Start eval...')
        s_mmd_all, p_mmd_all, a_mmd_all = [], [], []
        cov_all, match_dist_all = [], []
        results = {}

        i = 0

        all_cov_thr = {}

        for smiles, gen_mols in tqdm(gen_grouped.items(), 'All'):
            i += 1
            if smiles not in ref_grouped:
                logger.warning('Molecule not found in refset: %s' % smiles)
                continue
            ref_mols = ref_grouped[smiles]
            
            evaluator = Evaluator(gen_mols, ref_mols, use_FF=self.use_FF)
            s_dist = evaluator.single_dist_distrib(ignore_H=self.ignore_H)
            p_dist = evaluator.pair_dist_distrib(ignore_H=self.ignore_H)
            a_dist = evaluator.all_dist_distrib(ignore_H=self.ignore_H)
            s_mmd = [e['mmd'] for e in s_dist]
            p_mmd = [e['mmd'] for e in p_dist]
            a_mmd = a_dist['mmd']

            s_mmd_all += s_mmd
            p_mmd_all += p_mmd
            a_mmd_all.append(a_mmd)

            result = {
                's_dist': s_dist,
                'p_dist': p_dist,
                'a_dist': a_dist,
                's_mmd': s_mmd,
                'p_mmd': p_mmd,
                'a_mmd': a_mmd,
            }

            if self.eval_match:
                cov = evaluator.coverage()
                match_dist = evaluator.min_match_dist()
                result['coverage'] = cov
                result['match_dist'] = match_dist
                result['pairwise_rmsd'] = evaluator.rmsd_mat
                result['N_sample'] = len(gen_mols)
                result['N_ref'] = len(ref_mols)
                cov_all.append(cov[0])
                match_dist_all.append(match_dist[0])
                logger.info('[Match1] (%d) Coverage %.4f | Match(Median) %.4f | Match(Mean) %.4f | Match(Min) %.4f | Match(Max) %.4f' % (
                    i, cov[0], np.median(match_dist[0]), np.mean(match_dist[0]), np.min(match_dist[0]), np.max(match_dist[0])
                ))
                # logger.info('[Match2] Coverage %.4f | Match(Median) %.4f | Match(Mean) %.4f | Match(Min) %.4f' % (
                #     cov[1], np.median(match_dist[1]), np.mean(match_dist[1]), np.min(match_dist[1])
                # ))

                # Coverage with thresholds
                cur_cov_thr = {}
                for thr in np.linspace(0, 2, 41):
                    rmsd_mat_min = evaluator.rmsd_mat.min(axis=0)
                    assert rmsd_mat_min.shape[0] == len(ref_mols)
                    cur_cov_thr[thr] = (evaluator.rmsd_mat.min(axis=0) < thr).mean()
                all_cov_thr[smiles] = cur_cov_thr
                
            results[smiles] = result

        if self.eval_match:
            print(all_cov_thr)
            all_cov_thr_pd = pd.DataFrame(all_cov_thr)
        else:
            all_cov_thr_pd = None

        save_path = os.path.join(self.out_dir, 'report.pkl')
        logger.info('Saving results to %s' % save_path)
        report = {
            's_mmd_all': s_mmd_all,
            'p_mmd_all': p_mmd_all,
            'a_mmd_all': a_mmd_all,
            'cov_all': cov_all,
            'match_dist_all': match_dist_all,
            'mols': results,
            'all_cov_thr': all_cov_thr_pd,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(report, f)
        
        logger.info('%s SUMMARY %s' % ('='*10, '='*10))
        # logger.info('[Median] s_dist %.4f | p_dist %.4f | a_dist %.4f' % (
        #     np.median(s_mmd_all), np.median(p_mmd_all), np.median(a_mmd_all),
        # ))
        self.print_stats('SingleDist', s_mmd_all)
        self.print_stats('PairDist', p_mmd_all)
        self.print_stats('AllDist', a_mmd_all)

        aggr_match_dist = []
        for d in report['match_dist_all']:
            aggr_match_dist += d
        if self.eval_match:
            logger.info('%s Coverage / Match %s' % ('-'*10, '-'*10))
            self.print_stats('Coverage',  report['cov_all'])
            self.print_stats('MinMatchDist(Aggr)', aggr_match_dist)
            self.print_stats('MinMatchDist(Min)', [np.min(d) for d in report['match_dist_all']])
            self.print_stats('MinMatchDist(Mean)', [np.mean(d) for d in report['match_dist_all']])
            self.print_stats('MinMatchDist(Median)', [np.median(d) for d in report['match_dist_all']])
            self.print_stats('SingleMMD', report['s_mmd_all'])
            self.print_stats('PairMMD',   report['p_mmd_all'])
            self.print_stats('AllMMD',    report['a_mmd_all'])

            if all_cov_thr_pd is not None:
                logger.info('CovThr Mean' + repr(all_cov_thr_pd.mean(axis=1)))
                logger.info('CovThr Median' + repr(all_cov_thr_pd.median(axis=1)))



