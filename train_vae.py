import os
import argparse
import torch
import torch.utils.tensorboard
from torch_geometric.data import Batch, DataLoader
from tqdm.auto import tqdm
import time

from models.edgecnf import *
from models.cnf_edge import NONLINEARITIES, LAYERS, SOLVERS
from datasets import *
from utils.transforms import *
from utils.misc import *
from models.vae import ImplicitVAE
from utils.chem import ParallelAlignMol

import pdb

# Arguments
parser = argparse.ArgumentParser()
# BEGIN
# Model arguments
parser.add_argument('--activation', type=str, default='softplus')
parser.add_argument('--hidden_dim', type=int, default=64)
# Encoder
parser.add_argument('--latent_dim', type=int, default=8, 
                    help='Size of latent variable for each node')
parser.add_argument('--use_deterministic_encoder', action='store_true', default=False,
                    help='Whether to use a deterministic encoder.')
# Decoder
parser.add_argument('--kl_weight', type=float, default=1.,
                    help='weight for the kl loss')  # 1 for QM9; 0.1 for Drugs
parser.add_argument('--implicit_weight', type=float, default=.1,
                    help='weight for the implicit loss')
parser.add_argument("--num_blocks", type=int, default=1,
                    help='Number of stacked CNFs.')
parser.add_argument('--use_implicit_loss', action='store_true', default=False,
                    help='Whether to use the implicit RMSD loss.')
parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)
parser.add_argument('--spectral_norm', type=eval, default=True, choices=[True, False])
parser.add_argument('--train_noise_std', type=float, default=0.1)

# Datasets and loaders
parser.add_argument('--aux_edge_order', type=int, default=3)
parser.add_argument('--train_dataset', type=str, default='./data/qm9/QM9_train.pkl')
parser.add_argument('--test_dataset', type=str, default='./data/qm9/QM9_test.pkl')
parser.add_argument('--val_dataset', type=str, default='./data/qm9/QM9_val.pkl')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--sched_factor', type=float, default=0.5)
parser.add_argument('--sched_patience', type=int, default=3,
                    help='Patience steps = sched_patience * val_freq')
parser.add_argument('--sched_min_lr', type=int, default=1e-5)
parser.add_argument('--beta1', type=float, default=0.95)
parser.add_argument('--beta2', type=float, default=0.999)

# Load Pre-train Model
parser.add_argument('--load_cnf_path', type=str, default='./logs/ckpt_-0.943000_44700.pt',
                    help='Path for loading pre-trained CNF model')

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--prefix', type=str, default='VAE')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=1000*1000, 
                    help='Max iterations for MLE pre-training of CNF')
parser.add_argument('--val_freq', type=int, default=300)
parser.add_argument('--inspect_freq', type=int, default=50)
parser.add_argument('--resume', type=str, default=None)
# END
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(prefix=args.prefix, tag=args.tag)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir, device=args.device)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading dataset...')
tf = get_standard_transforms(order=args.aux_edge_order)
train_dset = MoleculeDataset(args.train_dataset, transform=tf)
train_iterator = get_data_iterator(DataLoader(train_dset,
                        batch_size=args.train_batch_size, 
                        shuffle=True, 
                        drop_last=True, 
                        num_workers=args.num_workers))
if args.val_dataset is not None:
    val_dset = MoleculeDataset(args.val_dataset, transform=tf)
    val_loader = DataLoader(val_dset, 
                            batch_size=args.val_batch_size, 
                            shuffle=False, drop_last=True,
                            num_workers=args.num_workers)
    logger.info('TrainSet %d | ValSet %d' % (len(train_dset), len(val_dset)))
else:
    logger.info('TrainSet %d' % (len(train_dset)))

# Model
logger.info('Building model...')
if args.resume is None:
    model = ImplicitVAE(args).to(args.device)
    if args.spectral_norm:
        add_spectral_norm(model.decoder, logger=logger)
else:
    logger.info('Resuming from %s' % args.resume)
    ckpt_resume = CheckpointManager(args.resume, logger=logger, device=args.device).load_latest()
    # model = ImplicitVAE(ckpt_resume['args']).to(args.device)
    model = ImplicitVAE(args).to(args.device)
    if ckpt_resume['args'].spectral_norm:
        add_spectral_norm(model.decoder, logger=logger)
    model.load_state_dict(ckpt_resume['state_dict'])
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay,
    betas=(args.beta1, args.beta2)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    factor=args.sched_factor,
    patience=args.sched_patience,
    min_lr=args.sched_min_lr
)

# Train and validation
def train(it, use_implicit_loss=False):
    time_start = time.time()
    model.train()
    optimizer.zero_grad()
    if args.spectral_norm:
        spectral_norm_power_iteration(model.decoder, n_power_iterations=1)
    batch = next(train_iterator).to(args.device)
    noise = torch.randn_like(batch.edge_length) * args.train_noise_std
    loss, loss_kl, loss_rec_d, loss_rec_x = model.get_loss(batch, batch.edge_length + noise, use_implicit=use_implicit_loss)
    nfe_forward = count_nfe(model)

    # pdb.set_trace()
    loss.backward()
    optimizer.step()

    nfe_total = count_nfe(model)
    nfe_backward = nfe_total - nfe_forward
    time_end = time.time()
    
    logger.info('[Train] Iter %04d | NFE_Forward %d | NFE_Backward %d | Time %.2f' % (it, nfe_forward, nfe_backward, time_end-time_start))
    if use_implicit_loss:
        logger.info('[Train] Loss %04d | Loss %.4f | KL %.4f | Rec_D %.4f | Rec_X %.4f' % (it, loss.item(), loss_kl.item(), loss_rec_d.item(), loss_rec_x.item()))
    else:
        logger.info('[Train] Loss %04d | Loss %.4f | KL %.4f | Rec_D %.4f' % (it, loss.item(), loss_kl.item(), loss_rec_d.item()))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/loss_kl', loss_kl, it)
    writer.add_scalar('train/loss_rec_d', loss_rec_d, it)
    if use_implicit_loss:
        writer.add_scalar('train/loss_rec_x', loss_rec_x, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/nfe_forward', nfe_forward, it)
    writer.add_scalar('train/nfe_backward', nfe_backward, it)
    writer.flush()

def validate(it, use_implicit_loss=False):
    # with torch.no_grad():
    sum_loss_kl = 0.
    sum_loss_rec_d = 0.
    sum_loss_rec_x = 0.
    sum_n = 0.
    model.eval()
    for batch in tqdm(val_loader, desc='Validating'):
        batch = batch.to(args.device)
        loss_kl, loss_rec_d, loss_rec_x = model.get_nll(batch, batch.edge_length, use_implicit=use_implicit_loss, eval=True)
        if use_implicit_loss:
            sum_loss_kl += model.kl_weight*loss_kl.detach().sum().item()
            sum_loss_rec_d += loss_rec_d.detach().sum().item()
            sum_loss_rec_x += model.implicit_weight*loss_rec_x.detach().sum().item()
        else:
            sum_loss_kl += model.kl_weight*loss_kl.detach().sum().item()
            sum_loss_rec_d += loss_rec_d.detach().sum().item()
        sum_n += batch.batch[-1]+1
    avg_loss_kl = sum_loss_kl / sum_n
    avg_loss_rec_d = sum_loss_rec_d / sum_n
    avg_loss_rec_x = sum_loss_rec_x / sum_n
    avg_loss = avg_loss_kl + avg_loss_rec_d + avg_loss_rec_x

    scheduler.step(avg_loss)

    logger.info('[Validate] Iter %04d | Loss %.6f ' % (it, avg_loss))
    writer.add_scalar('val/loss', avg_loss, it)
    logger.info('[Validate] Iter %04d | Loss_kl %.6f ' % (it, avg_loss_kl))
    writer.add_scalar('val/loss_kl', avg_loss_kl, it)
    logger.info('[Validate] Iter %04d | Loss_rec_d %.6f ' % (it, avg_loss_rec_d))
    writer.add_scalar('val/loss_rec_d', avg_loss_rec_d, it)
    if use_implicit_loss:
        logger.info('[Validate] Iter %04d | Loss_rec_x %.6f ' % (it, avg_loss_rec_x))
        writer.add_scalar('val/loss_rec_x', avg_loss_rec_x, it)
    writer.flush()
    return avg_loss

def inspect(it):
    logger.info('[Inspect] Sampling edge lengths...')
    with torch.no_grad():
        molecule = Batch.from_data_list([val_dset[0]]).to(args.device)
        model.eval()
        samples, _ = model.sample(molecule, num_samples=500)       # (E, num_samples)
        writer.add_embedding(samples, global_step=it, tag='edge_length')
        for i, edge_name in enumerate(molecule.edge_name[0]):   # Only one molecule
            if edge_name == '':
                continue
            mean = samples[i].mean().item()
            std = samples[i].std().item()
            name_seg = edge_name.split('_')
            logger.info('[Inspect] (%d) %s %s-%s | Dist %.6f | Mean %.6f | Std %.6f' % (
                i,
                name_seg[0],
                name_seg[1],
                name_seg[2],
                molecule.edge_length[i].item(),
                mean,
                std,
            ))
            writer.add_histogram('length/' + edge_name, samples[i], it)
        writer.flush()
    with torch.no_grad():
        model.eval()
        model.use_deterministic_encoder = True
        model.decoder.use_deterministic_encoder = True
        samples, _ = model.sample(molecule, num_samples=500)       # (E, num_samples)
        writer.add_embedding(samples, global_step=it, tag='edge_length_deterministic')
        for i, edge_name in enumerate(molecule.edge_name[0]):   # Only one molecule
            if edge_name == '':
                continue
            mean = samples[i].mean().item()
            std = samples[i].std().item()
            name_seg = edge_name.split('_')
            logger.info('[Inspect] (%d) %s %s-%s | Dist %.6f | Mean %.6f | Std %.6f' % (
                i,
                name_seg[0],
                name_seg[1],
                name_seg[2],
                molecule.edge_length[i].item(),
                mean,
                std,
            ))
            writer.add_histogram('length_deterministic/' + edge_name, samples[i], it)
        writer.flush()
        model.use_deterministic_encoder = False
        model.decoder.use_deterministic_encoder = False

# Main loop
logger.info('Start training...')
try:
    if args.resume is not None:
        start_it = ckpt_resume['iteration'] + 1
    else:
        start_it = 1
    for it in range(start_it, args.max_iters+1):
        train(it, use_implicit_loss=args.use_implicit_loss)
        if args.val_dataset is not None:
            if it % args.val_freq == 0 or it == args.max_iters:
                avg_val_loss = validate(it, use_implicit_loss=args.use_implicit_loss)
                ckpt_mgr.save(model, args, avg_val_loss, it)
            # if it % args.inspect_freq == 0:
            #     inspect(it)
        else:
            if it % args.val_freq == 0 or it == args.max_iters:
                ckpt_mgr.save(model, args, avg_val_loss, it)

except KeyboardInterrupt:
    logger.info('Terminating...')
