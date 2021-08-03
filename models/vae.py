import torch
from torch import Tensor
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter

from .common import *
from .cnf_edge import CNF, ODEfunc, ODEgnn, MovingBatchNorm1d, SequentialFlow, count_nfe, add_spectral_norm, spectral_norm_power_iteration
from .distgeom import *

import copy
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem.rdMolAlign import AlignMol, GetBestRMS
from utils.chem import *
import time
import pdb


class ImplicitEmbed3D(object):

    def __init__(self, alpha=0.5, mu=0, step_size=0.07, num_steps=200, logger=None):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.step_size = step_size
        self.num_steps = num_steps
        self.logger = logger

    def __call__(self, d_target, edge_index, init_pos, edge_order=None):
        return diff_embed_3D(
            d_target, edge_index, init_pos, edge_order,
            alpha=self.alpha,
            mu=self.mu,
            step_size=self.step_size,
            num_steps=self.num_steps,
            logger=self.logger
        )

def diff_embed_3D(d_target, edge_index, init_pos, edge_order=None, alpha=0.5, mu=0, step_size=None, num_steps=None, logger=None):
    assert torch.is_grad_enabled, '`embed_3D` requires gradients'
    step_size = 10.0 if step_size is None else step_size
    num_steps = 200 if num_steps is None else num_steps

    # d_target = d_target.view(-1)
    # pos = init_pos.clone().requires_grad_(True)
    # optimizer = torch.optim.SGD([pos], lr=step_size)

    if edge_order is not None:
        coef = alpha ** (edge_order.view(-1).float() - 1)
    else:
        coef = 1.0
    
    if mu > 0:
        noise = torch.randn_like(coef) * coef * mu + coef
        coef = torch.clamp_min(coef + noise, min=0)

    pos = init_pos.clone().requires_grad_(True).to(d_target)
    optimizer = torch.optim.Adam([pos], lr=step_size)

    for i in range(num_steps):
        if i >= num_steps-1:
            # optimizer.zero_grad()
            pos_old = pos.clone().requires_grad_(True).to(d_target)
            d_old = torch.norm(pos_old[edge_index[0]] - pos_old[edge_index[1]], dim=1)
            loss = (coef * ((d_target - d_old) ** 2)).mean()
            # loss.backward()
            pos_grad = torch.autograd.grad(loss, pos_old, create_graph=True)[0]
            # pos.grad = pos_grad
            # optimizer.step()
            pos = pos.add(pos_grad, alpha= -step_size)
        else:
            optimizer.zero_grad()
            d_old = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
            loss = (coef * ((d_target - d_old) ** 2)).mean()
            pos_grad = torch.autograd.grad(loss, pos)[0]
            pos.grad = pos_grad
            optimizer.step()
    if logger is not None:
        logger.info('Embed 3D: AvgLoss %.6f' % (loss.item() / d_target.size(0)))

    return pos, loss.detach() / d_target.size(0)


class GConv(MessagePassing):

    def __init__(self, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super(GConv, self).__init__(aggr='add', **kwargs)
        # self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
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

        # return self.nn(t, out)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.softplus(x_j + edge_attr)


class GNNPrior(torch.nn.Module):

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.act = F.softplus
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_emb = torch.nn.Embedding(100, hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, hidden_dim)

        self.conv1 = GConv()
        self.bn_conv1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GConv()
        self.bn_conv2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GConv()
        self.bn_conv3 = torch.nn.BatchNorm1d(hidden_dim)

        self.out_fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn_out1 = torch.nn.BatchNorm1d(hidden_dim)
        self.out_fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_out2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.out_fc3 = torch.nn.Linear(hidden_dim // 2, latent_dim*2)


    def forward(self, node_type, edge_type, edge_index, batch):

        node_attr = self.node_emb(node_type)
        edge_attr = self.edge_emb(edge_type)

        h = node_attr
        h = self.act(self.bn_conv1(self.conv1(h, edge_index, edge_attr)))
        h = self.act(self.bn_conv2(self.conv2(h, edge_index, edge_attr)))
        h = self.bn_conv3(self.conv3(h, edge_index, edge_attr))

        h_global = scatter(h, batch, dim=0, reduce='sum')
        node_feat = torch.cat([h, h_global[batch]], dim=-1)
        node_feat = self.act(self.bn_out1(self.out_fc1(node_feat)))
        node_feat = self.act(self.bn_out2(self.out_fc2(node_feat)))
        out = self.out_fc3(node_feat)

        mu = out[:, :self.latent_dim]
        sigma = out[:, self.latent_dim:]

        return mu, sigma


class GNNEncoder(torch.nn.Module):

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.act = F.softplus
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_emb = torch.nn.Embedding(100, hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, hidden_dim)

        self.d_fc1 = torch.nn.Linear(1, hidden_dim)
        self.bn_d1 = torch.nn.BatchNorm1d(hidden_dim)
        self.d_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn_d2 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv1 = GConv()
        self.bn_conv1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GConv()
        self.bn_conv2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GConv()
        self.bn_conv3 = torch.nn.BatchNorm1d(hidden_dim)

        self.out_fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn_out1 = torch.nn.BatchNorm1d(hidden_dim)
        self.out_fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_out2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.out_fc3 = torch.nn.Linear(hidden_dim // 2, latent_dim*2)


    def forward(self, x, node_type, edge_type, edge_index, batch):

        node_attr = self.node_emb(node_type)
        edge_attr = self.edge_emb(edge_type)

        d_emb = self.act(self.bn_d1(self.d_fc1(x)))   # Embedings for edge lengths `x`
        d_emb = self.bn_d2(self.d_fc2(d_emb))
        edge_attr = d_emb * edge_attr

        h = node_attr
        h = self.act(self.bn_conv1(self.conv1(h, edge_index, edge_attr)))
        h = self.act(self.bn_conv2(self.conv2(h, edge_index, edge_attr)))
        h = self.bn_conv3(self.conv3(h, edge_index, edge_attr))

        h_global = scatter(h, batch, dim=0, reduce='sum')
        node_feat = torch.cat([h, h_global[batch]], dim=-1)
        node_feat = self.act(self.bn_out1(self.out_fc1(node_feat)))
        node_feat = self.act(self.bn_out2(self.out_fc2(node_feat)))
        out = self.out_fc3(node_feat)

        mu = out[:, :self.latent_dim]
        sigma = out[:, self.latent_dim:]

        return mu, sigma


def build_flow(args, hidden_dim, num_blocks):
    def build_cnf():
        diffeq = ODEgnn(
            hidden_dim=hidden_dim,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol,
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    if args.batch_norm:
        bn_layers = [MovingBatchNorm1d(1, bn_lag=args.bn_lag, sync=args.sync_bn)
                     for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(1, bn_lag=args.bn_lag, sync=args.sync_bn)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)
    return model


class CNFDecoder(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.latent_emb = torch.nn.Linear(args.latent_dim, args.latent_dim)
        self.node_emb = torch.nn.Embedding(100, args.hidden_dim-args.latent_dim)
        self.edge_emb = torch.nn.Embedding(100, args.hidden_dim)
        self.flow = build_flow(
            args,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        )
        self.use_deterministic_encoder = args.use_deterministic_encoder

    def emb(self, node_type, edge_type, latent):
        node_attr = torch.cat([self.latent_emb(latent), self.node_emb(node_type)], dim=1)
        edge_attr = self.edge_emb(edge_type)
        return node_attr, edge_attr

    def get_d(self, data, z, latent):
        node_attr, edge_attr = self.emb(data.node_type, data.edge_type, latent)
        d = self.flow(
            z,
            node_attr = node_attr,
            edge_attr = edge_attr,
            edge_index = data.edge_index,
            reverse = True,
        )
        return d

    def get_z(self, data, d, latent):
        node_attr, edge_attr = self.emb(data.node_type, data.edge_type, latent)
        z = self.flow(
            d,
            node_attr = node_attr,
            edge_attr = edge_attr,
            edge_index = data.edge_index,
            reverse = False
        )
        return z

    def get_log_prob(self, data, d, latent):
        E = d.size(0)
        node_attr, edge_attr = self.emb(data.node_type, data.edge_type, latent)
        z, delta_logpz = self.flow(
            x = d,
            node_attr = node_attr,
            edge_attr = edge_attr,
            edge_index = data.edge_index,
            logpx=torch.zeros(E, 1).to(d)
        )
        log_pz = standard_normal_logprob(z).view(E, -1).sum(1, keepdim=True)
        log_pd = log_pz - delta_logpz
        return log_pd

    def get_loss(self, data, d, latent):
        log_pd = self.get_log_prob(data, d, latent)
        loss = - log_pd.mean()
        return loss

    def sample(self, data, num_samples, latent):
        E = data.edge_index.size(1)
        node_attr, edge_attr = self.emb(data.node_type, data.edge_type, latent)
        if self.use_deterministic_encoder:
            z = torch.zeros(num_samples*E, 1).to(edge_attr)
        else:
            z = torch.randn(num_samples*E, 1).to(edge_attr)

        edge_index_rep = []
        for i in range(num_samples):
            edge_index_rep.append(data.edge_index + data.num_nodes * i)
        edge_index_rep = torch.cat(edge_index_rep, dim=1)

        samples = self.flow(
            z,
            node_attr = node_attr.repeat(num_samples, 1),
            edge_attr = edge_attr.repeat(num_samples, 1),
            edge_index = edge_index_rep,
            reverse = True
        )

        samples = samples.reshape(-1, E).t()    # (E, num_samples)
        return samples, z


class ImplicitVAE(torch.nn.Module):

    def __init__(self, args):
        super(ImplicitVAE, self).__init__()
        self.device = args.device
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.latent_dim = args.latent_dim
        self.kl_weight = args.kl_weight
        self.implicit_weight = args.implicit_weight
        self.prior = GNNPrior(args.hidden_dim, args.latent_dim)
        self.encoder = GNNEncoder(args.hidden_dim, args.latent_dim)
        self.decoder = CNFDecoder(args)
        if args.use_implicit_loss:
            self.align_mol = ParallelAlignMol(num_workers=32)
    
    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    def get_nll(self, data, d, use_implicit=False, eval=False):
        E = d.size(0)

        if eval:
            with torch.no_grad():
                # sigma is logvar
                # q(z|G,D)
                mu_q, sigma_q = self.encoder(data.edge_length, data.node_type, 
                                            data.edge_type, data.edge_index, 
                                            data.batch)
                # p(z|G)
                mu_p, sigma_p = self.prior(data.node_type, data.edge_type, 
                                        data.edge_index, data.batch)
                # KL Distance
                loss_kl = self.compute_vae_kl(mu_q, sigma_q, mu_p, sigma_p)

                # infer latent
                if self.use_deterministic_encoder:
                    latent = mu_q + 0 * torch.exp(0.5 * sigma_q)
                else:
                    latent = self.reparameterize_gaussian(mu_q, sigma_q)

                # Reconstrcution
                ## p(D|G,z)
                log_pd = self.decoder.get_log_prob(data, d, latent)
                loss_rec_d = - log_pd
        else:
            # sigma is logvar
            # q(z|G,D)
            mu_q, sigma_q = self.encoder(data.edge_length, data.node_type, 
                                        data.edge_type, data.edge_index, 
                                        data.batch)
            # p(z|G)
            mu_p, sigma_p = self.prior(data.node_type, data.edge_type, 
                                    data.edge_index, data.batch)
            # KL Distance
            loss_kl = self.compute_vae_kl(mu_q, sigma_q, mu_p, sigma_p)

            # infer latent
            if self.use_deterministic_encoder:
                latent = mu_q + 0 * torch.exp(0.5 * sigma_q)
            else:
                latent = self.reparameterize_gaussian(mu_q, sigma_q)

            # Reconstrcution
            ## p(D|G,z)
            log_pd = self.decoder.get_log_prob(data, d, latent)
            loss_rec_d = - log_pd
        ## p(X|G,z)
        if use_implicit:
            loss_rec_x = self.implicit_loss(data, latent)
        else:
            loss_rec_x = None

        return loss_kl, loss_rec_d, loss_rec_x

    @staticmethod
    def compute_vae_kl(mu_q, logvar_q, mu_prior, logvar_prior):
        mu1 = mu_q
        std1 = torch.exp(0.5*logvar_q)
        mu2 = mu_prior
        std2 = torch.exp(0.5*logvar_prior)
        kl = - 0.5 + torch.log(std2 / (std1 + 1e-8) + 1e-8) + \
            ((torch.pow(std1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(std2, 2)))

        return kl

    def get_loss(self, data, d, use_implicit=False):
        loss_kl, loss_rec_d, loss_rec_x = self.get_nll(data, d, use_implicit=use_implicit)
        bs = data.batch[-1]+1
        loss_kl = self.kl_weight*loss_kl.sum() / bs
        loss_rec_d = loss_rec_d.sum() / bs
        if use_implicit:
            loss_rec_x = self.implicit_weight*loss_rec_x.sum() / bs
        else:
            loss_rec_x = 0

        loss = loss_kl + loss_rec_d + loss_rec_x
        
        return loss, loss_kl, loss_rec_d, loss_rec_x

    def sample(self, data, num_samples):
        # p(z|G)
        mu_p, sigma_p = self.prior(data.node_type, data.edge_type, 
                                data.edge_index, data.batch)
        if self.use_deterministic_encoder:
            latent = mu_p + 0 * torch.exp(0.5 * sigma_p)
        else:
            latent = self.reparameterize_gaussian(mu_p, sigma_p)
        
        samples = self.decoder.sample(data, num_samples, latent)

        return samples

    def implicit_loss(self, data, latent=None):
        pos, _, _ = self.implicit_layer(data, latent=latent)
        pos = pos[0]

        mols_truth = []
        mols_gen = []
        for i in range(data.num_graphs):
            mols_truth.append(copy.deepcopy(data.rdmol[i]))
            conf = pos[data.batch==i]
            mols_gen.append(set_rdmol_positions(data.rdmol[i], conf.clone().detach()))
        probe = self.align_mol(mols_truth, mols_gen).to(data.pos)   # (\sum_G num_atoms_of_G, 3)
        loss = scatter_mean(((probe - pos) ** 2).sum(-1), data.batch, dim=0, dim_size=data.num_graphs) ** (0.5)
        # loss = loss.sum()
        return loss
    
    # def implicit_loss(self, data, latent=None):
    #     loss = 0

    #     pos, _, _ = self.implicit_layer(data, latent=latent)
    #     pos = pos[0]
    #     for i in range(data.batch[-1]+1):
    #         mol_truth = copy.deepcopy(data.rdmol[i])
    #         conf = pos[data.batch==i]
    #         mol_gen = set_rdmol_positions(mol_truth, conf.clone().detach())
    #         AlignMol(mol_truth, mol_gen)
    #         probe = torch.Tensor(mol_truth.GetConformer(0).GetPositions()).to(data.pos)
    #         loss += ((probe-conf)**2).sum(-1).mean(0)**(.5)

    #     return loss
            
    
    def implicit_layer(self, data, num_samples=1, embedder=ImplicitEmbed3D(), dg_init_pos=None, latent=None):
        """
        Generate conformations in batch with d->pos.
        """
        # with torch.no_grad():
        if latent is None:
            d, _ = self.sample(data, num_samples)  # (E, num_samples)
        else:
            d, _ = self.decoder.sample(data, num_samples, latent)
        d = d.view(-1)

        edge_indices = []
        for i in range(num_samples):
            edge_indices.append(data.edge_index + i * data.num_nodes)
        edge_indices = torch.cat(edge_indices, dim=1)

        if dg_init_pos is None:
            # dg_init_pos = torch.randn(num_samples*data.num_nodes, 3).to(data.pos)
            dg_init_pos = []
            for i in range(len(data.rdmol)):
                mol = copy.deepcopy(data.rdmol[i])
                DP.Compute2DCoords(mol)
                dg_init_pos.append(torch.Tensor(mol.GetConformer(0).GetPositions()))
            dg_init_pos = torch.cat(dg_init_pos, dim=0).to(data.pos)
            dg_init_pos[:, -1] += torch.randn(dg_init_pos.shape[0]).to(data.pos)
        else:
            dg_init_pos = dg_init_pos.repeat(num_samples, 1).to(data.pos)

        pos, _ = embedder(
            d, 
            edge_indices,
            dg_init_pos,
            data.edge_order.repeat(num_samples),
        )   # (num_samples*N, 3)

        d_new = torch.norm(pos[edge_indices[0]] - pos[edge_indices[1]], dim=1).reshape(num_samples, -1)
        d_new = d_new.view(num_samples, -1)
        pos = pos.view(num_samples, -1, 3)

        return pos, d_new, d
    
