import torch
from tqdm.auto import tqdm

from .ebm import *
from .edgecnf import *
from .distgeom import *


def latent_mcmc(
    model_ebm, model_cnf, batch, num_steps, step_size, noise_scale, 
    embedder_init = Embed3D(mu=0.25, step_size=2.0, num_steps=1000),
    embedder_mcmc = Embed3D(mu=0, step_size=0.05, num_steps=200),
    logger=None
):
    # embedder_init = Embed3D(mu=0.25, step_size=2.0, num_steps=1000)
    # embedder_mcmc = Embed3D(mu=0, step_size=0.05, num_steps=200)

    data = batch.clone()
    # Comput. graph: z_new -> d_new -> (pos_new, log_q)
    z_new, _, pos_new, log_q = generate_conf_mcmc_step(model_cnf, data, embedder=embedder_init)
    traj = []


    for step in tqdm(range(num_steps), desc='MCMC-Latent'):
        data.pos = pos_new.detach()
        ener = model_ebm(data)
        log_p = -1 * ener.sum() + log_q.sum() / 2

        traj.append({
            'pos': data.pos.clone().cpu(),
            'energy': ener.clone().detach().cpu(),
            'log_p': log_p.clone().detach().cpu(),
        })

        z_grad = torch.autograd.grad(outputs=log_p, inputs=z_new)[0]

        if callable(step_size):
            s = step_size(step, z_new, ener, log_q, log_p)
        else:
            s = step_size
        z_new = z_new + s/2 * z_grad + (s**.5) * noise_scale * torch.randn_like(z_new)

        if logger is not None:
            logger.info('[Step %d] log_P %.4f | Ener %.4f (%.2f ~ %.2f) | log_Q %.4f | Grad_mag %.4f | Step %s' % (
                step,
                log_p.item(),
                ener.mean().item(), ener.min().item(), ener.max().item(),
                log_q.mean().item(),
                z_grad.abs().max(),
                repr(s)
            ))


        z_new, _, pos_new, log_q = generate_conf_mcmc_step(model_cnf, data, z=z_new.detach(), embedder=embedder_mcmc, dg_init_pos=pos_new.detach())
    
    return pos_new, z_new, traj


def xyz_mcmc(model_ebm, model_cnf, data, num_steps, step_size, noise_scale, logger=None):
    embedder_init = Embed3D(mu=0.25, step_size=5.0, num_steps=1000)
    embedder_mcmc = Embed3D(mu=0, step_size=0.05, num_steps=200)

    data = data.clone()
    model_cnf.eval()
    with torch.no_grad():
        d_init, _ = model_cnf.sample(data, 1)
    pos, _ = embedder_init(d_init, data.edge_index, torch.randn(data.num_nodes, 3).to(d_init), data.edge_order)
    traj = []

    for step in tqdm(range(num_steps), desc='MCMC-XYZ'):
        traj.append(pos.clone().cpu())
        data.pos = pos.requires_grad_(True)
        d = get_d_from_pos(pos, data.edge_index).view(-1, 1)
        log_q = model_cnf.get_log_prob(data, d)
        ener = model_ebm(data)
        log_p = -1 * ener.sum() + log_q.sum() * 2
        pos_grad = torch.autograd.grad(outputs=log_p, inputs=pos)[0]
        
        if logger is not None:
            logger.info('EnerMean %.6f | EnerMed %.6f | EnerMax %.6f | GradMag %.6f' % (
                ener.mean().item(), ener.median().item(), ener.max().item(), pos_grad.max().item()
            ))

        # grad_mag = torch.norm(pos_grad.clone().detach(), dim=1).max().item()
        # if grad_mag > 5.0:
        #     pos_grad /= grad_mag * 5.0

        pos_grad = torch.clamp(pos_grad, -5.0, 5.0)

        if callable(step_size):
            s = step_size(step, None, ener, log_q, log_p)
        else:
            s = step_size

        pos = pos + (s/2) * pos_grad + (s**.5) * noise_scale * torch.randn_like(pos)
        pos = pos.detach()

    return pos, traj
