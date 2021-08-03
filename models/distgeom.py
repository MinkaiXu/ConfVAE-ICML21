import torch


def embed_3D(d_target, edge_index, init_pos, edge_order=None, alpha=0.5, mu=0, step_size=None, num_steps=None, logger=None):
    assert torch.is_grad_enabled, '`embed_3D` requires gradients'
    step_size = 8.0 if step_size is None else step_size
    num_steps = 200 if num_steps is None else num_steps

    d_target = d_target.view(-1)
    pos = init_pos.clone().requires_grad_(True)
    # optimizer = torch.optim.Adam([pos], lr=step_size)
    optimizer = torch.optim.Adam([pos], lr=step_size)

    if edge_order is not None:
        coef = alpha ** (edge_order.view(-1).float() - 1)
    else:
        coef = 1.0
    
    if mu > 0:
        noise = torch.randn_like(coef) * coef * mu + coef
        coef = torch.clamp_min(coef + noise, min=0)

    for i in range(num_steps):
        optimizer.zero_grad()
        d_new = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        loss = (coef * ((d_target - d_new) ** 2)).sum()
        # loss.backward()
        pos.grad = torch.autograd.grad(loss, pos, create_graph=True)[0]
        optimizer.step()
        # pos = pos - step_size*pos_grad
    if logger is not None:
        logger.info('Embed 3D: AvgLoss %.6f' % (loss.item() / d_target.size(0)))

    return pos, loss.detach() / d_target.size(0)


class Embed3D(object):

    def __init__(self, alpha=0.5, mu=0, step_size=8.0, num_steps=1000, logger=None):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.step_size = step_size
        self.num_steps = num_steps
        self.logger = logger

    def __call__(self, d_target, edge_index, init_pos, edge_order=None):
        return embed_3D(
            d_target, edge_index, init_pos, edge_order,
            alpha=self.alpha,
            mu=self.mu,
            step_size=self.step_size,
            num_steps=self.num_steps,
            logger=self.logger
        )


class DistanceGeometry(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, d, pos, edge_index, eps=5e-3):
        """
        Args:
            d:    Input distances, (E, ) or (E, 1).
            pos:  Precomputed positions, (N, 3).
        """
        ctx.save_for_backward(d, pos, edge_index)
        d = d.flatten()
        d_ref = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        # print((d-d_ref).abs().max())
        assert torch.allclose(d, d_ref, atol=eps, rtol=eps)
        return pos

    @staticmethod
    def backward(ctx, grad_pos):
        """
        Compute gradiants.
        Args:
            grad_pos:   Gradients w.r.t. the output pos, (N, 3).
        Returns:
            grad_d, grad_pos, None, None
        """
        dist, pos, edge_index = ctx.saved_tensors 
        dist_size = dist.size()
        dist = dist.flatten()
        D = grad_pos[edge_index[0]] - grad_pos[edge_index[1]]   # (E, 3)
        L = pos[edge_index[0]] - pos[edge_index[1]]             # (E, 3)
        grad_d = (D * L).sum(dim=1) / dist
        return grad_d.view(dist_size), grad_pos, None, None


def get_d_from_pos(pos, edge_index):
    return torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
