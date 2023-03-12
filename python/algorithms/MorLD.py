import torch, math

class MorLD(object):
    def __init__(self, opts, init_support_primal:torch.Tensor, init_mass_primal:torch.Tensor, init_support_dual:torch.Tensor, init_mass_dual:torch.Tensor, mirror_map) -> None:

        if __debug__ and ((not isinstance(init_support_dual, torch.Tensor)) or (not isinstance(init_mass_dual, torch.Tensor)) or (len(init_support_dual) != len(init_mass_dual))):
            raise ValueError('the type of support and mass should be torch.Tensor, and the length of support should equal the length of mass')

        if __debug__ and not mirror_map:
            raise RuntimeError('the params should not be None')

        self.opts = opts
        self.mirror_map = mirror_map
        self.knType, self.bwType, self.bwVal = opts.knType, opts.bwType, opts.bwVal

        self.support_dual = init_support_dual
        self.mass_dual = init_mass_dual

    @torch.no_grad()
    def one_step_update(self, lr:float = None, task_funcs = None, **kw):
        r"""
        one step forward for mirror-langevin dynamics, use Riemannian information of target distribution in primal space

        Args:
            lr: learning rate
            task_funcs: a function warppered by partial from functools, takes a name as input and return the corresponding function object
        """
        if __debug__ and (not lr or not task_funcs):
            raise RuntimeError('the params should not be None')
        # calculate the primal support
        support_primal = self.mirror_map.nabla_psi_star(self.support_dual)
        # get the functions in dual space
        grad_fn = task_funcs('grad_logp_primal')
        # calculate precondition matrix
        _nabla2_psi_sqrt = self.mirror_map.nabla2_psi_sqrt(support_primal)
        self.support_dual += lr * grad_fn(support_primal) + torch.matmul(_nabla2_psi_sqrt, torch.randn_like(self.support_dual)[:, :, None]).squeeze() * math.sqrt(2 * lr)

    @torch.no_grad()
    def get_state(self):
        return self.mirror_map.nabla_psi_star(self.support_dual), self.mass_dual.clone(), self.support_dual, self.mass_dual

