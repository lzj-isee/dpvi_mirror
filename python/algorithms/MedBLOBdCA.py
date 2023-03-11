import torch, utils
from .MedBLOBd import MedBLOBd

class MedBLOBdCA(object):
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

    @classmethod
    def first_var_blob(cls, mass, potential, kernel):
        beta = torch.log(torch.sum(kernel * mass[None, :], dim = 1)) + torch.sum((kernel * mass[None, :]) / torch.sum(kernel * mass[None, :], dim = 1)[None, :], dim = 1) + potential
        beta_bar = beta - (beta * mass).sum()
        return beta_bar


    @torch.no_grad()
    def one_step_update(self, lr:float = None, task_funcs = None, **kw):
        r"""
        one step update of mirrored Blob method with dual kernel

        Args:
            lr: learning rate
            task_funcs: a function warppered by partial from functools, taske a name as input and return the corrsponding function object
        """
        if __debug__ and (not lr or not task_funcs):
            raise RuntimeError('the params should not be None')
        # get the functions
        grad_fn = task_funcs('grad_logp_dual')
        potential_fn = task_funcs('potential_dual')
        # calculate the value of kernel and the gradient of kernel, dual space
        kernel, nabla_kernel, _ = utils.kernel.kernel_func(self.support_dual, self.knType, self.bwType, self.bwVal, bw_only = False)
        # calculate the gradient of potential
        grads = grad_fn(self.support_dual)
        vector_field = MedBLOBd.vector_blob(self.mass_dual, grads, kernel, nabla_kernel)
        # calculate the fisher-rao
        potential = potential_fn(self.support_dual)
        first_var = MedBLOBdCA.first_var_blob(self.mass_dual, potential, kernel)
        # update the support and the mass
        self.support_dual += lr * vector_field
        self.mass_dual *= 1 - lr * kw['alpha'] * first_var
        self.mass_dual = self.mass_dual / self.mass_dual.sum() # eliminate numerical error

    @torch.no_grad()
    def get_state(self):
        return self.mirror_map.nabla_psi_star(self.support_dual), self.mass_dual.clone(), self.support_dual, self.mass_dual
        