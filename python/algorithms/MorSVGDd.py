import torch, utils

class MorSVGDd(object):
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
        one step forward

        Args:
            lr: learning rate
            task_funcs: a function warppered by partial from functools, takes a name as input and return the corresponding function object
        """
        if __debug__ and (not lr or not task_funcs):
            raise RuntimeError('the params should not be None')
        # get the functions in dual space
        grad_fn = task_funcs('grad_logp_primal')
        # calculate the particles in primal space
        support_primal = self.mirror_map.nabla_psi_star(self.support_dual)
        # calcualte the value of kernel and gradient of kernel, use support from primal space (primal kernel)
        kernel, nabla_kernel, _ = utils.kernel.kernel_func(self.support_dual, self.knType, self.bwType, self.bwVal, bw_only = False)
        # calculate the gradient of potential
        grads = grad_fn(support_primal)
        # calculate the gradint part and the repulsice part of SVGD method
        grad_part = torch.matmul(kernel, self.mass_dual[:, None] * grads)
        repulsive_part = (torch.matmul(nabla_kernel, self.mirror_map.nabla2_psi(support_primal)) * self.mass_dual[:, None, None]).sum(dim = 0)
        self.support_dual += lr * (grad_part + repulsive_part)

    @torch.no_grad()
    def get_state(self):
        return self.mirror_map.nabla_psi_star(self.support_dual), self.mass_dual.clone(), self.support_dual, self.mass_dual