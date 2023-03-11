import torch, utils

class MedBLOBp(object):
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
        one step forward  for mirrored BLOB with primal kernel

        Args:
            lr: learning rate
            task_funcs: a function warppered by partial from functools, takes a name as input and return the corresponding function object
        """
        if __debug__ and (not lr or not task_funcs):
            raise RuntimeError('the params should not be None')
        
        # get the functions 
        grad_fn = task_funcs('grad_logp_dual')
        # calculate the value of kernel and gradient of kernel, primal space
        support_primal = self.mirror_map.nabla_psi_star(self.support_dual)
        kernel, nabla_kernel, _ = utils.kernel.kernel_func(support_primal, self.knType, self.bwType, self.bwVal, bw_only = False)
        # calculate the gradient of potential, dual space
        grads = grad_fn(self.support_dual)
        # calculate the vector field
        _nabla2_psi_inv = self.mirror_map.nabla2_psi_inv(support_primal)
        repulsive = (torch.matmul(nabla_kernel, _nabla2_psi_inv) * self.mass_dual[None, :, None]).sum(1) / (kernel * self.mass_dual[None, :]).sum(1, keepdim = True) + (torch.matmul(nabla_kernel, _nabla2_psi_inv) * self.mass_dual[None, :, None] / (kernel * self.mass_dual[None, :]).sum(1)[None,:, None]).sum(1) + self.mirror_map.nabla_logdet_nabla2_psi_star(self.support_dual, support_primal)
        vector_field = grads - repulsive
        self.support_dual += lr * vector_field

    @torch.no_grad()
    def get_state(self):
        return self.mirror_map.nabla_psi_star(self.support_dual), self.mass_dual.clone(), self.support_dual, self.mass_dual