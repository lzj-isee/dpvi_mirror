import torch, utils, numpy as np
from optim import rmsprop, sgd

class SVMirrorD(object):
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

        self.optim = rmsprop(opts) if opts.optim == 'rmsprop' else sgd(opts)
    
    @torch.no_grad()
    def one_step_update(self, lr:float = None, task_funcs = None, **kw):
        r"""
        one step forward for mirrored SVGD with primal kernel, https://arxiv.org/abs/2106.12506

        Args:
            lr: learning rate
            task_funcs: a function warppered by partial from functools, takes a name as input and return the corresponding function object
        """
        if __debug__ and (not lr or not task_funcs):
            raise RuntimeError('the params should not be None')
        
        support_primal = self.mirror_map.nabla_psi_star(self.support_dual)
        # calculate the vector field, NOTE: 2023-3-7, lzj: 以下这些代码很难直接看懂，需要看原论文推一遍
        _nabla2_psi_inv_grad_primal = task_funcs('nabla2_psi_inv_grad_logp_primal')(support_primal)
        mu, v, v_theta, v_theta_grad = eigen_quantities(support_primal.clone(), support_primal, kernel_calc = utils.kernel.kernel_func, n_eigen_threshold = self.opts.n_eigen_threshold)
        _nabla2_psi = self.mirror_map.nabla2_psi(support_primal)
        _nabla2_psi_inv = self.mirror_map.nabla2_psi_inv(support_primal)
        _div_nabla2_psi_inv_diag = self.mirror_map.div_nabla2_psi_inv_diag(support_primal)
        
        mu_sqrt, eigen_num = torch.sqrt(mu), len(mu)
        # i_reduced = torch.einsum('i,ki,mi->km', mu_sqrt, v, v)
        i_reduced = (mu_sqrt[None, :, None] * (v[:, :, None] * v.t()[None, :, :])).sum(1)
        # weighted_grad = torch.einsum('km,j,lj,mj,mab,lb->ka', i_reduced, mu_sqrt, v_theta, v, nabla2_psi_, nabla2_psi_inv_grad_primal) / opts.particle_num**2
        temp = (mu_sqrt[None, :, None] * (v_theta[:, :, None] * v.t()[None, :, :])).sum(1)
        weighted_grad = (((i_reduced[:, None, :] * temp[None, :, :])[..., None, None] * _nabla2_psi[None, None, :, :, :]).sum(2) * _nabla2_psi_inv_grad_primal[None, :, None, :]).sum(dim = [1,3])
        # repul_term1 = torch.einsum('km,j,ljd,mj,mab,lbd->ka', i_reduced, mu_sqrt, v_theta_grad, v, nabla2_psi_, nabla2_psi_inv_) / opts.particle_num**2
        temp = (mu_sqrt[None, :, None, None] * (v_theta_grad[:, :, None, :] * v.t()[None, :, :, None])).sum(1)
        repul_term1 = (((i_reduced[:, None, :, None] * temp[None, :, :, :])[:, :, :, None, None, :] * _nabla2_psi[None, None, :, :, :, None]).sum(2) * _nabla2_psi_inv[None, :, None, :, :]).sum(dim = [1,3,4])
        # repul_term2 = torch.einsum('km,j,lj,mj,mab,lbd->ka', i_reduced, mu_sqrt, v_theta, v, nabla2_psi_, div_nabla2_psi_inv_diag_) / opts.particle_num**2
        temp = (mu_sqrt[None, :, None] * (v_theta[:, :, None] * v.t()[None, :, :])).sum(1)
        repul_term2 = (((i_reduced[:, None, :] * temp[None, :, :])[..., None, None] * _nabla2_psi[None, None, :, :, :]).sum(2) * _div_nabla2_psi_inv_diag.sum(-1)[None, :, None, :]).sum(dim = [1,3])
        
        # update in dual space
        _particle_num = self.support_dual.shape[0]
        vector_field = (weighted_grad + repul_term1 + repul_term2) / _particle_num**2
        self.support_dual = self.optim.apply_grads(self.support_dual, -vector_field)

    @torch.no_grad()
    def get_state(self):
        return self.mirror_map.nabla_psi_star(self.support_dual), self.mass_dual.clone(), self.support_dual, self.mass_dual
    

def eigen_quantities(theta_, theta, kernel_calc, n_eigen_threshold, jitter = 1e-5):
    par_num = theta.shape[0]
    Kxz, grad_Kxz, _ = kernel_calc(theta)
    # Kxz, grad_Kxz, _ = kernel_calc(theta, theta_)
    gram = Kxz.detach() + jitter * torch.eye(par_num, device = theta.device)
    eigval, eigvec = torch.linalg.eigh(gram)
    mu, v, v_theta, v_theta_grad = truncate_and_grad(eigval, eigvec.contiguous(), n_eigen_threshold, Kxz, grad_Kxz)
    return mu, v, v_theta, v_theta_grad

def truncate_and_grad(eigval, eigvec, n_eigen_threshold, Kxz, grad_Kxz):
    par_num = eigvec.shape[0]
    eigen_arr = torch.flip(eigval, dims = [0])
    eigen_arr /= torch.sum(eigen_arr)
    eigen_cum = torch.cumsum(eigen_arr, dim = 0)
    n_eigen = torch.sum(eigen_cum < n_eigen_threshold) + 1
    eigval = eigval[-n_eigen:]
    eigvec = eigvec[:, -n_eigen:]
    mu = eigval / par_num
    v = eigvec * np.sqrt(par_num)
    v_theta, v_theta_grad = nystrom(Kxz, eigval, eigvec, grad_Kxz)
    return mu, v, v_theta, v_theta_grad

def nystrom(Kxz, eigval, eigvec, grad_Kxz):
    par_num = Kxz.shape[-1]
    u = np.sqrt(par_num) * torch.matmul(Kxz, eigvec) / eigval[None, :]
    if grad_Kxz is not None:
        # grad_u = np.sqrt(par_num) * torch.einsum('nml,mj->njl', grad_Kxz, eigvec) / eigval[None, :, None]
        grad_u = np.sqrt(par_num) * (grad_Kxz[:, :, None, :] * eigvec[None, :, :, None]).sum(1) / eigval[None, :, None]
        return u, grad_u
    else:
        return u