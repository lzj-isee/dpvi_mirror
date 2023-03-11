import torch, importlib
from tqdm import tqdm
import numpy as np
import basic

@torch.no_grad()
def run(opts):
    task = importlib.import_module('tasks.{:}'.format(opts.task)).__getattribute__('functions')(opts)
    optim = basic.__getattribute__('{:}'.format(opts.optimizer))(opts)
    print('algorithm and setting: \n', task.save_name)
    # init particles 
    particles_primal, particles_dual = task.init_particles(particle_num = opts.particle_num, particle_dim = task.particle_dim)
    mass = torch.ones(opts.particle_num, device = task.device) / opts.particle_num
    curr_iter_count = 0
    #-------------------------------------------------- iter ---------------------------------------------------------------
    for epoch in tqdm(range(opts.epochs)):
        for iter, (train_features, train_labels) in enumerate(task.train_loader):
            curr_iter_count += 1
            nabla2_psi_inv_grad_primal = task.nabla2_psi_inv_grad_logp_primal(
                particles_primal = particles_primal, 
                features = train_features, 
                labels = train_labels
            )
            particles_primal_ = particles_primal.detach().clone()
            mu, v, v_theta, v_theta_grad = eigen_quantities(particles_primal_, particles_primal, kernel_calc = task.kernel_calc, n_eigen_threshold = opts.n_eigen_threshold)
            nabla2_psi_ = task.mirror_map.nabla2_psi(particles_primal_)
            nabla2_psi_inv_ = task.mirror_map.nabla2_psi_inv(particles_primal)
            div_nabla2_psi_inv_diag_ = task.mirror_map.div_nabla2_psi_inv_diag(particles_primal)
            mu_sqrt, eigen_num = torch.sqrt(mu), len(mu)
            # i_reduced = torch.einsum('i,ki,mi->km', mu_sqrt, v, v)
            i_reduced = (mu_sqrt[None, :, None] * (v[:, :, None] * v.t()[None, :, :])).sum(1)
            # weighted_grad = torch.einsum('km,j,lj,mj,mab,lb->ka', i_reduced, mu_sqrt, v_theta, v, nabla2_psi_, nabla2_psi_inv_grad_primal) / opts.particle_num**2
            temp = (mu_sqrt[None, :, None] * (v_theta[:, :, None] * v.t()[None, :, :])).sum(1)
            weighted_grad = (((i_reduced[:, None, :] * temp[None, :, :])[..., None, None] * nabla2_psi_[None, None, :, :, :]).sum(2) * nabla2_psi_inv_grad_primal[None, :, None, :]).sum(dim = [1,3])
            # repul_term1 = torch.einsum('km,j,ljd,mj,mab,lbd->ka', i_reduced, mu_sqrt, v_theta_grad, v, nabla2_psi_, nabla2_psi_inv_) / opts.particle_num**2
            temp = (mu_sqrt[None, :, None, None] * (v_theta_grad[:, :, None, :] * v.t()[None, :, :, None])).sum(1)
            repul_term1 = (((i_reduced[:, None, :, None] * temp[None, :, :, :])[:, :, :, None, None, :] * nabla2_psi_[None, None, :, :, :, None]).sum(2) * nabla2_psi_inv_[None, :, None, :, :]).sum(dim = [1,3,4])
            # repul_term2 = torch.einsum('km,j,lj,mj,mab,lbd->ka', i_reduced, mu_sqrt, v_theta, v, nabla2_psi_, div_nabla2_psi_inv_diag_) / opts.particle_num**2
            temp = (mu_sqrt[None, :, None] * (v_theta[:, :, None] * v.t()[None, :, :])).sum(1)
            repul_term2 = (((i_reduced[:, None, :] * temp[None, :, :])[..., None, None] * nabla2_psi_[None, None, :, :, :]).sum(2) * div_nabla2_psi_inv_diag_.sum(-1)[None, :, None, :]).sum(dim = [1,3])
            # update in dual space
            direction = (weighted_grad + repul_term1 + repul_term2) / opts.particle_num**2
            particles_dual = optim.apply_grads(particles_dual, -direction)
            # dual to primal
            particles_primal = task.mirror_map.nabla_psi_star(particles_dual)
            # check 
            basic.check(task.save_folder, particles_dual, mass, curr_iter_count)
            # evaluation
            if (curr_iter_count - 1) % opts.eval_interval == 0: 
                task.evaluation(particles_primal, particles_dual, mass, 
                writer = task.writer, global_step = epoch * len(task.train_loader) + iter)
    task.save_final_results(task.writer, task.save_folder, particles_primal, particles_dual, mass)
    task.writer.close()

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


