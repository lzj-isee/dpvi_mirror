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
            grads_dual = task.grad_logp_dual(
                particles_dual = particles_dual, 
                features = train_features, 
                labels = train_labels
            )
            grads_primal = task.grad_logp_primal(
                particles_primal = particles_primal, 
                features = train_features, 
                labels = train_labels
            )
            # cross_diff = particles_dual[:, None, :] - particles_dual[None, :, :]
            # sq_distance = torch.sum(cross_diff.pow(2), dim = 2)
            # bw_h = sq_distance + torch.diag(torch.diag(sq_distance) + sq_distance.max())
            # bw_h = bw_h.min(dim = 1)[0].mean()
            kernel, nabla_kernel, bw_h = task.kernel_calc(particles_dual)
            # update in dual space
            repulsive = nabla_kernel.sum(1) / kernel.sum(1, keepdim = True) + (nabla_kernel / kernel.sum(1)[None,:, None]).sum(1)
            # metric_tensor_1 = task.mirror_map.nabla2_psi_inv(particles_primal)
            metric_tensor_1 = task.mirror_map.nabla2_psi(particles_primal)
            # direction = grads_dual - repulsive[:, None, :].matmul(metric_tensor_1).squeeze()
            # debug = torch.linalg.inv(kernel)
            particles_dual += opts.lr * (grads_dual - repulsive[:, None, :].matmul(metric_tensor_1).squeeze())
            # particles_dual += opts.lr * (grads_dual  - repulsive)
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