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
            kernel, nabla_kernel, bw_h = task.kernel_calc(particles_dual)
            # update in dual space
            repulsive = (nabla_kernel * mass[None, :, None]).sum(1) / (kernel * mass[None, :]).sum(1, keepdim = True) + (nabla_kernel * mass[None, :, None] / (kernel * mass[None, :]).sum(1)[None,:, None]).sum(1)
            particles_dual += opts.lr * (grads_dual - repulsive)
            potential = task.potential_dual(particles_dual, train_features, train_labels) 
            beta = torch.log((mass * kernel).sum(1) + 1e-32) + ((kernel * mass) / torch.matmul(mass, kernel)).sum(1) + potential
            beta_bar = beta - (beta * mass).sum()
            mass = mass * (1 - beta_bar * opts.lr * opts.alpha)
            # dual to primal
            particles_primal = task.mirror_map.nabla_psi_star(particles_dual)
            # check 
            basic.check(task.save_folder, particles_dual, mass, curr_iter_count)
            mass = mass / mass.sum() # eliminate numerical error
            # evaluation
            if (curr_iter_count - 1) % opts.eval_interval == 0: 
                task.evaluation(particles_primal, particles_dual, mass, 
                writer = task.writer, global_step = epoch * len(task.train_loader) + iter)
    task.save_final_results(task.writer, task.save_folder, particles_primal, particles_dual, mass)
    task.writer.close()