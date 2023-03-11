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
            grads_primal = task.grad_logp_primal(
                particles_primal = particles_primal, 
                features = train_features, 
                labels = train_labels
            )
            kernel, nabla_kernel, bw_h = task.kernel_calc(particles_primal)
            # update in dual space
            particles_dual += opts.lr * (grads_primal - nabla_kernel.sum(1) / kernel.sum(1, keepdim = True))
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