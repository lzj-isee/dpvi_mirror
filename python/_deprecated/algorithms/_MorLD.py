import torch, importlib
from tqdm import tqdm
import numpy as np
import basic

@torch.no_grad()
def run(opts):
    task = importlib.import_module('tasks.{:}'.format(opts.task)).__getattribute__('functions')(opts)
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
            # update in dual space
            pre_condition = task.mirror_map.nabla2_psi_sqrt(particles_primal)
            noise = torch.matmul(pre_condition, torch.randn_like(particles_primal)[:, :, None]).squeeze() * np.sqrt(2 * opts.lr)
            particles_dual += opts.lr * grads_primal + noise
            # dual to primal
            particles_primal = task.mirror_map.nabla_psi_star(particles_dual)
            # check 
            basic.check(task.save_folder, particles_primal, mass, curr_iter_count)
            # evaluation
            if (curr_iter_count - 1) % opts.eval_interval == 0: 
                task.evaluation(particles_primal, particles_dual, 
                mass, writer = task.writer, global_step = epoch * len(task.train_loader) + iter)
    task.save_final_results(task.writer, task.save_folder, 
        particles_primal, particles_dual, mass)
    task.writer.close()