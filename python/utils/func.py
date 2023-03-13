import torch, ot, numpy as np
from typing import Union

def calculate_w2(supportA: torch.Tensor, massA: torch.Tensor, supportB: torch.Tensor, massB: torch.Tensor) -> float:
    r"""
    calculate the 2-wasserstein distance between distributionA and distributionB

    Args:
        supportA (Tensor): a mini-batch tensor of shape (B x D)
        massA (Tensor): a mini-batch tensor of shape (B)
        supportB (Tensor): a mini-batch tensor of shape (B x D)
        massB (Tensor): a mini-batch tensor of shape (B)

    Returns:
        result (float): the 2-wasserstein distance between distributionA and distributionB
    """
    if __debug__ and ((not isinstance(supportA, torch.Tensor)) or (not isinstance(massA, torch.Tensor)) or (not isinstance(supportB, torch.Tensor)) or (not isinstance(massB, torch.Tensor))):
        raise RuntimeError('input should be of type torch.Tensor')
    
    massA = massA / massA.sum()
    massB = massB / massB.sum()
    
    cost_matrix = torch.cdist(supportA, supportB, p = 2).pow(2)
    trans_plan = ot.emd(massA, massB, cost_matrix)
    result = (trans_plan * cost_matrix).sum().sqrt().item()
    return result

def duplicate_kill_particles(prob_list:torch.Tensor, kill_list: torch.Tensor, particles: torch.Tensor, noise_amp:float, mode:str = 'parallel') -> torch.Tensor:
    r"""
    An implementation of duplicate/kill technique, see https://arxiv.org/abs/1905.09863 and https://arxiv.org/abs/1902.01843 for more details

    Args:
        prob_list: a tensor of shape (B), the probability of particles to be duplicated/killed
        kill_list: a tensor of shape (B), if true, the corresponding particle will be kill, otherwise be duplicated
        particles: a tensor of shape (B x D)
        noise_amp: the std of injected noise, see https://arxiv.org/abs/1902.01843 for more details
        mode: 'sequential' or 'parallel', the second one will be faster
    
    Returns:
        a tensor of shape (B x D)
    """
    if __debug__:
        if len(prob_list.shape) != 1:
            raise ValueError('the shape of prob_list shoud be (B)')
        if len(kill_list.shape) != 1:
            raise ValueError('the shape of kill_list shoud be (B)')
        if len(prob_list) != len(kill_list):
            raise ValueError('the length of prob_list and kill_list should be the same')

    # will modify the input particles
    rand_number = torch.rand(particles.shape[0], device = particles.device)
    index_list = torch.linspace(0, particles.shape[0] - 1, particles.shape[0], dtype = torch.int, device = particles.device)
    if mode == 'sequential':
        rand_index = torch.randint(0, particles.shape[0] - 1, (particles.shape[0],), device = particles.device)
        for k in range(particles.shape[0]):
            if kill_list[k]: # kill particle k, duplicate with random noise 
                if rand_number[k] < prob_list[k]:
                    particles[k] = particles[index_list != k][rand_index[k]].clone() + torch.randn(particles.shape[1], device = particles.device) * noise_amp
            else: # duplicate particle k, duplicate with random noise
                if rand_number[k] < prob_list[k]:
                    particles[index_list != k][rand_index[k]] = particles[k].clone() + torch.randn(particles.shape[1], device = particles.device) * noise_amp
        return particles
    elif mode == 'parallel':
        unchange_particles = particles[(rand_number >= prob_list)]
        duplicate_particles = particles[torch.bitwise_and(rand_number < prob_list, torch.logical_not(kill_list))]
        new_particles = torch.cat([unchange_particles, duplicate_particles, duplicate_particles + torch.randn_like(duplicate_particles) * noise_amp], dim = 0)
        if new_particles.shape[0] == particles.shape[0]:
            pass
        elif new_particles.shape[0] < particles.shape[0]: # duplicate randomly
            rand_index = torch.randint(0, new_particles.shape[0], (particles.shape[0] - new_particles.shape[0], ), device = new_particles.device)
            new_particles = torch.cat([new_particles, new_particles[rand_index] + torch.randn_like(new_particles[rand_index]) * noise_amp], dim = 0)
        else: # kill randomly
            rand_index = torch.randperm(new_particles.shape[0], device = new_particles.device)
            new_particles = new_particles[rand_index][:particles.shape[0]].clone()
        assert new_particles.shape[0] == particles.shape[0], 'change the particle number!'
        return new_particles
    else:
        raise NotImplementedError
