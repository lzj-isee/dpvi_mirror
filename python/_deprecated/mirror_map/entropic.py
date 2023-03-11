import torch, os
import numpy as np
from torch._C import RRefType

@torch.no_grad()
def safe_log(x):
    return torch.log(torch.maximum(torch.ones_like(x) * 1e-32, x))

@torch.no_grad()
def safe_reciprocal(x):
    return 1. / torch.maximum(x, torch.ones_like(x) * 1e-32)

@torch.no_grad()
def psi_star(inputs, keepdim = False):
    # dual inputs
    result = torch.log(1 + inputs.exp().sum(1, keepdim = keepdim))
    return result

@torch.no_grad()
def nabla_psi(inputs): 
    # priaml to dual, primal inputs: [particle_num, dimension]
    result = torch.log(inputs / (1 - torch.sum(inputs, dim = 1, keepdim = True)))
    return result

@torch.no_grad()
def nabla_psi_star(inputs):
    # dual to primal, dual inputs: [particle_num, dimension]
    result = torch.exp(inputs) / (1 + torch.sum(torch.exp(inputs), dim = 1, keepdim = True))
    return result
    # inputs_ext = torch.cat([inputs, torch.zeros((inputs.shape[0], 1), device = inputs.device)], dim = 1)
    # return torch.softmax(inputs_ext, dim = 1)[:, :-1]

@torch.no_grad()
def nabla2_psi_inv(inputs):
    # priaml inputs: [particle_num, dimension], particle in primal
    result = torch.diag_embed(inputs) - inputs[:, :, None] * inputs[:, None, :]
    return result # [particle_num, dimension, dimension]

@torch.no_grad()
def nabla2_psi(inputs):
    # primal inputs: [particle_num, dimension]
    result = torch.diag_embed(safe_reciprocal(inputs)) + safe_reciprocal(1 - inputs.sum(dim = 1))[:, None, None]
    return result

@torch.no_grad()
def nabla2_psi_sqrt(inputs):
    # primal inputs: [particle_num, dimension]
    _nabla2_psi = nabla2_psi(inputs).cpu().numpy()
    u, s, vh = np.linalg.svd(_nabla2_psi)
    result = torch.from_numpy(np.matmul(u * np.sqrt(s)[:, None, :], vh)).to(inputs.device)
    return result

@torch.no_grad()
def nabla2_psi_inv_sqrt(inputs):
    # primal inputs: [particle_num, dimension]
    _nabla2_psi_inv = nabla2_psi_inv(inputs).cpu().numpy()
    u, s, vh = np.linalg.svd(_nabla2_psi_inv)
    result = torch.from_numpy(np.matmul(u * np.sqrt(s)[:, None, :], vh)).to(inputs.device)
    return result

@torch.no_grad()
def logdet_nabla2_psi_star(inputs):
    # dual inputs: [particle_num, dimension]
    results = inputs.sum(dim = 1) - (inputs.shape[-1] + 1) * psi_star(inputs)
    return results

@torch.no_grad()
def nabla_logdet_nabla2_psi_star(inputs, inputs_primal = None): 
    # dual inputs: [particle_num, dimension]
    coef = inputs.shape[-1] + 1
    if inputs_primal is None:
        inputs_primal = nabla_psi_star(inputs)
    result = 1 - coef * inputs_primal
    return result

@torch.no_grad()
def logdet_nabla2_psi(inputs):
    # primal space
    results = - (inputs.log().sum(dim = 1) + (1 - inputs.sum(dim = 1)).log())
    return results

@torch.no_grad()
def nabla_logdet_nabla2_psi(inputs):
    # primal inputs: [particle_num, dimension]
    results = - 1. / inputs + 1. / (1 - inputs.sum(dim = 1, keepdim = True))
    return results

@torch.no_grad()
def div_nabla2_psi_inv_diag(inputs):
    # primal inputs: [particle_num, dimension], perform divergence to each row of nabla2_psi_inv(particle_primal)
    results =  - inputs[:,:,None] - torch.diag_embed(inputs) + torch.eye(inputs.shape[-1], device = inputs.device)
    return results

