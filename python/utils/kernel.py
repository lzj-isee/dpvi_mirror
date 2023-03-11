import torch, numpy as np

def gaussian_stein_kernel(x:torch.Tensor, y:torch.Tensor, score_x:torch.Tensor, score_y:torch.Tensor, bw:float) -> torch.Tensor:
    r"""
    the kernel function defined in equ 3 of https://proceedings.mlr.press/v139/korba21a.html

    a necessary kernel to implement KSD descent method, see https://proceedings.mlr.press/v139/korba21a.html for more details

    Args:
        x: a mini-batch tensor of shape (B x D), the requires_grad should always be false, thus an extra .detach() should be applied
        y: a mini-batch tensor of shape (B x D), the requires_grad of y should be true when used to implement KSD descent method
        score_x: score function of x, a tensor of shape (B x D), the requires_grad should always be false, thus an extra .detach() should be applied
        score_y: score function of y, a tensor of shape (B x D), the requires_grad should be true when used to implement KSD descent method
        bw: the bandwidth parameter
    Returns:
        ksd_matrix: the kernel value, a tensor of shape (B x B)
    """ 
    if __debug__:
        inputs = [x, y, score_x, score_y]
        if any([not isinstance(x, torch.Tensor) for x in inputs]):
            raise ValueError('the type of input should be torch.Tensor')
        if any([len(x.shape) != 2 for x in inputs]):
            raise ValueError('the shape of input should be (B x D)')

    _, p = x.shape
    sq_dist = x.pow(2).sum(1).view(-1, 1) + y.pow(2).sum(1).view(1, -1) - 2 * torch.matmul(x, y.t())
    kernel = ( - sq_dist / bw).exp()
    part1 = torch.matmul(score_x, score_y.t()) * kernel
    part2 = 2 * ((score_x * x).sum(1, keepdim = True) - torch.matmul(score_x, y.t())) * kernel / bw
    part3 = -2 * (torch.matmul(x, score_y.t()) - (y * score_y).sum(1)) * kernel / bw
    part4 = (2 * p / bw - 4 * sq_dist / bw**2) * kernel
    ksd_matrix = part1 + part2 + part3 + part4
    return ksd_matrix

def kernel_func(particles:torch.Tensor, knType:str = 'rbf', bwType:str = 'med', bwVal:float = 1, bw_only:bool = None) -> tuple[torch.Tensor, torch.Tensor, float]:
    r"""
    calculate the kernel function for a batch of tensor

    Args:
        particles (Tensor): a mini-batch tensor of shape (B x D)
        knType (str): choose rbf kernel or imq kernel
        bwType (str): different type of bandwidth 
        bwVal (float): bandwidth param of kernel function
        bw_only (bool): if true, only return bandwidth

    Returns:
        kernel: a matrix tensor of shape (B x B), the value of kernel function
        nabla_kernel: the gradient of kernel function, a tensor of shape (B x B x D)
        bw_h: the value of bandwidth 
    """
    if __debug__:
        if (not isinstance(particles, torch.Tensor) or len(particles.shape) != 2):
            raise ValueError('the type of particle_num should be torch.Tensor and the shape should be (B x D)') 

    particle_num = particles.shape[0]
    cross_diff = particles[:, None, :] - particles[None, :, :]
    sq_distance = torch.sum(cross_diff.pow(2), dim = 2)
    if bwType == 'med': # SVGD
        bw_h = torch.median(sq_distance + 1e-5) / np.log(particle_num)
    elif bwType == 'nei': # GFSD, Blob
        bw_h = sq_distance + torch.diag(torch.diag(sq_distance) + sq_distance.max())
        bw_h = bw_h.min(dim = 1)[0].mean() if particle_num > 1 else 1
    elif bwType == 'fix': # fixed bandwidth
        bw_h = bwVal
    elif bwType == 'heu': # this bandwith is from Mirrored SVGD
        n_elems = sq_distance.shape[0] * sq_distance.shape[1]
        topk_values = torch.topk(sq_distance.view(-1), k = n_elems // 2, sorted = False).values
        bw_h = torch.min(topk_values)
        bw_h = torch.where(bw_h == 0, torch.ones_like(bw_h), bw_h)
    else: 
        raise NotImplementedError
    if bw_only: return None, None, bw_h
    if knType == 'imq': 
        kernel = (1 + sq_distance / bw_h).pow(-0.5)
        nabla_kernel = -kernel.pow(3)[:, :, None] * cross_diff / bw_h
    elif knType == 'rbf':
        kernel = (-sq_distance / bw_h).exp()
        nabla_kernel = -2 * cross_diff * kernel[:, :, None] / bw_h
    else:
        raise NotImplementedError
    return kernel, nabla_kernel, bw_h