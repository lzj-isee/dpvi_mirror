import torch

def safe_log(x):
    return torch.log(torch.maximum(torch.ones_like(x) * 1e-32, x))

def safe_reciprocal(x):
    return torch.reciprocal(torch.maximum(x, torch.ones_like(x) * 1e-32))

def psi_star(inputs: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """
    convex conjuate of entropic fucntion

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), dual space.
        keepdim (bool): whether the output tensor has dim retained or not, same as torch.

    Returns:
        result (Tensor): tensor of shape (B) or (B x 1)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    result = torch.log(1 + inputs.exp().sum(1, keepdim = keepdim))
    return result

def nabla_psi(inputs: torch.Tensor, safe_mode:bool = False) -> torch.Tensor: 
    """
    mirror map from primal to dual

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space
        safe_mode (bool): avoid inf while calculating torch.log

    Returns:
        result (Tensor): tensor of shape (B x D)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    if not safe_mode:
        result = torch.log(inputs) - torch.log(1 - torch.sum(inputs, dim = 1, keepdim = True))
    else:
        result = safe_log(inputs) - safe_log(1 - torch.sum(inputs, dim = 1, keepdim = True))
    # result = torch.log(inputs / (1 - torch.sum(inputs, dim = 1, keepdim = True)))
    return result

def nabla_psi_star(inputs: torch.Tensor) -> torch.Tensor:
    """
    mirror map from dual to primal

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), dual space

    Returns:
        result (Tensor): tensor of shape (B x D)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    result = torch.exp(inputs) / (1 + torch.sum(torch.exp(inputs), dim = 1, keepdim = True))
    return result

def nabla2_psi_inv(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the inverse of matrix $\nabla^2 \psi(x)$: $\nabla^2 \psi(x)^{-1}$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space
    
    Returns:
        result (Tensor): tensor of shape (B x D x D)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    result = torch.diag_embed(inputs) - inputs[:, :, None] * inputs[:, None, :]
    return result

def nabla2_psi(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the matrix $\nabla^2 \psi(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space
    
    Returns:
        result (Tensor): tensor of shape (B x D x D)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    result = torch.diag_embed(torch.reciprocal(inputs)) + torch.reciprocal((1 - inputs.sum(dim = 1))[:, None, None])
    return result

def nabla2_psi_sqrt(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the square root of matrix $\nabla^2 \psi(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space
    
    Returns:
        result (Tensor): tensor of shape (B x D x D)
    """
    # _nabla2_psi = nabla2_psi(inputs).cpu().numpy()
    # u, s, vh = np.linalg.svd(_nabla2_psi)
    # result = torch.from_numpy(np.matmul(u * np.sqrt(s)[:, None, :], vh)).to(inputs.device)
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    _nabla2_psi = nabla2_psi(inputs)
    u, s, vh = torch.linalg.svd(_nabla2_psi)
    result = u @ torch.diag_embed(torch.sqrt(s)) @ vh
    return result

def nabla2_psi_inv_sqrt(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the square root of the inverse of matrix $\nabla^2 \psi(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space
    
    Returns:
        result (Tensor): tensor of shape (B x D x D)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    _nabla2_psi_inv = nabla2_psi_inv(inputs)
    u, s, vh = torch.linalg.svd(_nabla2_psi_inv)
    result = u @ torch.diag_embed(torch.sqrt(s)) @ vh
    return result

def logdet_nabla2_psi_star(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the log-determinant of matrix $\nabla^2 \psi^*(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), dual space

    Returns:
        result (Tensor): tensor of shape (B)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    dimension = inputs.shape[-1]
    results = inputs.sum(dim = 1) - (dimension + 1) * psi_star(inputs)
    return results

def nabla_logdet_nabla2_psi_star(inputs: torch.Tensor, inputs_primal: torch.Tensor = None) -> torch.Tensor: 
    r"""
    calculate the gradient of log-determinant of matrix $\nabla^2 \psi^*(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), dual space

    Returns:
        result (Tensor): tensor of shape (B x D)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    if __debug__ and (inputs_primal is not None and not isinstance(inputs_primal, torch.Tensor)):
        raise ValueError('the type of inputs_primal should be torch.Tensor')
    if __debug__ and (inputs_primal is not None and len(inputs_primal.shape) != 2):
        raise ValueError('the shape of inputs_primal should be (B x D)')
    
    coef = inputs.shape[-1] + 1
    if inputs_primal is None:
        inputs_primal = nabla_psi_star(inputs)
    result = 1 - coef * inputs_primal
    return result

def logdet_nabla2_psi(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the log-determinant of matrix $\nabla^2 \psi(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space

    Returns:
        result (Tensor): tensor of shape (B)
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    results = inputs.log().sum(dim = 1) - (1 - inputs.sum(dim = 1)).log()
    return results

def nabla_logdet_nabla2_psi(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    calculate the gradient of log-determinant of matrix $\nabla^2 \psi(x)$

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space

    Returns:
        result (Tensor): tensor of shape (B x D)
    """
    # results = - 1. / inputs + 1. / (1 - inputs.sum(dim = 1, keepdim = True))
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    results = torch.reciprocal(1 - inputs.sum(dim = 1, keepdim = True)) - torch.reciprocal(inputs)
    return results

def div_nabla2_psi_inv_diag(inputs: torch.Tensor) -> torch.Tensor:
    r"""
    perform divergence to each row of nabla2_psi_inv(particle_primal), and apply diag to the last dimension

    Args:
        inputs (Tensor): mini-batch tensor of shape (B x D), primal space

    Returns:
        result (Tensor): tensor of shape (B x D X D), batch of diagnal matrix
    """
    if __debug__ and not isinstance(inputs, torch.Tensor):
        raise ValueError('the type of inputs should be torch.Tensor')
    if __debug__ and len(inputs.shape) != 2:
        raise ValueError('the shape of inputs should be (B x D)')
    
    # primal inputs: [particle_num, dimension], perform divergence to each row of nabla2_psi_inv(particle_primal)
    results =  - inputs[:,:,None] - torch.diag_embed(inputs) + torch.eye(inputs.shape[-1], device = inputs.device)
    return results

