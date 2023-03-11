import torch

class rmsprop(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.optim_rho = opts.optim_rho
        self.lr = opts.lr
        self.rms = 0

    def apply_grads(self, particles:torch.Tensor, grads:torch.Tensor) -> torch.Tensor:
        r"""
        one step of optimizer rmsprop

        Args:
            particles (Tensor): the support, a mini-batch tensor of shape (B x D)
            grads (Tensor): the gradient of particles, a mini-batch tensor of shape (B x D)

        Returns:
            the updated particles if shape (B x D)
        """
        if __debug__ and ((not isinstance(particles, torch.Tensor)) or (not isinstance(grads, torch.Tensor))):
            raise RuntimeError('the inputs should be type of torch.Tensor')
        if __debug__ and (particles.shape != grads.shape):
            raise RuntimeError('the shape of particles and grads should be the same')

        self.rms = self.optim_rho * self.rms + (1 - self.optim_rho) * grads.pow(2)
        return particles - self.lr * grads / (self.rms.sqrt() + 1e-7)

class sgd(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.lr = opts.lr

    def apply_grads(self, particles, grads):
        r"""
        one step of optimizer sgd

        Args:
            particles (Tensor): the support, a mini-batch tensor of shape (B x D)
            grads (Tensor): the gradient of particles, a mini-batch tensor of shape (B x D)

        Returns:
            the updated particles if shape (B x D)
        """
        if __debug__ and ((not isinstance(particles, torch.Tensor)) or (not isinstance(grads, torch.Tensor))):
            raise RuntimeError('the inputs should be type of torch.Tensor')
        if __debug__ and (particles.shape != grads.shape):
            raise RuntimeError('the shape of particles and grads should be the same')
        
        return particles - self.lr * grads