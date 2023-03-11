import torch, os, numpy as np, ot, matplotlib.pyplot as plt, utils, easydict
from typing import Optional, Union

class dirichletMulti(object):
    def __init__(self, opts, dataset_info:dict = None) -> None:
        self.opts = opts
        self.device = opts.device
        # set the mirror_map functions
        self.mirror_map = utils.entropic_map
        # generate the target distribution and reference samples in primal space
        self.dirichlet_param = torch.Tensor([1.0, 1.0, 1.0]).to(self.device)
        self.target_distribution = torch.distributions.dirichlet.Dirichlet(self.dirichlet_param)
        self.reference_primal = self.target_distribution.sample(sample_shape = torch.Size([opts.reference_num])).to(self.device)
        self.reference_dual = self.mirror_map.nabla_psi(self.reference_primal)
        # set the structure of records
        self.records = easydict.EasyDict(
            {
                'support_primal': [], 'mass_primal': [], 'support_dual': [], 'mass_dual': [], 
                'w2_primal': [], 'w2_dual': []
            }
        )
        # init particles
        self.init_particles()

    @torch.no_grad()
    def init_particles(self):
        init_distribution = torch.distributions.dirichlet.Dirichlet(torch.ones(3) * 16)
        self.init_support_primal = init_distribution.sample(sample_shape = torch.Size([self.opts.particle_num])).to(self.device)
        self.init_support_dual = self.mirror_map.nabla_psi(self.init_support_primal)
        self.init_mass_primal = torch.ones(len(self.init_support_primal), device = self.device)
        self.init_mass_primal /= self.init_mass_primal.sum()
        self.init_mass_dual = self.init_mass_primal.clone()

    def grad_logp_dual(self, particles_dual: torch.Tensor) -> torch.Tensor:
        r"""
        gradient of target distribution in dual space

        Args:
            particles_dual (Tensor): a mini-batch tensor of shape (B x D), in dual space

        Return:
            grads (Tensor): a mini-batch tensor of shape (B x D), the gradient of target distritbuion in dual space
        """
        grads = self.dirichlet_param[:-1].view(1, -1) - torch.sum(self.dirichlet_param) * self.mirror_map.nabla_psi_star(particles_dual)
        return grads
    
    def grad_logp_primal(self, particles_primal: torch.Tensor) -> torch.Tensor:
        r"""
        gradient of target distribution in primal space

        Args:
            particles_primal (Tensor): a mini-batch tensor of shape (B x D), in primal space

        Returns:
            grads (Tensor): a mini-batch tensor of shape (B x D), the gradient of target distritbuion in primal space
        """
        grads = (self.dirichlet_param[:-1] - 1).view(1, -1) * torch.reciprocal(particles_primal) - (self.dirichlet_param[-1] - 1) * torch.reciprocal(1 - particles_primal.sum(dim = 1, keepdim = True))
        return grads

    def nabla2_psi_inv_grad_logp_primal(self, particles_primal: torch.Tensor) -> torch.Tensor:
        r"""
        NOTE: (lzj), actually I do not know what this calculate, maybe the matrix multiple between inverse hessian and gradient of target distribution in primal space.

        Args:
            particles_primal (Tensor): a mini-batch tensor of shape (B x D), in primal space

        Returns:
            results (Tensor): a mini-batch tensor of shape (B x D)
        """
        results = self.dirichlet_param[:-1].view(1, -1) - 1 - torch.sum(self.dirichlet_param - 1) * particles_primal
        return results

    def potential_primal(self, particles_primal: torch.Tensor) -> torch.Tensor:
        r"""
        calculate the potenial of target distribution in primal space

        Args:
            particles_primal (Tensor): a mini-batch tensor of shape (B x D), in primal space
        
        Returns:
            results (Tensor): a mini-batch tensor of shape (B)
        """
        result =  - torch.sum((self.dirichlet_param[:-1].view(1, -1) - 1) * torch.log(particles_primal), dim = 1) - (self.dirichlet_param[-1] - 1) * torch.log(1 - particles_primal.sum(1))
        return result

    def potential_dual(self, particles_dual: torch.Tensor) -> torch.Tensor:
        r"""
        calculate the potenial of target distribution in dual space

        Args:
            particles_primal (Tensor): a mini-batch tensor of shape (B x D), in dual space
        
        Returns:
            results (Tensor): a mini-batch tensor of shape (B)
        """
        result = - torch.sum(self.dirichlet_param[:-1].view(1, -1) * particles_dual, dim = 1) + self.dirichlet_param.sum() * self.mirror_map.psi_star(particles_dual, keepdim = False)
        return result
    
    def func_call(self, func_name: str, features: torch.Tensor = None, labels: torch.Tensor = None):
        r"""
        a warpper that takes function's name as input, and return the corresponding function 

        the args features and labels will not be used 
        """
        if __debug__ and not isinstance(func_name, str):
            raise RuntimeError('the func_name should be type of str')
        if __debug__ and not hasattr(self, func_name):
            raise RuntimeError('%s has no method %s'%(self.__class__, func_name))
        return self.__getattribute__(func_name)


    @torch.no_grad()
    def evaluation(self, support_primal: torch.Tensor, mass_primal: torch.Tensor, support_dual: torch.Tensor, mass_dual: torch.Tensor, writer, logger, count: int, save_dir: str) -> None:
        r"""
        evaluate the approximation error and save the results

        Args:
            suppport_primal (Tensor): a mini-batch tensor of shape (B x D), in primal space
            mass_primal (Tensor): a mini-batch tensor of shape (B), in primal space
            support_dual (Tensor): a mini-batch tensor of shape (B x D), in dual space
            mass_dual (Tensor): a mini-batch tensor if shape (B), in dual space
            writer: SummaryWriter of tensorboard
            logger: logger from logging lib
            count (int): same as the global_step in tensorboard
            save_dir (str): the directory to save the logging result
        """
        # calculate the approximation error
        w2_primal = utils.func.calculate_w2(support_primal, mass_primal, self.reference_primal, torch.ones(len(self.reference_primal)) / len(self.reference_primal))
        w2_dual = utils.func.calculate_w2(support_dual, mass_dual, self.reference_dual, torch.ones(len(self.reference_dual)) / len(self.reference_dual))
        self.records.w2_primal.append(w2_primal)
        self.records.w2_dual.append(w2_dual)
        self.records.support_primal.append(support_primal.cpu())
        self.records.mass_primal.append(mass_primal.cpu())
        self.records.support_dual.append(support_dual.cpu())
        self.records.mass_dual.append(mass_dual.cpu())
        writer.add_scalar('w2_primal', w2_primal, global_step = count)
        writer.add_scalar('w2_dual', w2_dual, global_step = count)
        logger.info('count: {}, w2_primal: {:.2e}, w2_dual: {:.2e}'.format(count, w2_primal, w2_dual))
        # plot the particles in primal and dual space
        fig = self.plot_particles(support_primal, mass_primal, support_dual, mass_dual, size = 20)        
        plt.savefig(os.path.join(save_dir, 'count%s.png'%count), pad_inches = 0.0, dpi = 150)
        plt.close()


    @torch.no_grad()
    def final_process(self, support_primal: torch.Tensor, mass_primal: torch.Tensor, support_dual: torch.Tensor, mass_dual: torch.Tensor, writer, logger, save_dir, is_save:bool = False) -> None:
        r"""
        some post_processing after the main loop

        Args:
            suppport_primal (Tensor): a mini-batch tensor of shape (B x D), in primal space
            mass_primal (Tensor): a mini-batch tensor of shape (B), in primal space
            support_dual (Tensor): a mini-batch tensor of shape (B x D), in dual space
            mass_dual (Tensor): a mini-batch tensor if shape (B), in dual space
            writer: SummaryWriter of tensorboard
            logger: logger from logging lib
            save_dir (str): the directory to save the processing result
            is_save (bool): whether saving the evaluation result and the particles
        """
        if is_save:
            torch.save(vars(self.records), os.path.join(save_dir, 'records.pt'))
        # plot the particles in primal and dual space
        fig = self.plot_particles(support_primal, mass_primal, support_dual, mass_dual, size = 20)        
        plt.savefig(os.path.join(save_dir, 'final.png'), pad_inches = 0.0, dpi = 150)
        plt.close()

    def plot_particles(self, support_primal:Union[torch.Tensor, np.ndarray], mass_primal:Union[torch.Tensor, np.ndarray], support_dual:Union[torch.Tensor, np.ndarray], mass_dual:Union[torch.Tensor, np.ndarray], size:float):
        r"""
        return scatter figure of particles in both primal and dual space

        Args:
            suppport_primal (Tensor or ndarray): a mini-batch tensor of shape (B x D), in primal space
            mass_primal (Tensor or ndarray): a mini-batch tensor of shape (B), in primal space
            support_dual (Tensor or ndarray): a mini-batch tensor of shape (B x D), in dual space
            mass_dual (Tensor or ndarray): a mini-batch tensor if shape (B), in dual space
            size (float): size in plot.scatter

        Returns:
            figure of matplotlib.pyplot
        """
        # check the inputs 
        inputs = [support_primal, mass_primal, support_dual, mass_dual]
        if __debug__ and any([(not isinstance(x, torch.Tensor)) and (not isinstance(x, np.ndarray)) for x in inputs]):
            raise RuntimeError('input support or mass should be type of tensor or ndarray')
        support_primal = support_primal.detach().cpu().numpy() if isinstance(support_primal, torch.Tensor) else support_primal
        mass_primal = mass_primal.detach().cpu().numpy() if isinstance(mass_primal, torch.Tensor) else mass_primal
        support_dual = support_dual.detach().cpu().numpy() if isinstance(support_dual, torch.Tensor) else support_dual
        mass_dual = mass_dual.detach().cpu().numpy() if isinstance(mass_dual, torch.Tensor) else mass_dual
        # set the color for each particle
        colors = np.cos(np.pi * support_primal[:, 0]) * np.cos(np.pi * support_primal[:, 1])
        # plot the scatter of algorithms' approximation results
        fig = plt.figure(figsize = (4.8 * 2, 4.8 * 2))
        plt.subplot(221)
        weights = mass_primal * len(mass_primal) * size
        plt.scatter(support_primal[:, 0], support_primal[:, 1], alpha = 0.5, s = weights, c = colors, cmap = 'hsv')
        plt.title('results in primal space')
        plt.subplot(222)
        weights = mass_primal * len(mass_dual) * size
        plt.scatter(support_dual[:, 0], support_dual[:, 1], alpha = 0.5, s = weights, c = colors, cmap = 'hsv')
        plt.title('results in dual space')
        # plot the scatter of reference, directly sampled from dirichlet distribution
        reference_support_primal = self.reference_primal.cpu().numpy()
        reference_support_dual = self.reference_dual.cpu().numpy()
        colors = np.cos(np.pi * reference_support_primal[:, 0]) * np.cos(np.pi * reference_support_primal[:, 1])
        plt.subplot(223)
        plt.scatter(reference_support_primal[:, 0], reference_support_primal[:, 1], alpha = 0.5, s = 5, c = colors, cmap = 'hsv')
        plt.title('reference in primal space')
        plt.subplot(224)
        plt.scatter(reference_support_dual[:, 0], reference_support_dual[:, 1], alpha = 0.5, s = 5, c = colors, cmap = 'hsv')
        plt.tight_layout()
        return fig
        
           

    # def save_final_results(self, writer, save_folder, particles_primal, particles_dual, mass):
    #     dim0, dim1 = particles_dual[:, 0], particles_dual[:, 1]
    #     interval0, interval1 = (dim0.max() - dim0.min()).item(), (dim1.max() - dim1.min()).item()
    #     figure = self.plot_pars(particles_primal, xlim = [-0.1, 1], ylim = [-0.1, 1], alpha = 0.8)
    #     writer.add_figure(tag = 'primal', figure = figure)
    #     plt.close()
    #     figure = self.plot_pars(
    #         particles_dual, 
    #         xlim = [dim0.min().item() - 0.3 * interval0, dim0.max().item() + 0.3 * interval0], 
    #         ylim = [dim1.min().item() - 0.3 * interval1, dim1.max().item() + 0.3 * interval1], 
    #         alpha = 0.8)
    #     writer.add_figure(tag = 'dual', figure = figure)
    #     plt.close()     
    
    # def hmc_dual(self, burn_in, outer_iter, inner_iter, p_num, lr):
    #     _, particle_dual = self.init_particles(p_num, self.particle_dim)
    #     pars = []
    #     accu_accept_ratio = 0.0
    #     for i in tqdm(range(burn_in + outer_iter)):
    #         q = particle_dual.clone()
    #         velocity = torch.randn_like(particle_dual, device = self.device)
    #         p = velocity.clone()
    #         grads = - self.grad_logp_dual(q, features = None, labels = None)
    #         p = p - 0.5 * lr * grads
    #         for k in range(inner_iter):
    #             q = q + lr * p
    #             grads = - self.grad_logp_dual(q, features = None, labels = None)
    #             if k != (inner_iter - 1): p = p - lr * grads
    #         p = p - 0.5 * lr * grads
    #         p = -p
    #         curr_u = self.potential_dual(particle_dual, None, None)
    #         curr_k = velocity.pow(2).sum(1) / 2
    #         prop_u = self.potential_dual(q, None, None)
    #         prop_k = p.pow(2).sum(1) / 2
    #         accept_prob = torch.minimum(torch.exp(curr_u + curr_k - prop_u - prop_k), torch.ones(p_num, device = self.device))
    #         accu_accept_ratio += accept_prob.mean()
    #         rand = torch.rand(p_num, device = self.device)
    #         particle_dual[rand < accept_prob] = q[rand < accept_prob].clone() # accept
    #         if i >= burn_in:
    #             pars.append(particle_dual.clone())
    #         self.writer.add_scalar('mean_acc_prob', accu_accept_ratio / (i + 1), global_step = i)
    #     pars = torch.cat(pars, dim = 0)
    #     sq_dist = torch.cdist(pars, pars, p = 2)**2
    #     state = {'hmc_pars': pars, 'sq_dist': sq_dist.median()}
    #     basic.create_dirs_if_not_exist('./hmc_reference/%s'%(self.file_name))
    #     torch.save(state, './hmc_reference/%s/state.pth'%(self.file_name))
    #     # plot reference particles
    #     self.plot_pars(self.ref_par_primal, xlim = [-0.1, 1], ylim = [-0.1, 1], alpha = 0.3)
    #     plt.savefig('./hmc_reference/%s/ref_primal.png'%(self.file_name))
    #     plt.close()
    #     self.plot_pars(entropic.nabla_psi_star(pars), xlim = [-0.1, 1], ylim = [-0.1, 1], alpha = 0.3)
    #     plt.savefig('./hmc_reference/%s/hmc_primal.png'%(self.file_name))
    #     plt.close()
    #     # nabla_psi_ref_par_primal = entropic.nabla_psi(self.ref_par_primal[:, :-1])
    #     # dim0, dim1 = nabla_psi_ref_par_primal[:, 0], nabla_psi_ref_par_primal[:, 1]
    #     # interval0, interval1 = (dim0.max() - dim0.min()).item(), (dim1.max() - dim1.min()).item()
    #     # self.plot_pars(
    #     #     nabla_psi_ref_par_primal, 
    #     #     xlim = [dim0.min().item() - 0.3 * interval0, dim0.max().item() + 0.3 * interval0], 
    #     #     ylim = [dim1.min().item() - 0.3 * interval1, dim1.max().item() + 0.3 * interval1], 
    #     #     alpha = 0.3)
    #     # plt.savefig('./hmc_reference/%s/ref_dual.png'%(self.file_name))
    #     # plt.close()
    #     dim0, dim1 = pars[:, 0], pars[:, 1]
    #     interval0, interval1 = (dim0.max() - dim0.min()).item(), (dim1.max() - dim1.min()).item()
    #     self.plot_pars(
    #         pars, 
    #         xlim = [dim0.min().item() - 0.3 * interval0, dim0.max().item() + 0.3 * interval0], 
    #         ylim = [dim1.min().item() - 0.3 * interval1, dim1.max().item() + 0.3 * interval1],
    #         alpha = 0.3)
    #     plt.savefig('./hmc_reference/%s/hmc_dual.png'%(self.file_name))
    #     plt.close()

    # def plot_pars(self, particles, xlim, ylim, alpha = 0.2):
    #     fig = plt.figure(num = 1)
    #     plt.scatter(particles[:, 0].cpu().numpy(), particles[:, 1].cpu().numpy(), alpha = alpha, s = 10, c = 'r')
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    #     plt.tight_layout()
    #     return fig

