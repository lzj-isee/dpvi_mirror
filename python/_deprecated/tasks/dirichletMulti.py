from dataloader import myDataLoader
import torch, ot, basic, os
import numpy as np
import matplotlib.pyplot as plt
import importlib
from mirror_map import entropic
from tqdm import tqdm

class functions(myDataLoader):
    def __init__(self, opts) -> None:
        super().__init__(opts)
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.file_name = os.path.splitext(os.path.basename(__file__))[0]
        # set model parameters
        self.particle_dim = 2
        self.model_dim = 3
        self.alpha = torch.Tensor([0.1, 0.1, 0.1]).to(self.device)
        # plot setting
        self.plot_size, self.min_ratio, self.max_ratio = 50, 0.2, 5
        # reference
        self.ref_par_primal_num = 5000
        self.target_distribution = torch.distributions.dirichlet.Dirichlet(self.alpha)
        self.ref_par_primal = self.target_distribution.sample(sample_shape = torch.Size([self.ref_par_primal_num])).to(self.device)
        try: self.ref_par_dual = torch.load('./hmc_reference/%s/state.pth'%(self.file_name))['hmc_pars'].to(self.device)
        except: print('no hmc reference')
        self.w2_primal, self.w2_dual = [], []
        self.ed_primal, self.ed_dual = [], []
        # function
        self.mirror_map = importlib.import_module('mirror_map.entropic')

    def kernel_calc(self, particles):
        return super().kernel_calc(particles)

    @torch.no_grad()
    def init_particles(self, particle_num, particle_dim):
        init_distribution = torch.distributions.dirichlet.Dirichlet(torch.ones(particle_dim + 1) * 16)
        init_par_primal = init_distribution.sample(sample_shape = torch.Size([particle_num])).to(self.device)[:, :-1]
        return init_par_primal, entropic.nabla_psi(init_par_primal)

    @torch.no_grad()
    def grad_logp_dual(self, particles_dual, features = None, labels = None):
        grads = self.alpha[:-1].view(1, -1) - torch.sum(self.alpha) * entropic.nabla_psi_star(particles_dual)
        return grads
    
    @torch.no_grad()
    def grad_logp_primal(self, particles_primal, features = None, labels = None):
        grads = (self.alpha[:-1] - 1).view(1, -1) / particles_primal - \
            (self.alpha[-1] - 1) / (1 - particles_primal.sum(dim = 1, keepdim = True))
        return grads

    @torch.no_grad()
    def nabla2_psi_inv_grad_logp_primal(self, particles_primal, features = None, labels = None):
        results = self.alpha[:-1].view(1, -1) - 1 - torch.sum(self.alpha - 1) * particles_primal
        return results

    @torch.no_grad()
    def potential_primal(self, particles_primal, features = None, labels = None):
        result =  - torch.sum((self.alpha[:-1].view(1, -1) - 1) * torch.log(particles_primal), dim = 1) - \
            (self.alpha[-1] - 1) * torch.log(1 - particles_primal.sum(1))
        return result

    @torch.no_grad()
    def potential_dual(self, particles_dual, features = None, labels = None):
        result = - torch.sum(self.alpha[:-1].view(1, -1) * particles_dual, dim = 1) + \
            self.alpha.sum() * entropic.psi_star(particles_dual, keepdim = False)
        return result

    @torch.no_grad()
    def evaluation(self, particles_primal, particles_dual, mass, writer, global_step):
        # evaluate energy distance, primal
        x, y = self.ref_par_primal[:, :-1], particles_primal
        # xx = torch.cdist(x, x, p = 2).mean()
        # yy = torch.cdist(y, y, p = 2).mean()
        # xy = torch.cdist(x, y, p = 2).mean()
        # self.ed_primal.append((2 * xy - xx - yy).item())
        # evaluate 2-Wasserstein distance, primal
        cost_matrix = (torch.cdist(particles_primal, x)**2).cpu().numpy()
        mass_numpy = mass.cpu().numpy().astype(np.float64) # from tensor to numpy, need extra normalization
        transport_plan = ot.emd(mass_numpy / mass_numpy.sum(), ot.unif(self.ref_par_primal_num), cost_matrix)
        self.w2_primal.append(np.sqrt((cost_matrix * transport_plan).sum()))
        # evaluate 2-Wasserstein distance, dual
        cost_matrix = (torch.cdist(particles_dual, self.ref_par_dual)**2).cpu().numpy()
        transport_plan = ot.emd(mass_numpy / mass_numpy.sum(), ot.unif(len(self.ref_par_dual)), cost_matrix)
        self.w2_dual.append(np.sqrt((cost_matrix * transport_plan).sum()))  
        # save to tensorboard
        writer.add_scalar('w2 primal', self.w2_primal[-1], global_step = global_step)
        # writer.add_scalar('ED primal', self.ed_primal[-1], global_step = global_step)
        writer.add_scalar('w2 dual', self.w2_dual[-1], global_step = global_step)
           

    def save_final_results(self, writer, save_folder, particles_primal, particles_dual, mass):
        dim0, dim1 = particles_dual[:, 0], particles_dual[:, 1]
        interval0, interval1 = (dim0.max() - dim0.min()).item(), (dim1.max() - dim1.min()).item()
        figure = self.plot_pars(particles_primal, xlim = [-0.1, 1], ylim = [-0.1, 1], alpha = 0.8)
        writer.add_figure(tag = 'primal', figure = figure)
        plt.close()
        figure = self.plot_pars(
            particles_dual, 
            xlim = [dim0.min().item() - 0.3 * interval0, dim0.max().item() + 0.3 * interval0], 
            ylim = [dim1.min().item() - 0.3 * interval1, dim1.max().item() + 0.3 * interval1], 
            alpha = 0.8)
        writer.add_figure(tag = 'dual', figure = figure)
        plt.close()     
    
    def hmc_dual(self, burn_in, outer_iter, inner_iter, p_num, lr):
        _, particle_dual = self.init_particles(p_num, self.particle_dim)
        pars = []
        accu_accept_ratio = 0.0
        for i in tqdm(range(burn_in + outer_iter)):
            q = particle_dual.clone()
            velocity = torch.randn_like(particle_dual, device = self.device)
            p = velocity.clone()
            grads = - self.grad_logp_dual(q, features = None, labels = None)
            p = p - 0.5 * lr * grads
            for k in range(inner_iter):
                q = q + lr * p
                grads = - self.grad_logp_dual(q, features = None, labels = None)
                if k != (inner_iter - 1): p = p - lr * grads
            p = p - 0.5 * lr * grads
            p = -p
            curr_u = self.potential_dual(particle_dual, None, None)
            curr_k = velocity.pow(2).sum(1) / 2
            prop_u = self.potential_dual(q, None, None)
            prop_k = p.pow(2).sum(1) / 2
            accept_prob = torch.minimum(torch.exp(curr_u + curr_k - prop_u - prop_k), torch.ones(p_num, device = self.device))
            accu_accept_ratio += accept_prob.mean()
            rand = torch.rand(p_num, device = self.device)
            particle_dual[rand < accept_prob] = q[rand < accept_prob].clone() # accept
            if i >= burn_in:
                pars.append(particle_dual.clone())
            self.writer.add_scalar('mean_acc_prob', accu_accept_ratio / (i + 1), global_step = i)
        pars = torch.cat(pars, dim = 0)
        sq_dist = torch.cdist(pars, pars, p = 2)**2
        state = {'hmc_pars': pars, 'sq_dist': sq_dist.median()}
        basic.create_dirs_if_not_exist('./hmc_reference/%s'%(self.file_name))
        torch.save(state, './hmc_reference/%s/state.pth'%(self.file_name))
        # plot reference particles
        self.plot_pars(self.ref_par_primal, xlim = [-0.1, 1], ylim = [-0.1, 1], alpha = 0.3)
        plt.savefig('./hmc_reference/%s/ref_primal.png'%(self.file_name))
        plt.close()
        self.plot_pars(entropic.nabla_psi_star(pars), xlim = [-0.1, 1], ylim = [-0.1, 1], alpha = 0.3)
        plt.savefig('./hmc_reference/%s/hmc_primal.png'%(self.file_name))
        plt.close()
        # nabla_psi_ref_par_primal = entropic.nabla_psi(self.ref_par_primal[:, :-1])
        # dim0, dim1 = nabla_psi_ref_par_primal[:, 0], nabla_psi_ref_par_primal[:, 1]
        # interval0, interval1 = (dim0.max() - dim0.min()).item(), (dim1.max() - dim1.min()).item()
        # self.plot_pars(
        #     nabla_psi_ref_par_primal, 
        #     xlim = [dim0.min().item() - 0.3 * interval0, dim0.max().item() + 0.3 * interval0], 
        #     ylim = [dim1.min().item() - 0.3 * interval1, dim1.max().item() + 0.3 * interval1], 
        #     alpha = 0.3)
        # plt.savefig('./hmc_reference/%s/ref_dual.png'%(self.file_name))
        # plt.close()
        dim0, dim1 = pars[:, 0], pars[:, 1]
        interval0, interval1 = (dim0.max() - dim0.min()).item(), (dim1.max() - dim1.min()).item()
        self.plot_pars(
            pars, 
            xlim = [dim0.min().item() - 0.3 * interval0, dim0.max().item() + 0.3 * interval0], 
            ylim = [dim1.min().item() - 0.3 * interval1, dim1.max().item() + 0.3 * interval1],
            alpha = 0.3)
        plt.savefig('./hmc_reference/%s/hmc_dual.png'%(self.file_name))
        plt.close()

    def plot_pars(self, particles, xlim, ylim, alpha = 0.2):
        fig = plt.figure(num = 1)
        plt.scatter(particles[:, 0].cpu().numpy(), particles[:, 1].cpu().numpy(), alpha = alpha, s = 10, c = 'r')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()
        return fig




    

