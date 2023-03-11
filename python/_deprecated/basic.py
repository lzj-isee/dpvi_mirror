import torch, os, random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
# ------------------------------------------------------ common settings --------------------------------------------------
class basic(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts
        # set the random seed
        os.environ['PYTHONHASHSEED'] = str(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.random.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(opts.seed) 
            torch.cuda.manual_seed_all(opts.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # set device
        self.device = torch.device('cuda:{:}'.format(opts.gpu)) if opts.gpu != -1 else torch.device('cpu')
        # set information
        # time_now = time.strftime("_%m-%d_%H:%M:%S", time.localtime())
        if opts.algorithm in ['DMedBLOB']:
            self.save_name = opts.algorithm + '_{:.2e}'.format(opts.lr) + 'a[{:.2e}]'.format(opts.alpha) + '_s{}'.format(opts.seed)
        elif opts.algorithm in ['MedBLOBv']:
            self.save_name = opts.algorithm + '_{:.2e}'.format(opts.lr) + 'bwV[{:.1e}]'.format(opts.bwVal) + '_s{}'.format(opts.seed)
        else:
            try: self.save_name = opts.algorithm + '_{}'.format(opts.suffix) + '_{:.2e}'.format(opts.lr) + '_s{}'.format(opts.seed)
            except: self.save_name = opts.algorithm + '_{:.2e}'.format(opts.lr) + '_s{}'.format(opts.seed)
        self.save_folder = os.path.join(opts.save_folder, self.save_name)
        # creat log folder
        create_dirs_if_not_exist(self.save_folder)
        # clear old log files
        clear_log(self.save_folder)
        # save settings
        save_settings(self.save_folder, vars(self.opts))
        # creat tensorboard
        self.writer = SummaryWriter(log_dir = self.save_folder)

    def kernel_calc(self, particles, bw_h_spe = None):
        cross_diff = particles[:, None, :] - particles[None, :, :]
        sq_distance = torch.sum(cross_diff.pow(2), dim = 2)
        if bw_h_spe is not None:
            bw_h = bw_h_spe
        elif self.opts.bwType == 'med': # SVGD
            bw_h = torch.median(sq_distance + 1e-5) / np.log(self.opts.particle_num)
        elif self.opts.bwType == 'nei': # GFSD, Blob
            bw_h = sq_distance + torch.diag(torch.diag(sq_distance) + sq_distance.max())
            bw_h = bw_h.min(dim = 1)[0].mean()
        elif self.opts.bwType == 'fix': # fixed bandwidth
            bw_h = self.opts.bwVal
        elif self.opts.bwType == 'heu': # MSVGD
            n_elems = sq_distance.shape[0] * sq_distance.shape[1]
            topk_values = torch.topk(sq_distance.view(-1), k = n_elems // 2, sorted = False).values
            bw_h = torch.min(topk_values)
            bw_h = torch.where(bw_h == 0, torch.ones_like(bw_h), bw_h)
        else: 
            raise ValueError('no such bandwidth type')
        if self.opts.knType == 'imq': 
            kernel = (1 + sq_distance / bw_h).pow(-0.5)
            nabla_kernel = -kernel.pow(3)[:, :, None] * cross_diff / bw_h
        elif self.opts.knType == 'rbf':
            kernel = (-sq_distance / bw_h).exp()
            nabla_kernel = -2 * cross_diff * kernel[:, :, None] / bw_h
        else:
            raise ValueError('no such kernel type')
        return kernel, nabla_kernel, bw_h
# ------------------------------------------------------- optimizer ---------------------------------------------------------------
class rmsprop(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts
        self.rms = 0

    def apply_grads(self, particles, grads):
        self.rms = self.opts.optim_rho * self.rms + (1 - self.opts.optim_rho) * grads.pow(2)
        return particles - self.opts.lr * grads / (self.rms.sqrt() + 1e-7)

class sgd(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts

    def apply_grads(self, particles, grads):
        return particles - self.opts.lr * grads
# ------------------------------------------------------------------- utils ----------------------------------------------------------------
@torch.no_grad()
def check(save_folder, particles, mass, curr_iter_count):
    if particles.max() != particles.max(): 
        with open(os.path.join(save_folder, 'NanParticle.txt'), mode='w') as f:
            f.write('iter: {}'.format(curr_iter_count))
        raise ValueError('Nan')
    if mass.min() <= 0: 
        with open(os.path.join(save_folder, 'NegMass.txt'), mode='w') as f:
            f.write('iter: {}'.format(curr_iter_count))
        raise ValueError('non-positive mass')

def save_settings(save_folder, settings):
    with open(save_folder+'/settings.md',mode='w') as f:
        for key in settings:
            f.write(key+': '+'{}'.format(settings[key])+' \n')

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def save_final_results(save_folder, result_dict):
    # save results
    with open(os.path.join(save_folder, 'results.md'), mode='w') as f:
        for key in result_dict:
            f.write(key + ': ' + '{}'.format(result_dict[key])+ '\n')

def clear_log(save_folder):
    # clear log files
    if os.path.exists(save_folder):
        names = os.listdir(save_folder)
        for name in names:
            os.remove(save_folder+'/'+name)
        print('clear files in {}'.format(save_folder))
    else:
        pass