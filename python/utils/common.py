import torch, os, random, numpy as np, logging, sys, time, yaml
from torch.utils.tensorboard import SummaryWriter


def set_name(opts):
    r"""
    concatenate opts to create a save dir
    """
    keys = opts.keys()
    name = ''
    if 'algorithm' in keys:
        name += '%s_'%opts['algorithm']
    if opts.time_as_dir:
        name += time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        return name
    # if use time as the dir name, no specific param will be record
    if 'lr' in keys:
        name += 'lr[{:.1e}]'.format(opts.lr)
    if 'alpha' in keys:
        name += 'al[{:.1e}]'.format(opts.alpha)
    if 'knType' in keys and 'bwType' in keys:
        name += 'kn[{},{}]'.format(opts.knType, opts.bwType)
    if 'seed' in keys:
        name += 's[{}]'.format(opts.seed)
    return name
    

@torch.no_grad()
def check_nan(support: torch.Tensor, mass: torch.Tensor, curr_iter_count: int, logger = None) -> None:
    r"""
    check whether exists Nan value

    Args:
        support (Tensor): positions of particles 
        mass (Tensor): weight of particles
        curr_iter_count (int): global step in iteration, same as tensorboard
        logger: logger from lib logging
    """
    if support is not None and support.isnan().any(): 
        message = 'support value Nan at iter %d'%curr_iter_count
        if logger: logger.error(message)
        raise ValueError(message)
    if mass is not None and (mass <= 0).any(): 
        message = 'non-positive mass at iter %d'%curr_iter_count
        if logger: logger.error(message)
        raise ValueError(message)
    
def log_settings(logger, settings: dict):
    string = '\n'
    for key in settings: 
        string += ' - ' + key + ': ' + '{}'.format(settings[key])+' \n'
    logger.info(string)

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def clear_files(target_dir):
    if os.path.exists(target_dir):
        names = os.listdir(target_dir)
        for name in names:
            os.remove(os.path.join(target_dir, name))
        print('- clear files in {}'.format(target_dir))
    else:
        print('- empty target dir {}, nothing deleted'.format(target_dir))


def get_logger(opts, name: str, save_dir: str):
    r"""
    get logger from lib logging and lib tensorboard

    Args:
        opts: args in an launching
        name: used for lib logging (logging.getLogger(name))
        save_dir: the directory to save files from logging and tensorboard
    """
    # creat log folder
    create_dirs_if_not_exist(save_dir)
    # clear previous log files
    clear_files(save_dir)
    # creat tensorboard SummaryWriter
    writer = SummaryWriter(log_dir = save_dir)
    # creat logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler(stream = sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'), mode = 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # save the parameters
    log_settings(logger, vars(opts))
    with open(os.path.join(save_dir, 'args.yaml'), 'w') as file:
        file.write(yaml.dump(vars(opts)))
    return writer, logger, save_dir

def set_random_seed(seed):
    """
    set random seed for os, numpy, random and torch
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True