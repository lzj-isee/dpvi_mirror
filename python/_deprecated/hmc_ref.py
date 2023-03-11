import importlib
from easydict import EasyDict

# opts = {
#     'algorithm': 'HMC', 'gpu': -1, 'seed': 0, 'split_seed': 19, 'split_size': 0.1, 'task': 'dirichletSingle', 
#     'particle_num': 50, 'epochs': 1000, 'batch_size': 1, 'lr': 0.3, 'save_folder': './hmc_reference/dirichletSingle'
# }
# task =  importlib.import_module('tasks.{:}'.format('dirichletSingle')).__getattribute__('functions')(EasyDict(opts))
# task.hmc_dual(burn_in = 100, outer_iter = 100, inner_iter = 100, p_num = 50, lr = opts['lr'])

opts = {
    'algorithm': 'HMC', 'gpu': -1, 'seed': 0, 'split_seed': 19, 'split_size': 0.1, 'task': 'quadratic', 
    'particle_num': 50, 'epochs': 1000, 'batch_size': 1, 'lr': 0.2, 'save_folder': './hmc_reference/quadratic'
}
task =  importlib.import_module('tasks.{:}'.format('quadratic')).__getattribute__('functions')(EasyDict(opts))
task.hmc_dual(burn_in = 100, outer_iter = 100, inner_iter = 100, p_num = 50, lr = opts['lr'])

# opts = {
#     'algorithm': 'HMC', 'gpu': -1, 'seed': 0, 'split_seed': 19, 'split_size': 0.1, 'task': 'dirichletMulti', 
#     'particle_num': 50, 'epochs': 1000, 'batch_size': 1, 'lr': 0.2, 'save_folder': './hmc_reference/dirichletMulti'
# }
# task =  importlib.import_module('tasks.{:}'.format(opts['task'])).__getattribute__('functions')(EasyDict(opts))
# task.hmc_dual(burn_in = 100, outer_iter = 100, inner_iter = 100, p_num = 50, lr = opts['lr'])

# opts = {
#     'algorithm': 'HMC', 'gpu': -1, 'seed': 0, 'split_seed': 19, 'split_size': 0.1, 'task': 'dirichletUniform', 
#     'particle_num': 50, 'epochs': 1000, 'batch_size': 1, 'lr': 0.2, 'save_folder': './hmc_reference/dirichletUniform'
# }
# task =  importlib.import_module('tasks.{:}'.format(opts['task'])).__getattribute__('functions')(EasyDict(opts))
# task.hmc_dual(burn_in = 100, outer_iter = 100, inner_iter = 100, p_num = 50, lr = opts['lr'])