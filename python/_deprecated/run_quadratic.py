import importlib
from easydict import EasyDict

opts = {
    'gpu': 3, 'seed': 0, 'split_seed': 19, 'split_size': 0.1, 'task': 'quadratic', 
    'particle_num': 256, 'epochs': 1000, 'batch_size': 1, 'save_folder': './results', 'eval_interval': 20, 
    'optimizer': 'sgd', 'optim_rho': 0.9
}
# --------------------------------------------------------------------------------------------------------------
# opts['algorithm'] = 'MirrorSVGD'
# opts['knType'], opts['bwType'] = 'imq', 'nei'
# opts['lr'] = 0.06
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

opts['algorithm'] = 'SVMirrorD'
opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'imq', 'nei', 0.98
opts['lr'] = 0.06
importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredSVGDp'
# opts['knType'], opts['bwType'] = 'imq', 'nei'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredSVGDd'
# opts['knType'], opts['bwType'] = 'imq', 'nei'
# opts['lr'] = 0.25
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirrorGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-2
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirrorBLOB'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredBLOB'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-2
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirrorLD'
# opts['lr'] = 6e-4
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredLD'
# opts['lr'] = 2.0e-2
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))