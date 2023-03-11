import importlib
from easydict import EasyDict

opts = {
    'gpu': 3, 'seed': 19, 'split_seed': 19, 'split_size': 0.1, 'task': 'dirichletSingle', 
    'particle_num': 256, 'epochs': 2000, 'batch_size': 1, 'save_folder': './results', 'eval_interval': 20, 
    'optimizer': 'sgd', 'optim_rho': 0.9
}
# --------------------------------------------------------------------------------------------------------------
# opts['algorithm'] = 'SVMirrorD'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'imq', 'med', 0.98
# opts['lr'] = 0.05
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedSVGDp'
# opts['knType'], opts['bwType'] = 'imq', 'med'
# opts['lr'] = 0.25
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MedSVGDp', 'rbf'
# opts['knType'], opts['bwType'] = 'rbf', 'med'
# opts['lr'] = 0.25
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedSVGDd'
# opts['knType'], opts['bwType'] = 'imq', 'med'
# opts['lr'] = 0.25
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedVSVGDd'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 0.5
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 0.01
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedDSVGD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'], opts['alpha'] = 0.5, 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

opts['algorithm'] = 'MedDSVGD2'
opts['knType'], opts['bwType'] = 'rbf', 'nei'
opts['lr'], opts['alpha'] = 0.5, 0.01
importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedBLOB'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 0.01
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'DMedBLOB'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'], opts['alpha'] = 0.01, 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MedBLOBv'
# opts['knType'], opts['bwType'], opts['bwVal'] = 'rbf', 'nei', 0.0003
# opts['lr'] = 0.0001
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))













# opts['algorithm'] = 'MorSVGD'
# opts['knType'], opts['bwType'] = 'imq', 'med'
# opts['lr'] = 0.05
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))



# opts['algorithm'] = 'MirrorGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-2
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirrorLD'
# opts['lr'] = 1.0e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredLD'
# opts['lr'] = 2.0e-2
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))