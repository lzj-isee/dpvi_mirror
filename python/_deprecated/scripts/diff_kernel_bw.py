import importlib
from easydict import EasyDict
import sys, os
sys.path.append(os.path.dirname(os.path.realpath('__file__')))

opts = {
    'gpu': 3, 'seed': 0, 'split_seed': 19, 'split_size': 0.1, 'task': 'dirichlet', 
    'particle_num': 128, 'epochs': 4000, 'batch_size': 1, 'save_folder': './results_exp/kbw', 'eval_interval': 20, 
    'optimizer': 'sgd', 'optim_rho': 0.9
}
# -----------------------------------------------RBF Med------------------------------------------------------------
# opts['algorithm'], opts['suffix'] = 'MirrorSVGD', 'RbfMed'
# opts['knType'], opts['bwType'] = 'rbf', 'med'
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'SVMirrorD', 'RbfMed'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'rbf', 'med', 0.98
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDp', 'RbfMed'
# opts['knType'], opts['bwType'] = 'rbf', 'med'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDd', 'RbfMed'
# opts['knType'], opts['bwType'] = 'rbf', 'med'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# -----------------------------------------------IMQ MED------------------------------------------------------------
# opts['algorithm'], opts['suffix'] = 'MirrorSVGD', 'ImqMed'
# opts['knType'], opts['bwType'] = 'imq', 'med'
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'SVMirrorD', 'ImqMed'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'imq', 'med', 0.98
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDp', 'ImqMed'
# opts['knType'], opts['bwType'] = 'imq', 'med'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDd', 'ImqMed'
# opts['knType'], opts['bwType'] = 'imq', 'med'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# -----------------------------------------------RBF NEI------------------------------------------------------------
opts['algorithm'], opts['suffix'] = 'MirrorSVGD', 'RbfNei'
opts['knType'], opts['bwType'] = 'rbf', 'nei'
opts['lr'] = 0.03
importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'SVMirrorD', 'RbfNei'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'rbf', 'nei', 0.98
# opts['lr'] = 0.1
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDp', 'RbfNei'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 0.6
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDd', 'RbfNei'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 0.6
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# -----------------------------------------------IMQ NEI------------------------------------------------------------
# opts['algorithm'], opts['suffix'] = 'MirrorSVGD', 'ImqNei'
# opts['knType'], opts['bwType'] = 'imq', 'nei'
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'SVMirrorD', 'ImqNei'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'imq', 'nei', 0.98
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDp', 'ImqNei'
# opts['knType'], opts['bwType'] = 'imq', 'nei'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDd', 'ImqNei'
# opts['knType'], opts['bwType'] = 'imq', 'nei'
# opts['lr'] = 0.5
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# -----------------------------------------------RBF HEU------------------------------------------------------------
# opts['algorithm'], opts['suffix'] = 'MirrorSVGD', 'RbfHeu'
# opts['knType'], opts['bwType'] = 'rbf', 'heu'
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'SVMirrorD', 'RbfHeu'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'rbf', 'heu', 0.98
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDp', 'RbfHeu'
# opts['knType'], opts['bwType'] = 'rbf', 'heu'
# opts['lr'] = 0.1
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDd', 'RbfHeu'
# opts['knType'], opts['bwType'] = 'rbf', 'heu'
# opts['lr'] = 0.1
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# -----------------------------------------------IMQ HEU------------------------------------------------------------
# opts['algorithm'], opts['suffix'] = 'MirrorSVGD', 'ImqHeu'
# opts['knType'], opts['bwType'] = 'imq', 'heu'
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'SVMirrorD', 'ImqHeu'
# opts['knType'], opts['bwType'], opts['n_eigen_threshold'] = 'imq', 'heu', 0.98
# opts['lr'] = 0.03
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDp', 'ImqHeu'
# opts['knType'], opts['bwType'] = 'imq', 'heu'
# opts['lr'] = 0.3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'], opts['suffix'] = 'MirroredSVGDd', 'ImqHeu'
# opts['knType'], opts['bwType'] = 'imq', 'heu'
# opts['lr'] = 0.5
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))






















# opts['algorithm'] = 'MirrorGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredGFSD'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 3e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirrorBLOB'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 1e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredBLOB'
# opts['knType'], opts['bwType'] = 'rbf', 'nei'
# opts['lr'] = 3e-3
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))


# opts['algorithm'] = 'MirrorLD'
# opts['lr'] = 7e-4
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))

# opts['algorithm'] = 'MirroredLD'
# opts['lr'] = 1e-2
# importlib.import_module('algorithms.{:}'.format(opts['algorithm'])).__getattribute__('run')(EasyDict(opts))