import pretty_errors, time, argparse, importlib, os, torch, numpy as np, easydict, yaml
import utils
from dataloader import myDataLoader
from functools import partial

def main(opts):
    # set random seed
    utils.common.set_random_seed(opts.seed)
    # set logger and tensorboard writer
    writer, logger, save_dir = utils.common.get_logger(
        opts, opts.algorithm, 
        os.path.join(opts.result_dir, utils.common.set_name(opts))
    )
    # set dataloader
    myloader = myDataLoader(opts)
    train_loader = myloader.get_train_loader()
    train_iter = iter(train_loader)
    # import algorithm and task
    task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, myloader.get_dataset_info())
    algorithm = importlib.import_module('algorithms.%s'%opts.algorithm).__getattribute__(opts.algorithm)(
        opts, task.init_support_primal, task.init_mass_primal, task.init_support_dual, task.init_mass_dual, task.mirror_map
    )
    # main loop
    total_time = 0
    evaluation_count = 0
    step = 0
    while True:
        # get the particles at primal/dual space and check Nan value
        support_primal, mass_primal, support_dual, mass_dual = algorithm.get_state()
        assert (support_primal is not None) and (mass_primal is not None) and (support_dual is not None) and (mass_dual is not None), 'algorithm should provide particles in both primal and dual space for evaluation'
        utils.common.check_nan(support_primal, mass_primal, int(total_time) if opts.time_mode else step, logger)
        utils.common.check_nan(support_dual, mass_dual, int(total_time) if opts.time_mode else step, logger)
        # evaluation according to the param "eval_interval"
        if (not opts.time_mode) and (step >= evaluation_count * opts.eval_interval or step >= opts.max_iter) or (opts.time_mode) and (total_time >= evaluation_count * opts.eval_interval or total_time >= opts.max_time):
            evaluation_count += 1
            task.evaluation(support_primal, mass_primal, support_dual, mass_dual, writer = writer, logger = logger, count = int(total_time) if opts.time_mode else step, save_dir = save_dir)
        if (not opts.time_mode) and (step >= opts.max_iter) or (opts.time_mode) and (total_time >= opts.max_time):
            break
        # get features and labels from train iter
        try: features, labels = next(train_iter)
        except: 
            train_iter = iter(train_loader)
            features, labels = next(train_iter) 
        # one step forward
        _start_time = time.time()
        algorithm.one_step_update(
            lr = opts.lr, 
            alpha = opts.alpha, 
            task_funcs = partial(task.func_call, features = features, labels = labels)  # task.func_call is a function that returns specific function according to arg "name", e.g. grad_logp_dual
        )
        _end_time = time.time()
        total_time += _start_time - _end_time
        step += 1
    # final process
    task.final_process(
        support_primal, mass_primal, support_dual, mass_dual, writer = writer, logger = logger, save_dir = save_dir, is_save = opts.save_particles if hasattr(opts, 'save_particles') else None
    )
    # close the tensorboard
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = './configs/default.yaml', help = 'the path of yaml config file')
    parser.add_argument('--over_writen', action = 'store_true', help = 'if true, the args of argparser will overwrite the args of config-file')
    parser.add_argument('--algorithm', type = str, default = 'MSVGDd', help = 'the name of algorithm')
    parser.add_argument('--task', type = str, default = 'dirichletSingle', help = 'the name of task')

    parser.add_argument('--particle_num', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 0.2)
    parser.add_argument('--alpha', type = float, default = 0.02, help = 'the param of balancing fisher-rao step')
    parser.add_argument('--knType', type = str, default = 'rbf', choices = ['rbf', 'imq'], help = 'type of kernel function')
    parser.add_argument('--bwType', type = str, default = 'med', choices = ['med', 'heu', 'nei', 'fix'], help = 'type of bandwith of kernel function')
    parser.add_argument('--bwVal', type = float, default = 0.1)

    parser.add_argument('--n_eigen_threshold', type = float, default = 0.98, help = 'args in SVMirrorD method, https://arxiv.org/abs/2106.12506')

    parser.add_argument('--max_iter', type = int, default = 1500)
    parser.add_argument('--max_time', type = int, default = 120)
    parser.add_argument('--time_mode', action = 'store_true')
    parser.add_argument('--optim', type = str, default = 'sgd', choices = ['sgd', 'rmsprop'])
    parser.add_argument('--optim_rho', type = float, default = 0.9, help = 'param in RMSProp')

    parser.add_argument('--reference_num', type = int, default = 2000)

    parser.add_argument('--eval_interval', type = int, default = 100)
    parser.add_argument('--device', type = str, default = 'cuda:3')
    parser.add_argument('--result_dir', type = str, default = 'results')
    parser.add_argument('--time_as_dir', action = 'store_true', help = 'if true, the name of save_dir will be generated automatically according to the time')
    parser.add_argument('--seed', type = int, default = 0)
    opts_parser = parser.parse_args()

    # load yaml config
    opts = None
    if os.path.exists(opts_parser.config_path):
        with open(opts_parser.config_path, 'r') as file:
            opts = easydict.EasyDict(yaml.load(file.read(), Loader = yaml.FullLoader))
    # update params by parser
    if not opts:
        opts = easydict.EasyDict(vars(opts_parser))
    elif opts_parser.over_writen:
        opts.update(vars(opts_parser))
    else:
        pass

    main(opts)