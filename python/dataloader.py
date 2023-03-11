
class myDataLoader(object):
    def __init__(self, opts) -> None:
        self.opts = opts
        self.device = opts.device
        # load dataset
        if opts.task == 'dirichletSingle':
            pass
        elif opts.task == 'quadratic':
            pass
        elif opts.task == 'dirichletMulti':
            pass
        elif opts.task == 'dirichletUniform':
            pass
        else:
            raise NotImplementedError('task {} not support'.format(opts.task))
        
    def get_train_loader(self):
        r"""
        get the training dataloader for task in opts
        """
        if self.opts.task in set(['dirichletSingle', 'quadratic', 'dirichletMulti','dirichletUniform']):
            return [[0, 0]]


    def get_dataset_info(self):
        r"""
        get the information of dataset for task in opts
        """
        pass