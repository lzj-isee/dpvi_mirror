import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from basic import basic
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import scipy.io as scio

class myDataLoader(basic):
    def __init__(self, opts) -> None:
        super().__init__(opts)
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
            raise ValueError('task {} not support'.format(opts.task))
        # set dataloader
        if opts.task in ['dirichletSingle', 'quadratic', 'dirichletMulti','dirichletUniform']:
            self.train_loader = [[0, 0]]