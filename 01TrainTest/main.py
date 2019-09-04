from run import run
import utils

import sys 
import time 
import torch
import numpy as np

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)

start = time.time()
print('Take whole Volume!', flush=True)
plotter = utils.VisdomLinePlotter(env_name='Plots')
path_to_net = './Network/'
label_dir = './label/'
nii_dir = './numpy/'
run(path_to_net, label_dir, nii_dir, plotter)
print('Whole run took ', time.time()-start, flush=True)
print('Done!', flush=True)
