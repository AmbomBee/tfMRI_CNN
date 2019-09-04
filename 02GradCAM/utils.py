import torch
from sklearn.model_selection import KFold
import numpy as np
from visdom import Visdom
import matplotlib.pyplot as plt
import random
import csv

from torch.utils.data import DataLoader

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='fMRI'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, x_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_name,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

def get_dataloader(dataset, batch_size, num_GPU):
    """
    Returns the specific dataloader to the batch size
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0*num_GPU)
        
def save_scores(scores, phase, cv_num):
    with open(phase + '_cv' + str(cv_num) + '_scores.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(scores.keys())
        writer.writerows(zip(*scores.values()))
        
def save_loss(loss, phase, cv_num):
    with open(phase + '_cv' + str(cv_num) + '_loss.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(loss.keys())
        writer.writerows(zip(*loss.values()))
