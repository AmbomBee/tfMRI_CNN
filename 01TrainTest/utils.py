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

def get_train_cv_indices(indices, num_folds, random_state):
    """
    Creates a generator for train_indices and val_indices
    """
    kf = KFold(n_splits=num_folds,random_state=random_state)
    return kf.split(np.zeros(len(indices)), np.zeros(len(indices)))

def get_test_indices(indices, test_split):
    """
    Returns indices for test split
    """
    random.shuffle(indices)
    split = int(np.floor(test_split * len(indices)))
    test_indices, trainset_indices = indices[:split], indices[split:]
    return test_indices, trainset_indices


def get_dataloader(dataset, batch_size, num_GPU):
    """
    Returns the specific dataloader to the batch size
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0*num_GPU)


def save_net(path, batch_size, epoch, cycle_num, train_indices, 
             val_indices, test_indices, net, optimizer, criterion, iter_num=None):
    """
    Saves the networks specific components and the network itselfs 
    to a given path
    """
    if iter_num is not None:
        filename = path + 'cv_' + str(cycle_num) + '_iterin_' + str(epoch) + '_net.pt'
        #filename = path + 'cv_' + str(cycle_num) + 'net.pt'
        torch.save({
                'iter_num': iter_num,
                'batch_size': batch_size,
                'epoch': epoch,
                'cycle_num': cycle_num,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'net' : net
                }, filename)
        print('Network saved to ' + filename, flush=True)
    else:
        filename = path + 'cv_end_' + str(cycle_num) + 'net.pt'
        torch.save({
                'batch_size': batch_size,
                'epoch': epoch,
                'cycle_num': cycle_num,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'net' : net
                }, filename)
        print('Network saved to ' + filename, flush=True)
        
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
