import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

from torch.utils.data import DataLoader

def plot_map(sumed_heatmap, brain, name):
    sumed_heatmap /= np.max(sumed_heatmap)
    extent = 0, brain.shape[0], 0, brain.shape[1]
    for i in range(sumed_heatmap.shape[2]):
        plt.subplot(3,3,i+1)
        plt.imshow(brain[:,:,22*i+11], cmap='gray', extent=extent)
        plt.imshow(sumed_heatmap.squeeze()[i], alpha=0.5, extent=extent)

        plt.xlabel(f'slice {i}')
        plt.title(name)
    plt.show()
    
class BinaryClassificationMeter(object):
    """
        Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.bacc = 0

    def update(self, prediction, target):
        pred = prediction >= 0.5
        truth = target >= 0.5
        self.tp += np.multiply(pred, truth).sum(0)
        self.tn += np.multiply((1 - pred), (1 - truth)).sum(0)
        self.fp += np.multiply(pred, (1 - truth)).sum(0)
        self.fn += np.multiply((1 - pred), truth).sum(0)

        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.bacc = (self.tp.sum() / (self.tp + self.fn).sum() + self.tn.sum() / (self.tn + self.fp).sum()) * 0.5

    def get_scores(self):
        return {'Acc: ': self.acc,
                'Balanced acc: ': self.bacc}


def get_dataloader(dataset, batch_size, num_GPU):
    """
    Returns the specific dataloader to the batch size
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0*num_GPU)
