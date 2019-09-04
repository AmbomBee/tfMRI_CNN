import utils
import metrics

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import csv

def test(net, test_loader, device):
    """
    Applies testing on the network
    """
    net.to(device)
    net.eval()
    test_acc = 0
    # Setup metrics
    running_metrics_test = metrics.BinaryClassificationMeter()
    pred_dict = {'pred': [], 'label': [], 'score': [], 'output': []}
    for data in tqdm(test_loader, desc='Testdata'):
        # get the inputs
        inputs = data['fdata'].to(device)
        labels = data['label'].to(device)
        outputs = net(inputs)
        score, prediction = torch.max(outputs.data, 1)
        running_metrics_test.update(prediction.cpu().numpy(), labels.cpu().numpy())
        pred_dict['label'].append(labels.cpu().numpy())
        pred_dict['pred'].append(prediction.cpu().numpy())
        pred_dict['score'].append(score.cpu().numpy())
        pred_dict['output'].append(outputs.cpu().detach().numpy())
    return running_metrics_test, pred_dict, running_metrics_test.get_history()
