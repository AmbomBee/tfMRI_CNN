import numpy as np

class averageLossMeter(object):
    """
        Computes and stores the average and current value
        Adapted from metrics written by meetshah1995
        https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.loss_hist = []

    def update(self, loss, n=1):
        self.loss = loss
        self.sum += loss * n
        self.count += n
        self.avg = self.sum / self.count
        self.loss_hist.append(self.avg)
    
    def get_history(self):
        return {'loss': self.loss_hist}


class BinaryClassificationMeter(object):
    """
        Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp_hist = []
        self.tn_hist = []
        self.fp_hist = []
        self.fn_hist = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.bacc = 0
        self.pre = 0
        self.tnr = 0
        self.rec = 0
        self.f1 = 0
        self.avg_pre = 0
        self.avg_tnr = 0
        self.avg_rec = 0
        self.avg_f1 = 0

    def update(self, prediction, target):
        pred = prediction >= 0.5
        truth = target >= 0.5
        self.tp += np.multiply(pred, truth).sum(0)
        self.tn += np.multiply((1 - pred), (1 - truth)).sum(0)
        self.fp += np.multiply(pred, (1 - truth)).sum(0)
        self.fn += np.multiply((1 - pred), truth).sum(0)
        self.tp_hist.append(self.tp)
        self.tn_hist.append(self.tn)
        self.fp_hist.append(self.fp)
        self.fn_hist.append(self.fn)
        self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
        self.bacc = (self.tp.sum() / (self.tp + self.fn).sum() + self.tn.sum() / (self.tn + self.fp).sum()) * 0.5
        self.pre = self.tp / (self.tp + self.fp)
        self.tnr = self.tn / (self.tn + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
        self.avg_pre = np.nanmean(self.pre)
        self.avg_tnr = np.nanmean(self.tnr)
        self.avg_rec = np.nanmean(self.rec)
        self.avg_f1 = np.nanmean(self.f1)
        
    def get_scores(self):
        return {'Acc: ': self.acc,
                'Balanced acc: ': self.bacc,
                'Avg Precision: ': self.avg_pre,
                'Avg TNR': self.avg_tnr,
                'Avg Recall: ': self.avg_rec,
                'Avg F1: ': self.avg_f1
                }
                
    def get_history(self):
        return {'tp': self.tp_hist,
                'tn': self.tn_hist,
                'fp': self.fp_hist, 
                'fn': self.fn_hist        
                }

