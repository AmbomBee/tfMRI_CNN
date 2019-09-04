from test import test
import dataset
import utils
import network

import torch
import numpy as np  
import time
from tqdm.auto import tqdm

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)

start = time.time()

path_to_net = './net.pt'
label_dir = './label/'
nii_dir = './numpy/'

checkpoint = torch.load(path_to_net)
net = checkpoint['net']
test_indices = checkpoint['test_indices']
batch_size = checkpoint['batch_size']
random_state=666

device = torch.device('cuda:0' 
                      if torch.cuda.is_available() else 'cpu')
num_GPU = torch.cuda.device_count()
fMRI_dataset_test = dataset.fMRIDataset(label_dir, nii_dir, 
                            test_indices, transform=dataset.ToTensor())
test_length = len(fMRI_dataset_test)
test_loader = utils.get_dataloader(fMRI_dataset_test, 
                                     batch_size, num_GPU)

metrics, pred_dict, history = test(net, test_loader, device)
torch.save(history, './history.pt')
torch.save(pred_dict, './pred_dict.pt')
torch.save(metrics.get_scores(), './test_scores.pt')
score = metrics.get_scores()
for k, v in score.items():
    print(k, v, flush=True)
print('Whole run took ', time.time()-start, flush=True)
print('Done!', flush=True)
