import torch
from tqdm.auto import tqdm
import nibabel as nib
from network import Net
from dataset import *
from utils import *

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

path_to_net = './network.pt'
label_dir = './label/'
nii_dir = './numpy/'
checkpoints = torch.load(path_to_net)
net = Net()
net.load_state_dict(checkpoints['net_state_dict'])

test_idc = checkpoints['test_indices']
batch_size = 1
device = torch.device('cuda:0' 
                    if torch.cuda.is_available() else 'cpu')
num_GPU = torch.cuda.device_count()
print('Device: ', device, flush=True)
if  num_GPU > 1:
    print('Let us use', num_GPU, 'GPUs!', flush=True)
    net = nn.DataParallel(net)
net.to(device)

dataset = fMRIDataset(label_dir, nii_dir, test_idc, transform=ToTensor())
dataloader = get_dataloader(dataset, batch_size, num_GPU)
net.eval()

sumed_heatmap_B_b = []
sumed_heatmap_Z_b = []
sumed_heatmap_B_r = []
sumed_heatmap_B_n = []
sumed_heatmap_Z_n = []
sumed_heatmap_Z_r = []

acc_hist = BinaryClassificationMeter()
# get the image from the dataloader
for i, data in tqdm(enumerate(dataloader), desc='Dataiteration_Test'):
    img = data['fdata'].to(device)
    label = data['label'].to(device)
    label_ID = data['label_ID']
    # get the most likely prediction of the model
    pred = net(img)
    pred_label = net(img).argmax(dim=1)
    acc_hist.update(pred_label.cpu().numpy(), label.cpu().numpy())
    pred = pred[:, 1]
    # back-propagation with the logit of the 2nd class which represents a success
    # get the gradient of the output with respect to the parameters of the model
    pred.backward()
    # pull the gradients out of the model
    gradients = net.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=(0, 2, 3, 4))
    # get the activations of the last convolutional layer
    activations = net.get_activations(img).detach()
    # weight the channels by corresponding gradients
    for i in range(128):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    # save the heatmap
    if label_ID[0].find('B_b') != -1:
        sumed_heatmap_B_b.append(heatmap)
    elif label_ID[0].find('Z_b') != -1:
        sumed_heatmap_Z_b.append(heatmap)
    elif label_ID[0].find('B_n') != -1:
        sumed_heatmap_B_n.append(heatmap)
    elif label_ID[0].find('B_r') != -1:
        sumed_heatmap_B_r.append(heatmap)
    elif label_ID[0].find('Z_n') != -1:
        sumed_heatmap_Z_n.append(heatmap)
    elif label_ID[0].find('Z_r') != -1:
        sumed_heatmap_Z_r.append(heatmap)

for k,v in acc_hist.get_scores().items():
    print(k,v, flush=True)

torch.save(sumed_heatmap_B_b, 'sumed_heatmap_B_b.pt')
torch.save(sumed_heatmap_Z_b, 'sumed_heatmap_Z_b.pt')
torch.save(sumed_heatmap_B_n, 'sumed_heatmap_B_n.pt')
torch.save(sumed_heatmap_B_r, 'sumed_heatmap_B_r.pt')
torch.save(sumed_heatmap_Z_n, 'sumed_heatmap_Z_n.pt')
torch.save(sumed_heatmap_Z_r, 'sumed_heatmap_Z_r.pt')


