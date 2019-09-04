import torch
from tqdm.auto import tqdm
import nibabel as nib
from network import Net
from dataset import *
from utils import *

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


