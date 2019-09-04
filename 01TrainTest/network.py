import torch 
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Net(nn.Module):
    """
    The network architecture
    """
    def __init__(self, in_ch=8, out_ch=2):
        super(Net, self).__init__()
        self.Conv1 = conv_block(ch_in=in_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=512)
        self.Conv_last = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6144, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_ch)

    def forward(self, x):
        x = self.Conv1(x)
        #print('Conv1 ', x.size())
        #Conv1  torch.Size([32, 32, 27, 33, 27])
        x = self.Conv2(x)
        #print('Conv2 ', x.size())
        #Conv2  torch.Size([32, 64, 14, 17, 14])
        x = self.Conv3(x)
        #print('Conv3 ', x.size())
        #Conv3  torch.Size([32, 128, 7, 9, 7])
        x = self.Conv4(x)
        #print('Conv4', x.size())
        #Conv4 torch.Size([32, 256, 4, 5, 4])
        x = self.Conv5(x)
        #print('Conv5', x.size())
        #Conv5 torch.Size([32, 512, 2, 3, 2])
        x = F.relu(self.Conv_last(x))
        #print('Conv_last ', x.size())
        #Conv_last  torch.Size([32, 512, 2, 3, 2])
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
