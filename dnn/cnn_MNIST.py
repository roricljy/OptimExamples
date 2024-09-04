import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchvision import datasets, transforms

# MNIST Input: 1 channel 28 x 28 images
class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        conv1 = nn.Conv2d(1, 20, kernel_size=5)  # 6@24*24 (in_channels, out_channels, kernel_size, stride=1)
        pool1 = nn.MaxPool2d(2)     # 6@12*12
        dropout1 = nn.Dropout(0.2)
        conv2 = nn.Conv2d(20, 40, kernel_size=5) # 16@8*8
        pool2 = nn.MaxPool2d(2)     # 16@4*4
        dropout2 = nn.Dropout(0.5)

        self.conv_module = nn.Sequential(
            conv1,
            pool1,
            nn.ReLU(),
            dropout1,
            conv2,
            pool2,
            nn.ReLU(),
            dropout2
        )

        fc1 = nn.Linear(40*4*4, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

    def forward(self, x):
        out = self.conv_module(x)   # 16@4*4
        # make linear
        dim = 1
        for d in out.size()[1:]:    # 16,4,4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        result = F.softmax(out, dim=1)
        return out

def load_data(batch_size_trn, batch_size_val):
    trn_dataset = datasets.MNIST('dataset/mnist_data/',
                                download=True,
                                train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),  # image to Tensor
                                    transforms.Normalize((0.1307,), (0.3081,))   # image, label
                                ]))

    val_dataset = datasets.MNIST('dataset/mnist_data/',
                                download=True,
                                train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),  # image to Tensor
                                    transforms.Normalize((0.1307,), (0.3081,))   # image, label
                                ]))

    trn_loader = data_utils.DataLoader(trn_dataset,
                                batch_size=batch_size_trn,
                                shuffle=True)

    val_loader = data_utils.DataLoader(val_dataset,
                                batch_size=batch_size_val,
                                shuffle=True)

    return trn_loader, val_loader
