import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchvision import datasets, transforms

# CIFAR Input: 3 channel 32 x 32 images
class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 32@32*32 (in_channels, out_channels, kernel_size, stride=1)
        bn1 = torch.nn.BatchNorm2d(32)
        conv11 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 32@32*32 (in_channels, out_channels, kernel_size, stride=1)
        bn11 = torch.nn.BatchNorm2d(32)
        pool1 = nn.MaxPool2d(2)     # 32@16*16
        dropout1 = nn.Dropout(0.2)
        conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64@16*16
        bn2 = torch.nn.BatchNorm2d(64)
        conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # 64@16*16
        bn22 = torch.nn.BatchNorm2d(64)
        pool2 = nn.MaxPool2d(2)     # 64@8*8
        dropout2 = nn.Dropout(0.3)
        conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 128@8*8
        bn3 = torch.nn.BatchNorm2d(128)
        conv33 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 128@8*8
        bn33 = torch.nn.BatchNorm2d(128)
        pool3 = nn.MaxPool2d(2)     # 6@4*4
        dropout3 = nn.Dropout(0.5)

        self.conv_module = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            conv11,
            bn11,
            nn.ReLU(),
            pool1,
            dropout1,
            conv2,
            bn2,
            nn.ReLU(),
            conv22,
            bn22,
            nn.ReLU(),
            pool2,
            dropout2,
            conv3,
            bn3,
            nn.ReLU(),
            conv33,
            bn33,
            nn.ReLU(),
            pool3,
            dropout3
        )

        fc1 = nn.Linear(128*4*4, 128)
        fc2 = nn.Linear(128, 84)
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
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trn_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(*stats,inplace=True)])

    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    trn_dataset = datasets.CIFAR10('dataset/cifar10_data/',
                                download=False,
                                train=True,
                                transform=trn_transform)

    val_dataset = datasets.CIFAR10('dataset/cifar10_data/',
                                download=False,
                                train=False,
                                transform=val_transform)

    trn_loader = data_utils.DataLoader(trn_dataset,
                                batch_size=batch_size_trn,
                                num_workers=3,
                                pin_memory=True,
                                shuffle=True)

    val_loader = data_utils.DataLoader(val_dataset,
                                batch_size=batch_size_val,
                                num_workers=3,
                                pin_memory=True,
                                shuffle=True)
    return trn_loader, val_loader
