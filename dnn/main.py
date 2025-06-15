import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# load my optimizers & loss functions
from user_optim.lion import *
from user_optim.line_search import *
from user_loss.focal_loss import *

#random_seed = 1
#torch.manual_seed(random_seed)
#torch.backends.cudnn.enabled = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# network model
import cnn_MNIST as net; dataset_name='mnist_cnn'; arch='cnn'; ncls=10
#import cnn_CIFAR10 as net; dataset_name='cifar10_cnn'; arch='cnn'; ncls=10

model = net.TestCNN()
if use_cuda:
    model.to(device)

load_model = False         # continue learning from the saved checkpoint
save_model = False         # save model at each epoch
save_best = True            # save best model
show_test_result = True

num_epochs = 2
batch_size_trn = 64
batch_size_val = 1000
log_frequency = 10

# loss functions
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()              # least squares loss
#criterion = nn.HuberLoss()            # robust to outliers (M-estimator)
#criterion = nn.SmoothL1Loss()         # robust to outliers (M-estimator)
#criterion = nn.TripletMarginLoss()    # triplet margin loss
#criterion = FocalLoss(gamma=3.0)       # my loss function

# optimizers
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, dampening=0, weight_decay=1e-5, nesterov=True); opt_name = f'sgd_b{batch_size_trn}'
#optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99); opt_name = f'rmsprop_b{batch_size_trn}'
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5); opt_name = f'adam_b{batch_size_trn}'
#optimizer = Lion(model.parameters(), lr=0.0002, beta1=0.9, beta2=0.99, gamma=0.1); opt_name = f'lion_b{batch_size_trn}'
#optimizer = LineSearch(model.parameters(), max_step_size=0.5); opt_name = f'ls_b{batch_size_trn}'   # my optimizer

# learning rate schedular
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=0.00001)

# logging
experiment_name = f'{dataset_name}_{opt_name}'
log_path = f'{experiment_name}.txt'
pth_path = f'{experiment_name}.pth'
pth_best_path = f'{experiment_name}_best.pth'
if save_model:
    if load_model:
        log = open(log_path, 'a')
    else:
        log = open(log_path, 'w')

# network model
last_epoch = 0
best_accuracy = 0
if load_model:
    checkpoint = torch.load(pth_path)
    last_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
if use_cuda:
    model.to(device)

trn_loader,val_loader = net.load_data(batch_size_trn=batch_size_trn, batch_size_val=batch_size_val)
num_batches = len(trn_loader)
log_interval = round(num_batches/log_frequency)

def evaluate(model, eval_loader):
    loss = 0.0
    correct = 0
    with torch.no_grad():   # very very very very important!!!
        for j, val in enumerate(eval_loader):
            val_x, val_label = val
            if use_cuda:
                val_x = val_x.cuda()
                val_label = val_label.cuda()
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_label)
            loss += val_loss
            pred_label = val_pred.data.max(1, keepdim=False)[1]
            correct += pred_label.eq(val_label).sum()

    loss_avg = loss/len(eval_loader)
    accuracy = correct/len(eval_loader.dataset) * 100
    return loss_avg, accuracy  

# train
for epoch in range(num_epochs):
    trn_loss = 0.0
    current_epoch = epoch + last_epoch
    print(experiment_name)
    for batch_idx, data in enumerate(trn_loader):
        def closure():
            x, label = data
            if use_cuda:
                x = x.cuda()
                label = label.cuda()     
            optimizer.zero_grad() # step and zero_grad are always paired
            pred = model(x)
            loss = criterion(pred, label)
            loss.backward()
            del pred        # memory issue
            return loss

        loss = optimizer.step(closure)

        # trn loss summary
        trn_loss += loss.item()
        del loss        # memory issue

        # show stat
        if batch_idx % log_interval == log_interval - 1:
            model.eval()
            val_loss, accuracy = evaluate(model, val_loader)
            model.train()
            print("epoch: {}/{} | step: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f} | accuracy: {:.2f}".format(
                current_epoch+1, num_epochs, (batch_idx+1), num_batches, trn_loss/log_interval, val_loss, accuracy
                ))

            trn_loss_avg = trn_loss/log_interval
            if save_model:
                train_log = f'{current_epoch},{batch_idx},{num_batches},{trn_loss_avg:0.5f},{val_loss:0.5f},{accuracy:0.2f}'
                log.write(train_log + '\n')

            if save_best and accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': current_epoch,
                    'best_accuracy': best_accuracy,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, pth_best_path)

            trn_loss = 0.0

    scheduler.step()

    # save epoch result    
    if save_model:
        torch.save({
            'epoch': current_epoch,
            'best_accuracy': best_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, pth_path)

print("finish trainnig!")

if show_test_result:
    examples = enumerate(val_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        if use_cuda:
            example_data = example_data.cuda()            
        output = model(example_data)
    if use_cuda:
        example_data = example_data.cpu()
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    fig    
    plt.show()

if __name__ == "__main__":
    print(dataset_name)
