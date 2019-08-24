import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader, Dataset

from read_data import get_data
from visualize import show_samples
from model import Net
from CustomDataset import CustomDataset
from train import train
from predict import predict
from utils import samples, transform
from constants import height, width

device = torch.device("cpu")


# '''
parser = argparse.ArgumentParser(description='Rough Sketch Simplification')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--samples', type=bool, default=False, help='See sample images')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='train the model')
parser.add_argument('--root', default='.', type=str, help='Root Directory for Input and Target images')

opt = parser.parse_args()


if opt.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''

root = opt.root
# root = os.path.join('drive','My Drive','dl_data','rough-sketch-simplification','Data')
Input = os.path.join(root,'Data','Input')
Target = os.path.join(root,'Data','Target')


input_images, target_images = get_data(Input, Target)



if opt.samples:
    show_samples(input_images)
    show_samples(target_images)

if opt.train:

    model = Net()
    model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters())
    criterion = nn.MSELoss().to(device)


    training_set = CustomDataset(input_path=input_images, target_path=target_images, height=height, width=width, transform=transform)
    train_loader = DataLoader(training_set, batch_size=8, shuffle=True)

    import time
    since = time.time()
    epoch_num = 150
    best_val_acc = 0
    total_loss_val, total_acc_val = [],[]
    print ('Training')
    for epoch in range(1, epoch_num+1):
        loss_train, total_loss_train = train(train_loader, model, criterion, optimizer, epoch, device)
        if epoch%20 == 0 or epoch == 2:
            print (f'Epoch: {epoch}')
            samples(input_images[0], target_images[0], model)
            img = predict(model, input_images[0], device)
            pilimg = Image.fromarray(np.uint8(img))
            pilimg.save('pred/' + str(epoch)+'.png')

    print ('Time Taken: ',time.time()-since)
    fig = plt.figure(num = 1)
    fig1 = fig.add_subplot(2,1,1)
    # fig2 = fig.add_subplot(2,1,2)
    fig1.plot(total_loss_train, label = 'training loss')
    # fig2.plot(total_loss_val, label = 'validation loss')
    plt.legend()
    plt.show()


    for i in range (5):
        import random
        k = random.randint(1,64)
        samples(input_images[k], target_images[k], model)
