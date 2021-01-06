import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split



def train_model(train_dataloader, test_dataloader, optimizer, loss, net, device, NUM_EPOCH=5, scheduler=None):
    net = net.to(device)
    net.train()
    # spectra_weight = np.array([1, 100])
    # spectra_weight = torch.from_numpy(spectra_weight).to(device).float()

    if scheduler != None:
        print('*** WILL USE SCHEDULER ***')

    for i in range(NUM_EPOCH):
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            X, y = data
            # print(X.shape)
            # print(y.shape)

            X = X.to(device)
            y = y.to(device)

            # if idx==0:
            #     print(X.shape, y.shape)

            y_pred = net(X)

            optimizer.zero_grad()
            loss_train = loss(y_pred, y)
            # loss_train = loss(y_pred, y, spectra_weight)
            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

            # if (idx+1)%5==0:
            #     print('EPOCH '+str(i+1)+'/'+str(NUM_EPOCH)+' || '+'STEP '+str(idx+1)+'/'+str(len(train_dataloader))+' || '+'LOSS: '+str(running_loss/(idx+1)))
            #     print('===================================================')
        print('----------------------------------------------------------------------')
        print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS ' + str(running_loss / len(train_dataloader)))

        # test model for each epoch
        test_model(test_dataloader, loss, net, device)



def test_model(test_dataloader, loss, net, device):
    net = net.to(device)
    net.eval()
    spectra_weight = np.array([1,100])
    spectra_weight = torch.from_numpy(spectra_weight).to(device).float()

    running_loss = 0.0
    for idx, data in enumerate(test_dataloader):
        X, y = data

        X = X.to(device)
        y = y.to(device)

        y_pred = net(X)

        loss_train = loss(y_pred, y)
        # loss_train = loss(y_pred, y, spectra_weight)
        running_loss += loss_train.item()

    print('### TEST LOSS ', str(running_loss/len(test_dataloader)))
