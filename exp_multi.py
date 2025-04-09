#!/usr/bin/env python
# encoding: utf-8

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mmd
from variable import *
import random
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DEVICE = "cpu"

def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])

class DaNN(nn.Module):
    def __init__(self, n_input=3, n_hidden1=256, n_hidden2=256, n_class=88):
        super(DaNN, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden1)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden1, n_hidden2)
        self.layer_hidden2 = nn.Linear(n_hidden2, n_class)

    def forward(self, x):
        x1 = self.layer_input(x)
        x2 = self.relu(x1)
        x3 = self.layer_hidden(x2)
        x4 = self.relu(x3)
        y1 = self.layer_hidden2(x4)
        y = F.log_softmax(y1)
        y2 = F.log_softmax(y1/10)
        return y, x3, y2


def exp(shot=5,shuffle=False,epoch=500):
    torch.manual_seed(1)
    Random_State =0
    Step = 500
    Rate = 0.8
    Base1 = 88 * shot
    Base2 = 88 * 38
    Base3 = 88 * 75
    Item = Base2
    Learning = 0.12
    dataset = pd.read_csv("simulation6.csv", sep=',')
    X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], axis=1))
    X = np.float32(X)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = np.array(dataset['LA'])
    y = np.float32(y)
    Sy = torch.from_numpy(y).type(torch.LongTensor)
    Sx = torch.from_numpy(X)
    #print(Sx.shape, Sy.shape)
    Sx = Sx[0:Base2].to(DEVICE)
    Sy = Sy[0:Base2].to(DEVICE)
    #print(Sx.shape, Sy.shape)

    dataset2 = pd.read_csv("sample.csv", sep=',')
    X2 = np.array(dataset2.drop(['A', 'C', 'P', 'LA'], axis=1))
    X2 = np.float32(X2)
    X2 = scaler.transform(X2)
    y2 = np.array(dataset2['LA'])
    y2 = np.float32(y2)
    Tx = torch.from_numpy(X2).to(DEVICE)
    Ty = torch.from_numpy(y2).type(torch.LongTensor).to(DEVICE)
    #print(Tx.shape, Ty.shape)

    dataset3 = pd.read_csv("realworlddata.csv", sep=',')
    X3 = np.array(dataset3.drop(['A', 'C', 'P', 'LA'], axis=1))
    X3 = np.float32(X3)
    X3 = scaler.transform(X3)
    y3 = np.array(dataset3['LA'])
    y3 = np.float32(y3)
    TA = torch.from_numpy(X3)
    X_train1, X_test, y_train1, y_test = train_test_split(X3, y3, test_size = 0.4, random_state=Random_State)
    Tx_test = torch.from_numpy(X_test)
    Ty_test = torch.from_numpy(y_test).type(torch.LongTensor)
    Tx_train = torch.from_numpy(X_train1)
    Ty_train = torch.from_numpy(y_train1).type(torch.LongTensor)
    #print(Tx_test.shape, Ty_test.shape)
    #print(Tx_train.shape, Ty_train.shape)


    if shuffle:

        # Create an array of indices and shuffle them
        ids = np.arange(0, len(Tx_test))
        random.shuffle(ids)

        # Split the indices into validation and test sets
        id_val = ids[:int(len(Tx_test) / 2)]
        id_test = ids[int(len(Tx_test) / 2):]

        Tx_val = Tx_test[id_val].to(DEVICE)
        Ty_val = Ty_test[id_val].to(DEVICE)

        Tx_test = Tx_test[id_test].to(DEVICE)
        Ty_test = Ty_test[id_test].to(DEVICE)

    else:
        Tx_val = Tx_test[:int(len(Tx_test)/2)].to(DEVICE)
        Ty_val = Ty_test[:int(len(Ty_test)/2)].to(DEVICE)

        Tx_test = Tx_test[int(len(Tx_test)/2):].to(DEVICE)
        Ty_test = Ty_test[int(len(Ty_test)/2):].to(DEVICE)


    dataset4 = pd.read_csv("simulation10.csv", sep=',')
    X4 = np.array(dataset4.drop(['A', 'C', 'P', 'LA'], axis=1))
    X4 = np.float32(X4)
    scaler4 = StandardScaler().fit(X4)
    X4 = scaler4.transform(X4)
    Y4 = np.array(dataset4['LA'])
    Y4 = np.float32(Y4)
    SY4 = torch.from_numpy(Y4).type(torch.LongTensor)
    SX4 = torch.from_numpy(X4)
    #print(SX4.shape, SY4.shape)
    SX4 = SX4[0:Item].to(DEVICE)
    SY4 = SY4[0:Item].to(DEVICE)
    #print(SX4.shape, SY4.shape)


    #Teacher model
    teacher = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    teacher = teacher.to(DEVICE)
    optimizer = optim.Adam(teacher.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for e in range(1, 0 + 1):
        teacher.train()
        y_pred, _, _ = teacher.forward(Sx)
        loss_c = criterion(y_pred, Sy)
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()

        count = 0
        a = np.argmax(y_pred.detach().numpy(), axis = 1)
        b = Sy.detach().numpy()
        for it, it2 in zip(a, b):
            if it == it2:
                count = count + 1
        print(e, count/y_pred.shape[0])

    start = time.perf_counter()

    #New model
    student = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    student = student.to(DEVICE)
    #student.load_state_dict(teacher.state_dict())
    optimizer2 = optim.Adam(student.parameters(), lr=Learning)

    criterion2 = nn.CrossEntropyLoss()
    x_tar = Tx[0:Base1]
    y_target = Ty[0:Base1]
    max_accuracy=0

    max_val_accuracy=0
    max_test_accuracy=0
    max_iteration=0
    max1_accuracy=0
    max1_iteration=0
    i = 0

    for e in range(1, epoch + 1):
        student.train()

        y_pred, _, _ = student.forward(SX4)
        loss_mmd2 = 0.95 * criterion2(y_pred, SY4)

        y_pred, _, _ = student.forward(Sx)
        loss_mmd1 = 0.05 * criterion2(y_pred, Sy)

    #    loss_mmd = loss_mmd2
    #    loss_mmd = loss_mmd2 + loss_mmd1 + loss_mmd3
        loss_mmd = loss_mmd2 + loss_mmd1
        optimizer2.zero_grad()
        loss_mmd.backward()
        optimizer2.step()

        y_pred, _, _ = student.forward(x_tar)
        loss_c2 = criterion2(y_pred, y_target)
        optimizer2.zero_grad()
       # loss_total.backward()
        loss_c2.backward()
        optimizer2.step()


        # with torch.no_grad():
        #     y_pred, _, _ = student.forward(Tx_test)
        #     count = 0
        #     a = np.argmax(y_pred.detach().numpy(), axis = 1)
        #     b = Ty_test.detach().numpy()
        #     for it, it2 in zip(a, b):
        #         if it == it2:
        #             count = count + 1
        #     if (count/y_pred.shape[0]) > max_accuracy:
        #         max_time = time.perf_counter()
        #         max_accuracy = (count/y_pred.shape[0])
        #         max_iteration = e
        #     #print(e, count/y_pred.shape[0])
        #     #if e%10 == 0:
        #     #    print(count/y_pred.shape[0])
        #     #     print("Max till now", max_iteration, max_accuracy)
        #     #     max_accuracy=0
        #     #     max_iteration=0
        #
        #     if e == 1000:
        #         print("Max under 1000", max_iteration, max_accuracy)
        #     if e == 2000:
        #         print("Max under 2000", max_iteration, max_accuracy)
        #


        with torch.no_grad():
            y_pred, _, _ = student.forward(Tx_val)
            count = 0
            a = np.argmax(y_pred.detach().cpu().numpy(), axis = 1)
            b = Ty_val.detach().cpu().numpy()
            for it, it2 in zip(a, b):
                if it == it2:
                    count = count + 1
            if (count/y_pred.shape[0]) > max_val_accuracy:
                max_time = time.perf_counter()
                max_val_accuracy = (count/y_pred.shape[0])
                max_iteration = e

                #when val update, test acc update
                y_pred, _, _ = student.forward(Tx_test)
                count = 0
                a = np.argmax(y_pred.detach().cpu().numpy(), axis = 1)
                b = Ty_test.detach().cpu().numpy()
                for it, it2 in zip(a, b):
                    if it == it2:
                        count = count + 1
                max_test_accuracy = count/y_pred.shape[0]


            #print(e, count/y_pred.shape[0])
            #if e%10 == 0:
            #    print(count/y_pred.shape[0])
            #     print("Max till now", max_iteration, max_accuracy)
            #     max_accuracy=0
            #     max_iteration=0

            if e == 1000:
                print("Max under 1000", max_iteration, max_accuracy)
            if e == 2000:
                print("Max under 2000", max_iteration, max_accuracy)


    end = time.perf_counter()
    #print("Run time:", end - start, "Max time:", max_time - start)
    # print("Learning ", Learning, "RS ", Random_State, "Step ", Step, "Gamma ", Rate, "Target data ", Base1)
    # print("Two simulators Accuracy on Validation Dataset:", max_val_accuracy)
    print("Accuracy on testing dataset:", max_test_accuracy)

    return max_val_accuracy,max_test_accuracy