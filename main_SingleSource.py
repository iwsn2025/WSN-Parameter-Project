#!/usr/bin/env python
# encoding: utf-8

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
import csv

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'

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
        return y, x3, y2    #softmax(output) and mmd and softmax(output/T)

torch.manual_seed(1)


def exp(shot=5,shuffle=False,epoch=500): # TODO: change epoch up to 1000


    dataset = pd.read_csv("simulation.csv", sep=',')
    X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], axis=1))
    X = np.float32(X)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = np.array(dataset['LA'])
    y = np.float32(y)
    Sy = torch.from_numpy(y).type(torch.LongTensor).to(DEVICE)
    Sx = torch.from_numpy(X).to(DEVICE)
    # print(Sx.shape, Sy.shape)

    dataset2 = pd.read_csv("sample.csv", sep=',')
    X2 = np.array(dataset2.drop(['A', 'C', 'P', 'LA'], axis=1))
    X2 = np.float32(X2)
    X2 = scaler.transform(X2)
    y2 = np.array(dataset2['LA'])
    y2 = np.float32(y2)
    Tx = torch.from_numpy(X2).to(DEVICE)
    Ty = torch.from_numpy(y2).type(torch.LongTensor).to(DEVICE)
    # print(Tx.shape, Ty.shape)

    dataset3 = pd.read_csv("realworlddata.csv", sep=',')
    X3 = np.array(dataset3.drop(['A', 'C', 'P', 'LA'], axis=1))
    X3 = np.float32(X3)
    X3 = scaler.transform(X3)
    y3 = np.array(dataset3['LA'])
    y3 = np.float32(y3)
    X_train1, X_test, y_train1, y_test = train_test_split(X3, y3, test_size = 0.4, random_state=42)
    Tx_test = torch.from_numpy(X_test).to(DEVICE)
    Ty_test = torch.from_numpy(y_test).type(torch.LongTensor).to(DEVICE)

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

    # print(Tx_test.shape, Ty_test.shape)

    #Teacher model
    teacher = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    teacher = teacher.to(DEVICE)
    optimizer = optim.SGD(
        teacher.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=L2_WEIGHT
    )
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
        a = np.argmax(y_pred.detach().cpu().numpy(), axis = 1)
        b = Sy.detach().cpu().numpy()
        for it, it2 in zip(a, b):
            if it == it2:
                count = count + 1
        # print(count/y_pred.shape[0])



    #Student model
    student = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    student = student.to(DEVICE)
    student.load_state_dict(teacher.state_dict())
    #optimizer2 = optim.Adam(student.parameters(), lr=0.1)
    optimizer2 = optim.SGD(
        student.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=L2_WEIGHT
    )
    criterion2 = nn.CrossEntropyLoss()


    max_val_accuracy=0
    max_test_accuracy=0

    for e in range(1, epoch + 1):
        temp1 = 10
        temp2 = 88*shot
        for i in range(temp1):
            data = Sx[i*temp2:i*temp2+temp2]
            target = Sy[i*temp2:i*temp2+temp2]

            x_tar = Tx[0:temp2]
            y_target = Ty[0:temp2]

            y_src, x_src_mmd, raw_y1 = teacher(data)
            y_tar, x_tar_mmd, raw_y2 = student(data)

            tt = np.argmax(raw_y1.detach().cpu().numpy(), axis = 1)
            tt = torch.from_numpy(tt).to(DEVICE)
            loss_high_t = criterion(raw_y2, tt)
            #print(raw_y1.shape, raw_y2.shape, target.shape)
            #print(tt.shape, tt.dtype)


            #loss_c1 = criterion(y_src, target)
            #loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd) + loss_high_t
            #loss = loss_c1 + LAMBDA * loss_mmd
            #optimizer.zero_grad()
            #optimizer2.zero_grad()
            loss_high_t.backward()
            #loss_mmd.backward()
            #optimizer.step()
            #optimizer2.zero_grad()
            #loss_mmd.backward()
            optimizer2.step()

            y_src, x_src_mmd, y_src_soft = teacher(x_tar)
            y_tar, x_tar_mmd, y_tar_soft = student(x_tar)
            loss_mmd = LAMBDA * mmd_loss(x_src_mmd, x_tar_mmd)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss_mmd.backward()
            optimizer.step()
            optimizer2.step()

            #optimizer2.zero_grad()
            #loss_soft = criterion(y_tar_soft, y_src_soft)
            #loss_soft.backward()
            #optimizer2.step()

            optimizer2.zero_grad()
            y_tar, _, _ = student(x_tar)
            loss_c2 = criterion(y_tar, y_target)
            loss_c2.backward()
            optimizer2.step()

        with torch.no_grad():
            y_pred, _, _ = student.forward(Tx_val)
            count = 0
            a = np.argmax(y_pred.detach().cpu().numpy(), axis = 1)
            b = Ty_val.detach().cpu().numpy()
            for it, it2 in zip(a, b):
                if it == it2:
                    count = count + 1
            # print(count/y_pred.shape[0])
            if count/y_pred.shape[0]>max_val_accuracy:
                max_val_accuracy = count/y_pred.shape[0]

                #when val update, test update
                y_pred, _, _ = student.forward(Tx_test)
                count = 0
                a = np.argmax(y_pred.detach().cpu().numpy(), axis = 1)
                b = Ty_test.detach().cpu().numpy()
                for it, it2 in zip(a, b):
                    if it == it2:
                        count = count + 1
                # print(count/y_pred.shape[0])
                if count/y_pred.shape[0]>max_test_accuracy:
                    max_test_accuracy = count/y_pred.shape[0]

    # print("One Simulator Accuracy on Validation Dataset:",max_val_accuracy)
    print("Accuracy on testing data:",max_test_accuracy)

    return max_val_accuracy,max_test_accuracy

val_accs=[]
test_accs=[]

for i in range(1,6):
    print("------------------Experiment with %d shot(s) of data on one simulator (TOSSIM)--------------------------"%(i))
    print("Training data: simulation data and %d shot(s) of physical data"%(i))
    print("Testing data: physical data")
    val_acc,test_acc = exp(shot=i,shuffle=False,epoch=400)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

# Save to CSV file (overwrite every time)
csv_file = "csvs/single_source_accuracy.csv" #TODO: fix single source values to replicate figure in paper

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Shots", "Single Source Accuracy"])
    for i, acc in enumerate(test_accs, start=1):
        writer.writerow([i, acc])

print(f"Single source accuracy data saved to {csv_file}")

# Create the plot
plt.figure(figsize=(10, 6))
# plt.plot(val_accs, label='Validation Accuracy', marker='o', linestyle='-')
plt.plot(range(1,6),test_accs, label='Accuracy', marker='s', linestyle='--')

plt.xticks(range(1,6))

# Add title and labels
plt.title('One simulator: TOSSIM', fontsize=16)
plt.xlabel('Number of shots of physical data', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0,1)

# Add legend
# plt.legend(loc='best', fontsize=12)

# Add grid
plt.grid(True)

# Display the plot
plt.show()

