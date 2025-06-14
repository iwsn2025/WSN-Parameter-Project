#!/usr/bin/env python
# encoding: utf-8
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import mmd
from variable import *
import random

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


def exp(shot=5,shuffle=False,epoch=500,sim=True):


    dataset = pd.read_csv(simulation, sep=',')
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

    dataset3 = pd.read_csv(realworlddata, sep=',')
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
    print("----Teacher model training with simulation data")

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

    if sim:

        for e in range(1, 100 + 1):
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

    print("----Student model initialization ")


    #Student model
    student = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    student = student.to(DEVICE)
    print("----Student model loading weights from teacher model ")

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
    print("----Student model domain adaptation with simulation and physical data ")

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
    # print("One Simulator Accuracy on Testing Dataset:",max_test_accuracy)


    #test over time

    test_accs = []
    # for test_data_id in range(11,17):
    #     dataset3 = pd.read_csv("test%d.csv"%(test_data_id), sep=',')
    # for test_data_id in range(11,17):
    i=0
    for test_data_id in [12,13,16,14,15,11]:
        i+=1
        print("-------------------------Testing on physical data collected after %d days-------------------------"%(np.power(2,i+1)))

        if test_data_id<11:
            dataset3 = pd.read_csv("data%d.csv"%(test_data_id), sep=',')
        else:
            dataset3 = pd.read_csv("test%d.csv"%(test_data_id), sep=',')

        X3 = np.array(dataset3.drop(['A', 'C', 'P', 'LA'], axis=1))
        X3 = np.float32(X3)
        X3 = scaler.transform(X3)
        y3 = np.array(dataset3['LA'])
        y3 = np.float32(y3)
        X_train1, X_test, y_train1, y_test = train_test_split(X3, y3, test_size = 0.4, random_state=42)
        Tx_test = torch.from_numpy(X_test).to(DEVICE)
        Ty_test = torch.from_numpy(y_test).type(torch.LongTensor).to(DEVICE)


        with torch.no_grad():
            y_pred, _, _ = student.forward(Tx_test)
            count = 0
            a = np.argmax(y_pred.detach().cpu().numpy(), axis = 1)
            b = Ty_test.detach().cpu().numpy()
            for it, it2 in zip(a, b):
                if it == it2:
                    count = count + 1
            test_acc =count/y_pred.shape[0]
            test_accs.append(test_acc)
            print("----Accuracy: ",test_acc)

    # Save test accuracies to CSV
    csv_file = "csvs/accuracy_over_time.csv"  # this will be overwritten every run
    days = np.power(2, range(2, 2 + len(test_accs)))

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Days Passed", "Accuracy"])
        for day, acc in zip(days, test_accs):
            writer.writerow([day, acc])

    print(f"Accuracy data saved to {csv_file}")

    return test_accs


print("-------------------------Experiment with Domain Adaptation-------------------------")
test_accs = exp(shot=10,shuffle=False,epoch=500,sim=False)


days = np.power(2,range(2,2+len(test_accs)))
labels = range(1,len(test_accs)+1)
# Create the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, test_accs, color='skyblue', edgecolor='blue',width=0.5)

# Add title and labels
plt.title("Accuracy changes over time with domain adaptation", fontsize=16)
plt.xlabel("Number of days passed ", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=12)

# Set x-axis to logarithmic scale
# plt.xscale('log')

# Customize x-ticks
plt.xticks(labels, [str(day) for day in days])

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.ylim(0,1)

# Display the plot
plt.show()
