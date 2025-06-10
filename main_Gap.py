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
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        y = F.log_softmax(y1, dim=1)
        y2 = F.log_softmax(y1 / 10, dim=1)
        return y, x3, y2

torch.manual_seed(1)

def exp(shot=5, shuffle=False, epoch=500, sim=True):
    # Load simulation data
    dataset = pd.read_csv("simulation.csv", sep=',')
    X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], axis=1)).astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = dataset['LA'].values.astype(np.float32)
    Sx = torch.from_numpy(X).to(DEVICE)
    Sy = torch.from_numpy(y).long().to(DEVICE)

    # Load physical data
    dataset3 = pd.read_csv("realworlddata.csv", sep=',')
    X3 = scaler.transform(np.array(dataset3.drop(['A', 'C', 'P', 'LA'], axis=1)).astype(np.float32))
    y3 = dataset3['LA'].values.astype(np.float32)
    X_train1, X_test, y_train1, y_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
    Tx = torch.from_numpy(X_train1).to(DEVICE)
    Ty = torch.from_numpy(y_train1).long().to(DEVICE)
    Tx_test = torch.from_numpy(X_test).to(DEVICE)
    Ty_test = torch.from_numpy(y_test).long().to(DEVICE)

    if shuffle:
        ids = np.arange(len(Tx_test))
        random.shuffle(ids)
        id_val = ids[:len(Tx_test)//2]
        id_test = ids[len(Tx_test)//2:]
        Tx_val = Tx_test[id_val]
        Ty_val = Ty_test[id_val]
        Tx_test = Tx_test[id_test]
        Ty_test = Ty_test[id_test]
    else:
        Tx_val = Tx_test[:len(Tx_test)//2]
        Ty_val = Ty_test[:len(Tx_test)//2]
        Tx_test = Tx_test[len(Tx_test)//2:]
        Ty_test = Ty_test[len(Ty_test)//2:]

    # Teacher model
    teacher = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88).to(DEVICE)
    optimizer = optim.Adam(teacher.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    # Train teacher
    for e in range(1, 301):
        teacher.train()
        if sim:
            y_pred, _, _ = teacher(Sx)
            loss = criterion(y_pred, Sy)
        else:
            y_pred, _, _ = teacher(Tx)
            loss = criterion(y_pred, Ty)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate teacher on simulation data
    if sim:
        with torch.no_grad():
            y_pred, _, _ = teacher(Sx)
            pred_labels = torch.argmax(y_pred, dim=1)
            ds_ds_acc = (pred_labels == Sy).float().mean().item()
            print("Dˢ → Dˢ Accuracy:", ds_ds_acc)

    # Student model (cloned from teacher)
    if sim: # initializes student model with teacher model
        student = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88).to(DEVICE)
        student.load_state_dict(teacher.state_dict())
        optimizer2 = optim.Adam(student.parameters(), lr=0.1)
    else: # initialize student model randomly
        student = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88).to(DEVICE)
        optimizer2 = optim.Adam(student.parameters(), lr=0.1)

    max_val_accuracy = 0
    max_test_accuracy = 0

    # Domain adaptation path (simulation only)
    if sim:
        for e in range(1, epoch + 1):
            temp1 = 10
            temp2 = 88 * shot
            for i in range(temp1):
                data = Sx[i*temp2:i*temp2+temp2]
                target = Sy[i*temp2:i*temp2+temp2]
                perm = torch.randperm(Tx.shape[0])[:temp2]
                x_tar = Tx[perm]
                y_target = Ty[perm]

                # Knowledge distillation
                y_src, x_src_mmd, raw_y1 = teacher(data)
                y_tar, x_tar_mmd, raw_y2 = student(data)
                tt = torch.argmax(raw_y1.detach(), dim=1)
                loss_high_t = criterion(raw_y2, tt)
                loss_high_t.backward()
                optimizer2.step()

                # MMD loss
                y_src, x_src_mmd, _ = teacher(x_tar)
                y_tar, x_tar_mmd, _ = student(x_tar)
                loss_mmd = LAMBDA * mmd_loss(x_src_mmd, x_tar_mmd)
                optimizer.zero_grad()
                optimizer2.zero_grad()
                loss_mmd.backward()
                optimizer.step()
                optimizer2.step()

                # Final classification loss on physical data
                optimizer2.zero_grad()
                y_tar, _, _ = student(x_tar)
                loss_c2 = criterion(y_tar, y_target)
                loss_c2.backward()
                optimizer2.step()

            # Validation
            with torch.no_grad():
                y_pred, _, _ = student(Tx_val)
                pred_labels = torch.argmax(y_pred, dim=1)
                val_acc = (pred_labels == Ty_val).float().mean().item()
                if val_acc > max_val_accuracy:
                    max_val_accuracy = val_acc
                    y_pred, _, _ = student(Tx_test)
                    pred_labels = torch.argmax(y_pred, dim=1)
                    test_acc = (pred_labels == Ty_test).float().mean().item()
                    if test_acc > max_test_accuracy:
                        max_test_accuracy = test_acc

    else:
        # Full-batch supervised training for Dᵖ → Dᵖ
        for e in range(1, epoch + 1):
            student.train()
            y_pred, _, _ = student(Tx)
            loss = criterion(y_pred, Ty)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            with torch.no_grad():
                y_pred, _, _ = student(Tx_val)
                pred_labels = torch.argmax(y_pred, dim=1)
                val_acc = (pred_labels == Ty_val).float().mean().item()
                if val_acc > max_val_accuracy:
                    max_val_accuracy = val_acc
                    y_pred, _, _ = student(Tx_test)
                    pred_labels = torch.argmax(y_pred, dim=1)
                    test_acc = (pred_labels == Ty_test).float().mean().item()
                    if test_acc > max_test_accuracy:
                        max_test_accuracy = test_acc

    if sim:
        print("Dˢ → Dᵖ Accuracy:", max_test_accuracy)
        return ds_ds_acc, max_test_accuracy
    else:
        print("Dᵖ → Dᵖ Accuracy:", max_test_accuracy)
        return max_val_accuracy, max_test_accuracy



# Run experiments
print("\n----------------Experiment with Simulation Data-------------------------")
sim_val_acc, sim_test_acc = exp(shot=0, shuffle=True, epoch=80, sim=True)

print("\n----------------Experiment with Physical Data-------------------------")
phy_val_acc, phy_test_acc = exp(shot=5, shuffle=True, epoch=80, sim=False)

# Plot results
labels = ['Dˢ → Dˢ', 'Dˢ → Dᵖ', 'Dᵖ → Dᵖ']
values = [sim_val_acc, sim_test_acc, phy_test_acc]
positions = [0.2, 0.7, 1.2]

plt.figure(figsize=(10, 6))
bars = plt.bar(positions, values, color='skyblue', edgecolor='black', width=0.3)
plt.xticks(positions, labels)
plt.title('Simulation-to-Reality Gap', fontsize=16)
plt.xlabel('Domain Adaptation Setting', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval*100:.2f}%', ha='center', va='bottom', fontsize=14)

plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.show()
