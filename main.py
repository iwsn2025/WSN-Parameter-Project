#!/usr/bin/env python
# encoding: utf-8

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
from nnmodel import DaNN
from data import  dataload, medataload1
from data import  medataloadr, medataloadtm 
from copy import deepcopy
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "cpu"
torch.manual_seed(1)


accu_list = [0 for _ in range(10)]

x_spt, y_spt, x_qry, y_qry = medataload1(simulation, task, size1, size2)
#print(x_spt[0].shape, y_spt[1].shape, x_qry[2].shape, y_qry[2].shape)

teacher = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
teacher = teacher.to(DEVICE)

class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.update_step = update_step ## task-level inner update steps
        self.update_step_test = update_step_test
        self.net = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
        self.base_lr = 1 * 1e-2
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr = 0.1)

    def forward(self, x_spt, y_spt, x_qry, y_qry, train):
        if(train == 1):
            task_num = 5
        if(train == 0):
            task_num = task1
        if(train == 2):
            task_num = task2

        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            y_hat = self.net(x_spt[i], params = None)
            loss = F.cross_entropy(y_hat, y_spt[i]) 
            grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=True)
            tuples = zip(grad, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            self.net.parameters(fast_weights) 
           
            with torch.no_grad():
                y_hat1 = self.net(x_qry[i], self.net.parameters())
                loss_qry = F.cross_entropy(y_hat1, y_qry[i])
                loss_list_qry[0] += loss_qry
                pred_qry = F.softmax(y_hat1, dim=1).argmax(dim=1) 
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                #print("0 ", i, correct/y_hat1.shape[0])
        
            with torch.no_grad():
                y_hat1 = self.net(x_qry[i], fast_weights)
                loss_qry = F.cross_entropy(y_hat1, y_qry[i])
                loss_list_qry[1] += loss_qry
                pred_qry = F.softmax(y_hat1, dim=1).argmax(dim=1)  
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()     
                #print("1 ", i, correct/y_hat1.shape[0])
            
            for k in range(1, self.update_step):
            
                y_hat1 = self.net(x_spt[i], params = fast_weights) 
                loss = F.cross_entropy(y_hat1, y_spt[i])
                grad = torch.autograd.grad(loss, self.net.parameters())
                tuples = zip(grad, fast_weights) 
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
                
                y_hat1 = self.net(x_qry[i], params = fast_weights)
                loss_qry = F.cross_entropy(y_hat1, y_qry[i])
                loss_list_qry[k+1] += loss_qry
                
                with torch.no_grad():
                    pred_qry = F.softmax(y_hat1,dim=1).argmax(dim=1)
                    correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                    #print(k, correct/y_hat1.shape[0])
                      
        loss_qry = loss_list_qry[-1] / task_num   
        self.meta_optim.zero_grad() 
        loss_qry.backward()
        self.meta_optim.step()

        return y_hat1   

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, accu_counter):
        
        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt, params = None)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        with torch.no_grad():
            y_hat = new_net(x_qry,  params = new_net.parameters())
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  
            #print(pred_qry.shape)
            #print(pred_qry)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            
            print("I ", correct/y_hat.shape[0])

        with torch.no_grad():
            y_hat = new_net(x_qry, params = fast_weights)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  
            correct = torch.eq(pred_qry, y_qry).sum().item()
            #print("1 ", correct/y_hat.shape[0])

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params = fast_weights)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = torch.autograd.grad(loss, new_net.parameters())
            fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, fast_weights)))
            
            y_hat = new_net(x_qry, fast_weights)
            
            with torch.no_grad():
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct = torch.eq(pred_qry, y_qry).sum().item()
                
                if (k >= 0):
                    print(correct/y_hat.shape[0])
                if (k == (update_step_test-1)):
                    accu_list[accu_counter] = correct/y_hat.shape[0]

        del new_net
        return fast_weights

meta = MetaLearner().to(DEVICE)
start = time.perf_counter()

for e in range(1, 2 + 1):
    y_pred = meta(x_spt, y_spt, x_qry, y_qry, 1)

print("--------------------------------------------")

x_spt, y_spt, x_qry, y_qry = medataloadr(realworlddata, task, size1, size2)
for e in range(1, 5 + 1):
    y_pred = meta(x_spt, y_spt, x_qry, y_qry, 0)

x_spt, y_spt, x_qry, y_qry = medataloadtm(6, spt_size)
for e in range(1, epoch + 1):
    y_pred = meta(x_spt, y_spt, x_qry, y_qry, 2)
    #print("P")

meta.finetunning(x_spt[5], y_spt[5], x_qry[5], y_qry[5], 0)



print("---------------------------------------------------------")

end = time.perf_counter()
print("Run time:", end - start)
