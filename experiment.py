#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mmd
from variable import *
from MatSampler import MatSampler
import random


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#calculate mmd loss
def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])

#Definition of the MLP Model in the paper
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


#Execute the experiment using a simulator in the student model training process, with a total of 500 epochs.
#The sampling algorithm will be employed as the default method.
#Without sampling 5 shots of physical data from the target domain will be used.
def experiment(simulator,student_epoch=500,shots=5,sampling_enabled=True):

    #initialize the random seed
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(1)

    #load source data(simulation data) with certain simulator



    dataset = pd.read_csv("data/training/simulationData/%s.csv"%(simulator), sep=',')

    X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], 1))
    X = np.float32(X)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = np.array(dataset['LA'])
    y = np.float32(y)
    Sy = torch.from_numpy(y).type(torch.LongTensor)
    Sx = torch.from_numpy(X)
    Sy = Sy.to(DEVICE)
    Sx = Sx.to(DEVICE)

    print(Sx.shape, Sy.shape)

    #number of configurations
    n_configs = 88

    # Load physical data, which represents our target domain data.
    # If sampling is enabled, utilize data samples handpicked by our algorithm, paired with a specific simulator.
    # In cases where sampling is disabled, resort to the full dataset of 440 (88*5) samples.
    if sampling_enabled:
        dataset2 = pd.read_csv("data/training/physicalDataWithSamplingAlgorithm/%s_selected_samples.csv"%(simulator), sep=',')[0:n_configs*shots]
    else:
        print("read from data/training/physicalData/physicalData.csv ")
        dataset2 = pd.read_csv("data/training/physicalData/physicalData.csv", sep=',')[0:n_configs*shots]

    X2 = np.array(dataset2.drop(['A', 'C', 'P', 'LA'], 1))
    X2 = np.float32(X2)
    X2 = scaler.transform(X2)
    y2 = np.array(dataset2['LA'])
    y2 = np.float32(y2)
    Tx = torch.from_numpy(X2)
    Ty = torch.from_numpy(y2).type(torch.LongTensor)

    Ty = Ty.to(DEVICE)
    Tx = Tx.to(DEVICE)

    print(Tx.shape, Ty.shape)

    #load data for testing (target domain data)

    dataset3 = pd.read_csv("data/testing/testData.csv", sep=',')
    X3 = np.array(dataset3.drop(['A', 'C', 'P', 'LA'], 1))
    X3 = np.float32(X3)
    X3 = scaler.transform(X3)
    y3 = np.array(dataset3['LA'])
    y3 = np.float32(y3)
    # X_train1, X_test, y_train1, y_test = train_test_split(X3, y3, test_size = 0.4, random_state=42)

    X_test = X3
    y_test = y3
    Tx_test = torch.from_numpy(X_test)
    Ty_test = torch.from_numpy(y_test).type(torch.LongTensor)
    Ty_test = Ty_test.to(DEVICE)
    Tx_test = Tx_test.to(DEVICE)
    print(Tx_test.shape, Ty_test.shape)

    #Teacher model
    teacher = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    teacher = teacher.to(DEVICE)
    optimizer = optim.Adam(teacher.parameters(), lr=LEARNING_RATE*10)

    criterion = nn.CrossEntropyLoss()

    #Teacher Model training process
    print("Teacher Model training process")
    #100 epoches
    for e in range(1, 100 + 1):
        teacher.train()
        y_pred, _, _ = teacher.forward(Sx)
        loss_c = criterion(y_pred, Sy)
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()

        count = 0
        if DEVICE == "cpu":
            a = np.argmax(y_pred.detach().numpy(), axis = 1)
            b = Sy.detach().numpy()
        else:
            a = np.argmax(y_pred.cpu().data.detach().numpy(), axis = 1)
            b = Sy.cpu().data.detach().numpy()

        for it, it2 in zip(a, b):
            if it == it2:
                count = count + 1
        if e%50==0:
            print("epoch %d : accuraccy "%(e),count/y_pred.shape[0])


    print("-----------")

    #Student model
    student = DaNN(n_input=3, n_hidden1=120, n_hidden2=84, n_class=88)
    student = student.to(DEVICE)
    student.load_state_dict(teacher.state_dict())
    optimizer2 = optim.SGD(
        student.parameters(),
        lr=LEARNING_RATE*3,
        momentum=MOMENTUM,
        weight_decay=L2_WEIGHT
    )

    #Student Model training process
    print("Student Model training process")
    max_acc = 0.00
    #for each epoch
    for e in range(1, student_epoch + 1):

        #10 * 88 *5 of data for training
        temp1 = 10
        temp2 = 88*5
        for i in range(temp1):

            #each batch in source domain for training
            data = Sx[i*temp2:i*temp2+temp2]

            #88*5 data from target domain for training
            x_tar = Tx[0:temp2]
            y_target = Ty[0:temp2]

            y_src, x_src_mmd, raw_y1 = teacher(data)
            y_tar, x_tar_mmd, raw_y2 = student(data)

            if DEVICE == "cpu":
                tt = np.argmax(raw_y1.detach().numpy(), axis = 1)
            else:
                tt = np.argmax(raw_y1.cpu().data.detach().numpy(), axis = 1)


            tt = torch.from_numpy(tt).to(DEVICE)
            loss_high_t = criterion(raw_y2, tt)

            loss_high_t.backward()

            optimizer2.step()

            y_src, x_src_mmd, y_src_soft = student(x_tar)
            y_tar, x_tar_mmd, y_tar_soft = student(x_tar)
            loss_mmd = LAMBDA * mmd_loss(x_src_mmd, x_tar_mmd)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss_mmd.backward()
            optimizer.step()
            optimizer2.step()



            optimizer2.zero_grad()
            y_tar, _, _ = student(x_tar)
            loss_c2 = criterion(y_tar, y_target)
            loss_c2.backward()
            optimizer2.step()

        #testing the accuracy of the model in each epoch
        with torch.no_grad():
            y_pred, _, _ = student.forward(Tx_test)
            count = 0

            if DEVICE == "cpu":
                a = np.argmax(y_pred.detach().numpy(), axis = 1)
                b = Ty_test.detach().numpy()
            else:
                a = np.argmax(y_pred.cpu().data.detach().numpy(), axis = 1)
                b = Ty_test.cpu().data.detach().numpy()

            for it, it2 in zip(a, b):
                if it == it2:
                    count = count + 1


            if count/y_pred.shape[0] >max_acc:
                max_acc = count/y_pred.shape[0]

            if e%50==0:
                print("epoch %d : accuraccy "%(e),count/y_pred.shape[0])

    return max_acc


def testWithSampleNumbers(extraSample=0,sampling_enabled=True,simulator = "tossim",seed = 1):
#experiment with or without sampling algorithm using one simulator.

            sampleCount = 88*5
            #load sample data

            if sampling_enabled:
                ms = MatSampler(simulator=simulator)
                sc= ms.sampleByMahalanobis(extraSample=extraSample,seed=seed)
                #update sample count after sampling
                sampleCount = np.sum(sc)
                print("Actual sample count: ",sampleCount)

            #train the model and test it on testing dataset
            experiment_acc = experiment(simulator=simulator,student_epoch=500,shots=5,sampling_enabled=sampling_enabled)

            #save the experiment results
            experiment_result={
                "extraSample":extraSample,
                "sampleCount":sampleCount,
                "experiment_acc":experiment_acc
            }

    #print out the experiment results
            print(experiment_result)
            return experiment_result




def test(sampling_enabled=True):
#experiment with or without sampling algorithm using four simulators.

    print("sampling_enabled",sampling_enabled)
    simulators = ["ns3","cooja","tossim","omnet"]


    experiment_result=dict()

    #for each simulator
    for i in simulators:
            sampleCount = 88*5

            # Trigger sampling algorithm based on Mahalanobis distance if sampling is enabled
            # This will regenerate selected physical data samples according to our algorithm
            if sampling_enabled:
                ms = MatSampler(simulator=i)
                sc= ms.sampleByMahalanobis()

                sampleCount = np.sum(sc)
                print("----------------------------------------------------------------------------")
                print("Experiment with sampling algorithm using %s simulator."%(i))
                print("Sample count in each network configurations:",sampleCount)
            else:
                print("----------------------------------------------------------------------------")
                print("Experiment without sampling algorithm using %s simulator."%(i))

            print("----------")
            # Utilize physical or physical sampling data to train the model
            # Validate the model's performance using a separate testing dataset
            # Return the model's output for further analysis
            experiment_acc = experiment(simulator="omnet",student_epoch=500,shots=5,sampling_enabled=sampling_enabled)

            #save the experiment results
            experiment_result[i]={
                "sampleCount":sampleCount,
                "experiment_acc":experiment_acc
            }

    #print out the experiment results
    print("----------------------------------------------------------------------------")
    print(experiment_result)

    return experiment_result


def experimentWithDiffSampleNumber():
#experiment with different number of samples using TOSSIM simulators by adding or removing samples to our algorithm selected dataset.

    resss = []

    #run expriment 10 times using different random seeds
    for i in range(1,11):
        print("----------------------------------------------------------------------------")
        print("Experiment with random seed %d"%(i))

        ress = []
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        for j in range(-160,240,40):
            # Perform a test using a sampling algorithm on the 'tossim' simulator to select a specific number of physical data samples.
            # If the variable 'i' is greater than 0, it indicates adding 'i' additional samples from the remaining ones.
            # If the variable 'i' is less than 0, it implies removing 'i' extra samples from the remaining ones.

            print("Experiment with TOSSIM simulator with %d extra samples to our algorithm selected dataset "%(j))
            res = testWithSampleNumbers(extraSample=j,sampling_enabled=True,simulator = "tossim",seed=i)
            ress.append(res)
        print("----------")
        # print(ress)
        resss.append(ress)
    print("----------------------------------------------------------------------------")
    print("Expriment results:")
    # print(resss)

    #collect the results by sample count
    mat_res = dict()
    for i in resss:
           # print(i)

           for j in i:
                  if j["sampleCount"] not in mat_res.keys():
                         mat_res[j["sampleCount"]] = []
                  mat_res[j["sampleCount"]].append(j["experiment_acc"])

    #print out the experiment results
    print("----------------------------------------------------------------------------")
    print(mat_res)

    return mat_res