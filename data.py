import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
import torch
import torch.nn as nn
from variable import *

trainfile = [newtrain11, newtrain12, newtrain13, newtrain14, newtrain15, newtrain16]
testfile = [newtest11, newtest12, newtest13, newtest14, newtest15, newtest16]

def dataload(path):
	dataset = pd.read_csv(path, sep=',')
	X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], 1))
	X = np.float32(X)
	scaler = StandardScaler().fit(X) 
	X = scaler.transform(X) 
	y = np.array(dataset['LA'])
	y = np.float32(y)
	Sy = torch.from_numpy(y).type(torch.LongTensor)
	Sx = torch.from_numpy(X)
	return Sx, Sy

def medataload1(path, task_num, size1, size2):
	dataset = pd.read_csv(path, sep=',')
	X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], axis=1))
	X = np.float32(X)
	scaler = StandardScaler().fit(X) 
	X = scaler.transform(X) 
	X = torch.from_numpy(X)
	#print(X.shape)
	x_spt, y_spt, x_qry, y_qry = [], [], [], []
	for task in range (0, task_num):
		M1 = X[(task*(size1+size2)):(task*(size1+size2)+size1)]
	#	print(M1.shape)
		M2 = X[(task*(size1+size2)+size1):((task+1)*(size1+size2))]
		for e in range(1,88):
			N1 = X[(task*(size1+size2)+e*75):(task*(size1+size2)+size1+e*75)]
			N2 = X[(task*(size1+size2)+size1+e*75):((task+1)*(size1+size2)+e*75)]
			M1 = np.vstack((M1,N1))
			M2 = np.vstack((M2,N2))
		M1 = torch.from_numpy(M1)
		M2 = torch.from_numpy(M2)
		
		x_spt.append(M1)
		x_qry.append(M2)
	

	Y = np.array(dataset['LA'])
	Y = np.float32(Y)
	Y = torch.from_numpy(Y).type(torch.LongTensor)

	for task in range (0, task_num):
		R1 = Y[(task*(size1+size2)):(task*(size1+size2)+size1)]
		R2 = Y[(task*(size1+size2)+size1):((task+1)*(size1+size2))]
		for e in range(1,88):
			S1 = Y[(task*(size1+size2)+e*75):(task*(size1+size2)+size1+e*75)]
			S2 = Y[(task*(size1+size2)+size1+e*75):((task+1)*(size1+size2)+e*75)]
			R1 = np.hstack((R1,S1))
			R2 = np.hstack((R2,S2))
		R1 = torch.from_numpy(R1)
		R2 = torch.from_numpy(R2)
	
		y_spt.append(R1)
		y_qry.append(R2)

	return x_spt, y_spt, x_qry, y_qry
	


def medataloadr(path, task_num, size1, size2):
	dataset2 = pd.read_csv(simulation, sep=',')
	X2 = np.array(dataset2.drop(['A', 'C', 'P', 'LA'], axis=1))
	X2 = np.float32(X2)
	scaler = StandardScaler().fit(X2) 

	dataset = pd.read_csv(path, sep=',')
	X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], axis=1))
	X = np.float32(X)
	
	X = scaler.transform(X) 
	X = torch.from_numpy(X)
	#print(X.shape)
	x_spt, y_spt, x_qry, y_qry = [], [], [], []
	for task in range (0, task_num):
		M1 = X[(task*(size1+size2)):(task*(size1+size2)+size1)]
	#	print(M1.shape)
		M2 = X[(task*(size1+size2)+size1):((task+1)*(size1+size2))]
		for e in range(1,88):
			N1 = X[(task*(size1+size2)+e*75):(task*(size1+size2)+size1+e*75)]
			N2 = X[(task*(size1+size2)+size1+e*75):((task+1)*(size1+size2)+e*75)]
			M1 = np.vstack((M1,N1))
			M2 = np.vstack((M2,N2))
		M1 = torch.from_numpy(M1)
		M2 = torch.from_numpy(M2)
		#print(M1)
		#print(M2.shape)
		x_spt.append(M1)
		x_qry.append(M2)
	
#	print(x_spt[0].shape)
#	print(x_spt[0])
#	print(x_qry[2].shape)
#	print(x_qry[2])

	Y = np.array(dataset['LA'])
	Y = np.float32(Y)
	Y = torch.from_numpy(Y).type(torch.LongTensor)

	for task in range (0, task_num):
		R1 = Y[(task*(size1+size2)):(task*(size1+size2)+size1)]
		R2 = Y[(task*(size1+size2)+size1):((task+1)*(size1+size2))]
		for e in range(1,88):
			S1 = Y[(task*(size1+size2)+e*75):(task*(size1+size2)+size1+e*75)]
			S2 = Y[(task*(size1+size2)+size1+e*75):((task+1)*(size1+size2)+e*75)]
			R1 = np.hstack((R1,S1))
			R2 = np.hstack((R2,S2))
		R1 = torch.from_numpy(R1)
		R2 = torch.from_numpy(R2)
	#	print(R1.shape)
	#	print(R2.shape)
		y_spt.append(R1)
		y_qry.append(R2)
#	print(y_spt[1].shape)
#	print(y_spt[1])
#	print(y_qry[2].shape)
#	print(y_qry[2])
	return x_spt, y_spt, x_qry, y_qry
	
	

#Load multiple physcial data
def medataloadtm(task_num, size1):
	dataset2 = pd.read_csv(simulation, sep=',')
	X2 = np.array(dataset2.drop(['A', 'C', 'P', 'LA'], axis=1))
	X2 = np.float32(X2)
	scaler = StandardScaler().fit(X2) 
	#print(X2.shape)

	x_spt, y_spt, x_qry, y_qry = [], [], [], []
	for i in range (0, task_num):
		dataset = pd.read_csv(trainfile[i], sep=',')
		X = np.array(dataset.drop(['A', 'C', 'P', 'LA'], axis=1))
		#if (i == 0):
		#	print(X)
		X = np.float32(X)
		
		X = scaler.transform(X) 
		X = torch.from_numpy(X)
		Sx = X[0:88*size1]
		
		x_spt.append(Sx)
	
		Y = np.array(dataset['LA'])
		Y = np.float32(Y)
		Y = torch.from_numpy(Y).type(torch.LongTensor)
		Sy = Y[0:88*size1]
		y_spt.append(Sy)
	#print(x_spt[1])
	#print(y_spt)

	for i in range (0, task_num):
		dataset1 = pd.read_csv(testfile[i], sep=',')
		X1 = np.array(dataset1.drop(['A', 'C', 'P', 'LA'], axis=1))
		
		X1 = np.float32(X1)
	
		X1 = scaler.transform(X1) 
		Sx1 = torch.from_numpy(X1)
		x_qry.append(Sx1)
		
		Y1 = np.array(dataset1['LA'])
		Y1 = np.float32(Y1)
		Sy1 = torch.from_numpy(Y1).type(torch.LongTensor)
		y_qry.append(Sy1)
	#print(x_qry[0])
	#print(y_qry[1].shape)
	#print(x_qry[1])
	return x_spt, y_spt, x_qry, y_qry

