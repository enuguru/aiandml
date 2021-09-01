#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:23:31 2018

@author: sanjeet
"""

import pandas as pd
import numpy as np
import time
from math import sqrt
from numpy import random
from sklearn.metrics import mean_absolute_error,mean_squared_error





cols=['userId','itemId','rating','timestamp']
#reading data
read_start_time=time.time()
df=pd.read_table('ml-1m/ratings.dat',sep='::',header=None,names=cols)
df.drop(columns=['timestamp'],inplace=True)
read_end_time = time.time()
print("Read time = ",read_end_time-read_start_time)


# total user and movie
n_users=df.userId.max()
n_movie=df.itemId.max()



#converting data to matrix form
mat_start=time.time()
rating=np.zeros((n_users,n_movie))
for  row in df.itertuples():
    rating[row[1]-1,row[2]-1]=row[3]
mat_end=time.time()
print("Matrix conversion time = ",mat_end-mat_start)


#train_test_splitting
trainset = np.copy(rating)
testset=np.zeros((n_users,n_movie))
#z for iterate over each row
train_test=time.time()
z=0
for row in rating:
    nz_in=np.nonzero(row)
    per_20=int(len(nz_in[0])*0.2)
    rand =random.choice(nz_in[0],per_20,replace=False)
    for i in range(per_20):
        testset[z,rand[i]] =rating[z,rand[i]]
        trainset[z,rand[i]] = 0
    z =z+1

train_test_end=time.time()
print("train_test split time = ",train_test_end-train_test)
        

#adjusted cosine similarity calculation
def adj_c_sim(train_data):
    start=time.time()
    u_m = train_data.sum(axis=1)/(train_data !=0).sum(axis=1)
    rating_m_sub = np.where((train_data !=0),train_data-u_m[:,None],train_data)
    sim=np.zeros((n_movie,n_movie))
    for i in range(n_movie):
        print(i)
        #st=time.time()
        for j in range(i,n_movie):
            num=0
            dem1=0
            dem2=0
            set_c_u=np.where((train_data[:,i] !=0) * (train_data[:,j]) )[0]
            for k in set_c_u:
                num=num+rating_m_sub[k][i] * rating_m_sub[k][j]
                dem1=dem1 + rating_m_sub[k][i]**2
                dem2=dem2 + rating_m_sub[k][j]**2
                sim[i,j] = num/sqrt(dem1*dem2 +10**-12)
        #en=time.time()-st
        #print(en)
    end=time.time() -start
    print("sim time = ",end)
    return sim
                
sim=adj_c_sim(trainset)
#copying below diagonl of similarity matrix
upp_tr=np.triu(sim,k=1)
upp_tr=upp_tr.T
sim=sim+upp_tr

sim=np.where((sim <0),0,sim)
#save sim mat to sim.txt
#np.savetxt('sim.txt',sim, fmt='%.4f',delimiter=' ')

#prediction
mul=trainset.dot(sim)
div=np.zeros((n_users,n_movie))
stt=time.time()
for i in range(n_users) :
    #print(i)
    nzi=np.nonzero(trainset[i])
    #print(nzi)
    for j in range(n_movie):
        sm=(sim[j,nzi]).sum()
        div[i,j] = sm
    endd=time.time() -stt
    print(endd)
    
#np.nan_to_num(div,copy=False)
pred=mul/div
np.nan_to_num(pred,copy=False)
#save pred mat to sim.txt
#np.savetxt('pred.txt',pred, fmt='%.4f',delimiter=' ')

#Error calculation

MAE=mean_absolute_error(testset[testset!=0],pred[testset!=0])
MSE=mean_squared_error(testset[testset!=0],pred[testset!=0])
RMSE=sqrt(MSE)

print("MAE = ",MAE)
print("RMSE = ",RMSE)

#precision,Recall,F1-measure
pred_nz=pred[testset !=0]
test_nz=testset[testset !=0]
tp=0
fp=0
fn=0
th=4 #threshold value
for i in range(len(pred_nz)):
   if test_nz[i] >=th and pred_nz[i] >=th:
       tp+=1
   elif test_nz[i] < th and pred_nz[i] >= th :
       fp+=1
   elif test_nz[i] >= th and pred_nz[i] < th :
       fn+=1
       
precision=tp/(tp+fp)
recall=tp/(tp+fn)
#f1 measure= 2*(precision *recall)/(precision +recall)
f1_measure=2*(precision *recall)/(precision +recall)
