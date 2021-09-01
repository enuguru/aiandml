#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 00:18:08 2018

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



st=time.time()
k=5
test_f=np.zeros((5,n_users,n_movie))
train_f=np.zeros((5,n_users,n_movie))
for i in range(rating.shape[0]):
    nzi=np.nonzero(rating[i])
    np.random.shuffle(nzi[0])
   # per_20=int(len(nzi[0])/k)
    l=len(nzi[0])
    for j in range(k):
        test_f[j,i,nzi[0][int(l*j/k):int((l*(j+1))/k)]] = rating[i,nzi[0][int(l*j/k):int((l*(j+1))/k)]]

for x in range(k):
    train_f[x]=np.copy(rating)
    train_f[x] = np.where((rating !=0),rating-test_f[x],rating)

end=time.time()-st
print("5 fold split time = ",end)       


mae=[]
rmse=[]
precsn=[]
rcll=[]
fm=[]
#adjusted cosine similarity calculation
for t in range(k):
    start=time.time()
    u_m = train_f[t].sum(axis=1)/(train_f[t]!=0).sum(axis=1)
    rating_m_sub = np.where((train_f[t] !=0),train_f[t]-u_m[:,None],train_f[t])
    sim=np.zeros((n_movie,n_movie))
    for i in range(n_movie):
        print(i)
        #st=time.time()
        for j in range(i,n_movie):
            num=0
            dem1=0
            dem2=0
            set_c_u=np.where((train_f[t][:,i] !=0) * (train_f[t][:,j]) )[0]
            for z in set_c_u:
                num=num+rating_m_sub[z][i] * rating_m_sub[z][j]
                dem1=dem1 + rating_m_sub[z][i]**2
                dem2=dem2 + rating_m_sub[z][j]**2
                sim[i,j] = num/sqrt(dem1*dem2 +10**-12)
        #en=time.time()-st
        #print(en)
    end=time.time() -start
    print("sim cal time for" ,t ," is ",end)
    upp_tr=np.triu(sim,k=1)
    upp_tr=upp_tr.T
    sim=sim+upp_tr
    sim=np.where((sim <0),0,sim)
    
    
    #prediction
    
    
    mul=train_f[t].dot(sim)
    div=np.zeros((n_users,n_movie))
    stt=time.time()
    for i in range(n_users) :
        print("in prediction ",i)
        nzi=np.nonzero(train_f[t][i])
        for j in range(n_movie):
            sm=(sim[j,nzi]).sum()
            div[i,j] = sm
    endd=time.time() -stt
    print("prediction end ",endd)
    #np.nan_to_num(div,copy=False)
    pred=mul/div
    np.nan_to_num(pred,copy=False)
    
    MAE=mean_absolute_error(test_f[t][test_f[t]!=0],pred[test_f[t]!=0])
    MSE=mean_squared_error(test_f[t][test_f[t]!=0],pred[test_f[t]!=0])
    RMSE=sqrt(MSE)
    print("MAE = ",MAE)
    print("RMSE = ",RMSE)
    pred_nz=pred[test_f[t] !=0]
    test_nz=test_f[t][test_f[t] !=0]
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
    print(f1_measure)
    mae.append(MAE)
    rmse.append(RMSE)
    precsn.append(precision)
    rcll.append(recall)
    fm.append(f1_measure)
    
    
tot_time=time.time()-read_start_time
print(tot_time)






