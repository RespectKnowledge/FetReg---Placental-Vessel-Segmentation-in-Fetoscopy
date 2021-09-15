# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:23:24 2021

@author: Abdul Qayyum
"""

#%% Prepare dataset for training and validation
import os 
root='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FetaEndo_challenege\\FetReg2021_Task1_Segmentation'
lstdir=os.listdir(root)

import random
#random.seed(0)
def Trian_val(data_list,test_size=0.15):
    n=len(data_list)
    m=int(n*test_size)
    test_item=random.sample(data_list,m)
    train_item=list(set(data_list)-set(test_item))
    return train_item,test_item
tr_list,test_list=Trian_val(lstdir,test_size=0.20)
import pandas as pd
df_test= pd.DataFrame(test_list,columns=['PatientID'])
df_train= pd.DataFrame(tr_list,columns=['PatientID'])
# df.to_csv("testing_list1.csv",index=False)
# df1e=df11.drop(['index'],axis=0)
# df11.values()
df_test.to_csv('valid_fold5.csv', index=False) 
df_train.to_csv('train_fold5.csv', index=False)        