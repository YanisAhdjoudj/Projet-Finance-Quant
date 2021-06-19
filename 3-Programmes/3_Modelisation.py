# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:15:31 2021

@author: mehdi
"""
import pandas as pd
import numpy as np
import os



# IMPORTATION DES DONNEES TRAITEES
path = os.path.abspath(os.path.join(os.path.dirname( os.getcwd() ), '.'))
df = pd.read_csv(path+"\\2-Données\\VAE_imputed.csv",sep=';')



df.Dates.unique()

df.columns

df_train = df[df['Dates'].isin(df.Dates.unique().tolist()[:-3])].drop(columns=['Dates','date','NUMBER_EMPLOYEES_CSR',\
                                                                               'SP 500','Entreprises'])

df_test = df[~df['Dates'].isin(df.Dates.unique().tolist()[:-3])].drop(columns=['Dates','date','NUMBER_EMPLOYEES_CSR',\
                                                                               'SP 500','Entreprises'])


X_train = df_train.drop(columns=['yearly_return'])
X_test = df_test.drop(columns=['yearly_return'])

conditions_train = [df_train['yearly_return']<-25 , df_train['yearly_return']>15]
conditions_test = [df_test['yearly_return']<-25 , df_test['yearly_return']>15]

y_train =  np.select(conditions_train , [0,2] , 1)
y_test = np.select(conditions_test , [0,2] , 1)





