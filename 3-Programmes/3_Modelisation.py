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
y_train = df_train['yearly_return']

X_test = df_test.drop(columns=['yearly_return'])
y_test = df_test['yearly_return']





