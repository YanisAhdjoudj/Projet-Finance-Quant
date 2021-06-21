# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:15:31 2021

@author: mehdi
"""
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 
from sklearn.preprocessing import StandardScaler , MinMaxScaler , OneHotEncoder , LabelEncoder
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# IMPORTATION DES DONNEES TRAITEES
path = os.path.abspath(os.path.join(os.path.dirname( os.getcwd() ), '.'))
df = pd.read_csv(path+"\\2-Données\\Imputed_data\\VAE_imputed.csv",sep=';')

# ONE HOT ENCODAGE des entreprises
# entreprises = pd.get_dummies(df['Entreprises'] , drop_first=True)
# df = df.join(entreprises).drop('Entreprises',axis=1)

# Label encoding des entreprises
label = LabelEncoder()
df['Entreprises'] = label.fit_transform(df['Entreprises'])

df_train = df[df['Dates'].isin(df.Dates.unique().tolist()[:-3])].drop(columns=['Dates','^GSPC', '^N100','AUDIT_COMMITTEE_MEETINGS',\
                                                                               '^STOXX50E','NUMBER_EMPLOYEES_CSR'])

df_test = df[~df['Dates'].isin(df.Dates.unique().tolist()[:-3])].drop(columns=['Dates','^GSPC', '^N100','AUDIT_COMMITTEE_MEETINGS',\
                                                                               '^STOXX50E','NUMBER_EMPLOYEES_CSR'])


X_train = df_train.drop(columns=['yearly_return'])
X_test = df_test.drop(columns=['yearly_return'])

# conditions_train = [df_train['yearly_return']<-25 , df_train['yearly_return']>15]
# conditions_test = [df_test['yearly_return']<-25 , df_test['yearly_return']>15]
# y_train =  np.select(conditions_train , [0,2] , 1)
# y_test = np.select(conditions_test , [0,2] , 1)

y_train = np.where(df_train['yearly_return']>15 , 1 , 0)
y_test = np.where(df_test['yearly_return']>15 , 1 , 0)

print(np.bincount(y_train))
print(np.bincount(y_test))

##########################################################################
#                            MODELISATION                                #
##########################################################################


# Standardisation
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test =  sc.transform(X_test)


# Logistic Regression

lr = LogisticRegression(C=1.5 , class_weight='balanced', random_state=1, solver='lbfgs', max_iter=100, n_jobs=-1)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test , y_pred))



#### RANDOM FOREST 
rf = RandomForestClassifier(n_estimators=1000,  criterion='gini', max_depth=None,
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1,
                            random_state=1, verbose=0, warm_start=False, class_weight=None,
                            ccp_alpha=0.0, max_samples=None)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test , y_pred))




# XGBOOST SANS IMPUTATION
df = pd.read_csv(path+"\\2-Données\\Data\\data.csv",sep=';')

label = LabelEncoder()
df['Entreprises'] = label.fit_transform(df['Entreprises'])

df_train = df[df['Dates'].isin(df.Dates.unique().tolist()[:-3])].drop(columns=['Dates','^GSPC', '^N100','AUDIT_COMMITTEE_MEETINGS',\
                                                                               '^STOXX50E','NUMBER_EMPLOYEES_CSR'])

df_test = df[~df['Dates'].isin(df.Dates.unique().tolist()[:-3])].drop(columns=['Dates','^GSPC', '^N100','AUDIT_COMMITTEE_MEETINGS',\
                                                                               '^STOXX50E','NUMBER_EMPLOYEES_CSR'])


X_train = df_train.drop(columns=['yearly_return'])
X_test = df_test.drop(columns=['yearly_return'])


y_train = np.where(df_train['yearly_return']>15 , 1 , 0)
y_test = np.where(df_test['yearly_return']>15 , 1 , 0)



D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)


param = {
    'eta': 0.3, 
    'max_depth': 5,  
    'objective': 'multi:softprob',  
    'num_class': 2} 

steps = 40 


model = xgb.train(param, D_train, steps)

raw_preds = model.predict(D_test)
y_pred = np.asarray([np.argmax(line) for line in raw_preds])



print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test , y_pred))



################################################################################### 

#                       CONSTRUCTION DU PORTEFEUILLE                              #

###################################################################################



df_18 = df[df['Dates'] == df.Dates.unique().tolist()[-2] ].drop(columns=['Dates','^GSPC', '^N100','AUDIT_COMMITTEE_MEETINGS',\
                                                                               '^STOXX50E','NUMBER_EMPLOYEES_CSR'])

X_18 = df_18.drop(columns=['yearly_return'])
y_18 = np.where(df_18['yearly_return']>15 , 1 , 0)

pred_18 = rf.predict(X_18)

ent = label.inverse_transform(X_18['Entreprises'])

ptf = pd.DataFrame({'Ent' : ent , 'is_long' : pred_18 })

ptf['X'] = ptf['is_long']/ptf['is_long'].sum()


returns = df[df['Dates'] == df.Dates.unique().tolist()[-1] ][['Entreprises','yearly_return']]

returns['Ent'] = label.inverse_transform(returns['Entreprises'])

returns.drop('Entreprises', axis=1,inplace=True)

ptf = pd.merge(ptf , returns ,how='inner', on = ['Ent'])






