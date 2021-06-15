# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:21:01 2021

@author: mehdi
"""
import pandas as pd
import numpy as np
import os
from Utils.fonctions_VAE import VAE_imputer , locate_outliers_zscore , locate_nans , intersect_dicos

# IMPORTATION DES DONNEES TRAITEES
path = os.path.abspath(os.path.join(os.path.dirname( os.getcwd() ), '.'))
df = pd.read_csv(path+"\\2-Données\\data.csv",sep=';')


# SELECTION ET ISOLATION DES COLONNES NON ESG A NE PAS IMPUTER 
non_esg = df.iloc[:,[0,1,2,3,-1,-2]]
column_names = df.iloc[:,4:-2].columns.tolist()

# IMPUTATION DES DONNEES ESG
imputed_arr = VAE_imputer(df)
imputed = pd.DataFrame(data=imputed_arr , columns=column_names)



# SELECTION DES INDEX DES NAN PAR COLONNE
nans_index = locate_nans(df.iloc[:,4:-2],column_names)

# CORRECTION DES IMPUTATIONS
outliers_index = locate_outliers_zscore(imputed , columns = column_names , treshold = 2)


correction_index = intersect_dicos(nans_index, outliers_index) # toujours nan_index en premier


for k in correction_index :
    to_change = correction_index[k]
    mean_index = np.delete( np.arange(0,df.shape[0]) , to_change)
    imputed.loc[to_change,k] = imputed.loc[mean_index,k].mean()

# REGROUPER LES DONNEES ESG AVEC LES AUTRES
imputed = non_esg.join(imputed)


# Get variable categories
cat_vars = ["CHG_OF_CTRL_BFIT_GOLD_CHUTE_AGR" ,"CLAWBACK_PROVISION_FOR_EXEC_COMP" ,\
 "GENDER_PAY_GAP_BREAKOUT","BLANK_CHECK_PREFERRED_AUTHORIZED" ]

for col in cat_vars:
    imputed[col] = imputed[col].apply(lambda x : round(x)).astype('int')    
    