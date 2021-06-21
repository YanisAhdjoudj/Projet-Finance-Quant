# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 01:31:48 2021

@author: yanis
"""


import pandas as pd 
import numpy as np
import os

path = r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Finance_Quant\Projet-Finance-Quant"
os.chdir(path+"\\3-Programmes")


# Importation des données

df=pd.read_excel(path+"\\2-Données\\Raw_data\yearly_returns_for_portfolio.xls")
df_final_return=df.iloc[-1:]


###### Portefeuille Long only ######


def Return_on_long_pf(list_long,df,df_final_return):
    
# On construit le portefeuille 
    df_pf=df[list_long]
    
    
# Rendement du portefeuille
    rtrn=df_final_return[list_long]
    weight=1/rtrn.shape[1]
    weights=np.ones(rtrn.shape[1])*weight
    w_pf=rtrn*weights
    pf_return=w_pf.sum(axis=1).iloc[0]
    
        
# Variance du portefeuille
    weights=np.ones(rtrn.shape[1])*weight
    covar=np.cov(df_pf.T)
    pf_var=np.dot(weights.T,np.dot(covar,weights))
    pf_vol=np.sqrt(pf_var)
    
    
#Ratio de sharpe
    RS=pf_return/pf_vol


# Mise en forme

    Var_Resultat=[pf_return,pf_vol,RS]
    Index_Resultat=["Return","Volatility'","Sharpe Ratio"]
    
    Resultat=pd.Series(data=Var_Resultat, index=Index_Resultat)
    
    return Resultat


###### Portefeuille Long-short only ######

def Return_on_long_short_pf(list_long,list_short,df,df_final_return):
    
# On construit le portefeuille 
    df_pf=df[list_long]
    
    
# Rendement du portefeuille
    rtrn=df_final_return[list_long]
    weight=1/rtrn.shape[1]
    weights=np.ones(rtrn.shape[1])*weight
    weights=weights*np.array(list_short)
    w_pf=rtrn*weights
    pf_return=w_pf.sum(axis=1).iloc[0]
    
        
# Variance du portefeuille
    weights=np.ones(rtrn.shape[1])*weight
    covar=np.cov(df_pf.T)
    pf_var=np.dot(weights.T,np.dot(covar,weights))
    pf_vol=np.sqrt(pf_var)
    
    
#Ratio de sharpe
    RS=pf_return/pf_vol


# Mise en forme

    Var_Resultat=[pf_return,pf_vol,RS]
    Index_Resultat=["Return","Volatility'","Sharpe Ratio"]
    
    Resultat=pd.Series(data=Var_Resultat, index=Index_Resultat)
    
    return Resultat


###### Application ######

# Long
list_long_1=["DPW GR Equity","CS FP Equity","DTE GR Equity"]
Return_on_long_pf(list_long=list_long_1,df=df,df_final_return=df_final_return)


df.columns

# Short
list_long_1=["DPW GR Equity","CS FP Equity","DTE GR Equity"]
list_short=[1,-1,1]
Return_on_long_short_pf(list_long=list_long_1,list_short=list_short,df=df,df_final_return=df_final_return)



