# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:23:28 2021

@author: yanis
"""

############### Préparation des package et de l'environnement ###############

import pandas as pd 
import numpy as np
import os

path = os.path.abspath(os.path.join(os.path.dirname( os.getcwd() ), '.'))

path = r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Finance_Quant\Projet-Finance-Quant"
os.chdir(path+"\\3-Programmes")

from Utils.utils_finance import get_stock,get_multiple_stock,get_returns








########################## Importation des données ##########################

# Nous disposons de données pour 46 entreprises européeenne 
# coté chez euronext

# Importation des données ESG des 46 entreprises

df_ESG=pd.read_excel(path+"/2-Données/Raw_data/ESG_data.xls")


# Importation des rendements annuels des 46 entreprises

df_yearly_returns=pd.read_excel(path+"/2-Données/Raw_data/yearly_returns.xls")


# Importation des données boursieres des 46 entreprises

ticker_list=["ALV.DE","KNEBV.HE","KER.PA","FP.PA","VOW3.DE","SAN.PA","SU.PA","ENEL.MI","ISP.MI","INGA.AS","BAYN.DE","IBE.MC",
             "MC.PA","CRH.L","SIE.DE","DAI.DE","LIN.DE","ENGI.PA","AI.PA","ENI.MI","PHIA.AS","MUV2.DE","BAS.DE","BMW.DE","OR.PA","RI.PA"]
 
# a completer 

start_date= "2004-01-01"
end_date="2020-01-01"

df_stocks=get_multiple_stock(tickers=ticker_list,start=start_date, end=end_date , value = 'Open',index_as_date=True)
df_stocks_returns=get_returns(df_stocks)




# Importation de l'Euro Stoxx 50, l'Euronext 100 et du S&P 500 qui nous servent de benchmark

df_benchmark=get_multiple_stock(["^STOXX50E","^N100","^GSPC"],start=start_date, end=end_date, value="Open", index_as_date=True)
df_benchmark_returns=get_returns(df_benchmark)

# Importation des données retravaillé et annualisé 
df_benchmark_yearly=pd.read_excel(path+"/2-Données/Raw_data/benchmark_yearly.xls")
df_benchmark_yearly_returns=pd.read_excel(path+"/2-Données/Raw_data/benchmark_yearly_returns.xls")









########################## Préparation des données ##########################

# Jointure entre les données ESG et les rendements annuels

df_1 = pd.merge(df_ESG, df_yearly_returns , how='left' , on = ['date','Entreprises'])


# Jointure entre les données et les rendements annuels des benchmarks

# Retraitement des dates
df_1["year"]=pd.DatetimeIndex(df_1['date']).year
df_benchmark_yearly_returns["year"]=(pd.DatetimeIndex(df_benchmark_yearly_returns['Date']).year)-1
#Jointure
df = pd.merge(df_1, df_benchmark_yearly_returns , how='left' , on = ['year'])

# gestion des colonnes

df.drop(["date","year","Date"], inplace=True, axis=1)


# Creation de la variable cible 

# Voir nass




# Calcul des taux de croissance par variable
# Pas fini

var_list_growth=[""]

for var in var_list_growth :
    
    df[var+"_GROWTH"] = df[var].pct_change()
    
 


# Création des dummys pour savoir si il y'a une valeur manquantes

var_list_miss=[""]

for var in var_list_miss :
    
    df[var+"_MISS"] = (pd.to_numeric(df[var], errors='coerce').notnull() > 0).astype('int')
 

########################## Exportation des données ##########################

#df_stocks.to_csv(os.path.join(path, '2-Données' , 'stocks_data.csv'),sep=';',header=True,index=True)
df.to_csv(os.path.join(path, '2-Données' , 'data.csv'),sep=';',header=True,index=False)

