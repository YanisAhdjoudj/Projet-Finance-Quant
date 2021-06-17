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
<<<<<<< HEAD
path = r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Finance_Quant\Projet-Finance-Quant"
os.chdir(path+"\\3-Programmes")
=======
#os.chdir(path+"\\3-Programmes")
>>>>>>> 167fade9abeb06bbc79cdbe61c09733216c0e087
from Utils.utils_finance import get_stock,get_multiple_stock,get_returns

########################## Importation des données ##########################

# Nous disposons de données pour 26 entreprises européeenne 
# coté chez euronext

# Importation des données ESG des 26 entreprises

df_ESG=pd.read_excel(path+"/2-Données/data_fi.xls")


# Importation des rendements annuels des 26 entreprises

df_yearly_returns=pd.read_excel(path+"/2-Données/yearly_returns.xls")



# Importation des données boursieres des 26 entreprises

ticker_list=["ALV.DE","KNEBV.HE","KER.PA","FP.PA","VOW3.DE","SAN.PA","SU.PA","ENEL.MI","ISP.MI","INGA.AS","BAYN.DE","IBE.MC",
             "MC.PA","CRH.L","SIE.DE","DAI.DE","LIN.DE","ENGI.PA","AI.PA","ENI.MI","PHIA.AS","MUV2.DE","BAS.DE","BMW.DE","OR.PA","RI.PA"]

start_date= "2005-01-01"
end_date="2020-01-01"

df_stocks=get_multiple_stock(tickers=ticker_list,start=start_date, end=end_date , value = 'Open',index_as_date=True)
df_stocks_returns=get_returns(df_stocks)




# Importation de l'Euro Stoxx 50, l'Euronext 100 et du S&P 500 qui nous servent de benchmark

df_benchmark=get_multiple_stock(["^STOXX50E","^N100","^GSPC"],start=start_date, end=end_date, value="Open", index_as_date=True)
df_benchmark_returns=get_returns(df_benchmark)



########################## Préparation des données ##########################

# Jointure entre les données ESG et les rendements annuels

main_df = pd.merge(df_ESG, df_yearly_returns , how='left' , on = ['date','Entreprises'])

# Creation de la variable cible 
niveaux_notes=pd.cut(main_df['yearly_return'],bins=[-25, 15],labels=['Long', 'Mid', 'Short'])

main_df['target'] = main_df.apply(lambda x: 1 if x['yearly_return']<15 else if x['yearly_return'], axis=1)


# Création des dummys pour savoir si il y'a une valeur manquantes

main_df['dummy1'] = (pd.to_numeric(main_df.NUMBER_EMPLOYEES_CSR, errors='coerce').notnull() > 0).astype('int')
########################## Exportation des données ##########################

df_stocks.to_csv(os.path.join(path, '2-Données' , 'stocks_data.csv'),sep=';',header=True,index=True)
main_df.to_csv(os.path.join(path, '2-Données' , 'data.csv'),sep=';',header=True,index=False)

