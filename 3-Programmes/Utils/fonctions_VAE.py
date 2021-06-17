# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 01:19:18 2021

@author: mehdi
"""
from typing import Union
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.cluster import DBSCAN
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from Utils.autoencoders import TFVariationalAutoencoder


def VAE_imputer(main_df):
    # DEFINE HYPERPARAMETERS
    
    # VAE network size:
    Decoder_hidden1 = 20
    Decoder_hidden2 = 20
    Encoder_hidden1 = 20
    Encoder_hidden2 = 20
    
    # dimensionality of latent space:
    latent_size = 5
    
    # training parameters:
    training_epochs = 500
    batch_size = 250
    learning_rate = 0.001
    
    # specify number of imputation iterations:
    ImputeIter = 25
    '''
    ==============================================================================
    '''
    # LOAD DATA
    # Load data from a csv for analysi
    Xdata_df = main_df.iloc[:,4:-2]
    Xdata = Xdata_df.values
    del Xdata_df
    
    # Load data with missing values from a csv for analysis:
    Xdata_df = main_df.iloc[:,4:-2]
    Xdata_Missing = Xdata_df.values
    del Xdata_df
    
    # Properties of data:
    n_x = Xdata_Missing.shape[1] # dimensionality of data space
    
    ObsRowInd = np.where(np.nansum(Xdata_Missing,axis=1)!=0)
    

    NanIndex = np.where(np.isnan(Xdata_Missing))

    sc = StandardScaler()
    Xdata_Missing_complete = np.copy(Xdata_Missing[ObsRowInd[0],:])
    # standardise using complete records:
    sc.fit(Xdata_Missing_complete)
    Xdata_Missing[NanIndex] = 0
    Xdata_Missing = sc.transform(Xdata_Missing)
    Xdata_Missing[NanIndex] = np.nan
    del Xdata_Missing_complete
    Xdata = sc.transform(Xdata)
    
    
    '''
    ==============================================================================
    '''
    # INITIALISE AND TRAIN VAE
    # define dict for network structure:
    network_architecture = \
        dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
             n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
             n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
             n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
             n_input=n_x, # data input size
             n_z=latent_size)  # dimensionality of latent space
    
    # initialise VAE:
    vae = TFVariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    
    
    # train VAE on corrupted data:
    vae = vae.train(XData=Xdata_Missing,
                    training_epochs=training_epochs)
    
    
    '''
    ==============================================================================
    '''
    # IMPUTE MISSING VALUES
    # impute missing values:
    X_impute = vae.impute(X_corrupt = Xdata_Missing, max_iter = ImputeIter)
    
    # Standardise Xdata_Missing and Xdata wrt Xdata:
    X_impute = sc.inverse_transform(X_impute)
    
    vae.sess.close()
    
    return X_impute 

def locate_outliers_dbscan(data, columns = ['ead','MT_INI_FIN_','mt_appo_']):
    scaler = MinMaxScaler() 
    df = scaler.fit_transform(data[columns])
    
    outlier_detection = DBSCAN(eps = 0.1, metric="euclidean", 
                                 min_samples = 5,
                                 n_jobs = -1)
    clusters = outlier_detection.fit_predict(df)
    
    return np.where(clusters==-1)[0].tolist()
    

def locate_outliers_zscore(data , columns = ['ead','MT_INI_FIN_','mt_appo_'] , treshold = 3):
    """
    Cette fonction permet de localiser les valeurs abberantes, 
    Elle prend argument un dataframe <data>, une liste de colonne <columns> et 
    un seuil du zscore <treshold>.
    Elle renvoie un dictionnaire, qui contient pour chaque colonne, les indices où
    se trouvent les valeurs aberrantes.
    
    Paramètres :
            Dataframe : pd.DataFrame 
            liste de colonnes : list[str]
            seuil : float
            
    returns :
            Dictionnaire : dict{ str : list[int] }

    """
    df = data[columns]
    z_score = np.abs(stats.zscore(df))
    outliers= np.where(z_score>treshold , True , False)
    
    r , c = np.where(outliers)
    c = [columns[x] for x in c]
    c2 = list(set(c))
    
    mapper = { i : [r[j] for j in range(len(c)) if i == c[j]] for i in c2  }
                
    
    return mapper

def locate_nans(df , columns):
    mapper = {}
    for col in columns :
        mapper[col] = list(np.where(df[col].isna())[0])
    
    return mapper


def intersect_dicos(dic1, dic2):
    for k in dic1 :
        if k not in dic2:
            dic2[k] = []
    mapper = {}
    for k in dic1:
        mapper[k] = list(set(dic1[k]) & set(dic2[k]))

        
    
    return mapper


def show_outliers_zscore(df,mapper):
    """
    Cette fonction permet de montrer les valeurs aberrantes par colonne.
    Elle prend comme paramètres le df et le dictionnaire des index.
    Elle renvoi un dictionnaire qui contient pour chaque colonne les valeurs aberrantes.
    """
    return { k : df.loc[mapper[k],k].tolist() for k in mapper }



def delete_outliers(df , mapper : Union[dict , list] ):
    """
    Cette fonction permet de supprimer les lignes qui contiennent des valeurs manquantes
    
    """
    if type(mapper) == dict :
        to_drop = list(set([x for k in mapper for x in mapper[k]]))
    else:
        to_drop = mapper
        
    df.reset_index(drop = True , inplace= True)
    
    return df.drop(index = to_drop)

def treat_outliers(df ,mapper : Union[dict , list] ):
    """
    Cette fonction permet de créer une variable indicatrice <outliers> qui localise les outliers
    
    """
    if type(mapper) == dict :
        to_treat = list(set([x for k in mapper for x in mapper[k]]))
    else:
        to_treat = mapper
        
    df.reset_index(drop = True , inplace= True)    
    
    condition = [x in to_treat for x in df.index.tolist()]
    df['outliers'] = np.where(condition , 1 , 0)
    
    return df

def outliers_processing(data ,type_locate='dbscan' , treat_or_delete='treat',\
                        columns = ['ead','MT_INI_FIN_','mt_appo_'] ,):
    """
    Cette fonction permet d'appliquer le traitement des outliers complet
    en utilisant les fonctions prédéfinies en haut

    """
    if type_locate =='dbscan':
        mapper = locate_outliers_dbscan(data, columns = columns )
    else:
        mapper = locate_outliers_zscore(data, columns = columns )
    
    if treat_or_delete == 'treat':
        data = treat_outliers(data , mapper )
    else:
        data = delete_outliers(data , mapper )
    
    return data