# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:20:28 2021

@author: yanis
"""

import pandas as pd
from typing import Union
import yfinance as yf



def get_stock(stock, start, end , value: Union[str , list], index_as_date=True):
    """
    Parameters
    ----------
    stock : str
        Stock abreviation
    start : str
        Start date in format yyyy-mm-dd
    end : str
        End date in format yyyy-mm-dd
    value : Union[str , list]
        Open, High, Low, Close, Adj Close or Volume
    index_as_date : bool
        Set True if you want dates as index and False if you don't want dates.

    Returns
    -------
    pd.DataFrame
        Dataframe with requested data.
    """
    
    serie = yf.download(stock , start = start , end = end)
    
    if index_as_date == True:
        serie = serie[value]
        serie.index = serie.index.to_period('d')
    else:
        serie = serie[value].reset_index(drop=True)
    

    return serie 

#########################################################################################


def get_multiple_stock(tickers, start, end, value: Union[str , list], index_as_date=True):
    """

    Parameters
    ----------
    tickers : list
        The list with the stocks wanted (as strings)
    start : str
        Start date in format yyyy-mm-dd
    end : str
        End date in format yyyy-mm-dd
    value : Union[str , list]
        Open, High, Low, Close, Adj Close or Volume
    index_as_date : str, optional
        dafaut True if you want dates as index and False if you don't want dates.

    Returns
    -------
    stocks_df : pd.Dataframe
        Dataframe with requested data.

    """
    list_stock=[]
    
    for ticker in tickers:
        stock_serie = get_stock(stock=ticker, start=start, end=end,  value=value, index_as_date=index_as_date)
        list_stock.append(stock_serie)
    
    stocks_df=pd.concat(list_stock, axis=1)
    stocks_df.columns = tickers
    
    return stocks_df

#########################################################################################

def get_returns(df_stocks):
    """
    
    Parameters
    ----------
    df_stocks : pd.DataFrame
        Dataframe with stocks value

    Returns
    -------
    returns_df : pd.DataFrame
        Dataframe with stocks returns

    """

    list_returns=[]
    
    for ticker in df_stocks.columns:
        
        serie_return= df_stocks[ticker].pct_change()
        list_returns.append(serie_return)
    
    returns_df=pd.concat(list_returns, axis=1)
    returns_df.columns = df_stocks.columns
    
    returns_df = returns_df.iloc[1: , :]
    
    return returns_df


#########################################################################################

def get_mu_and_sigma(df):
    '''
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the series of interest
    Returns
    -------
    df_param : pd.DataFrame
        A DataFrame with the mean et the standard deviation of all the columns

    '''
    
    list_mu=[]
    list_sigma=[]
    
    for i in df:
        
        mu=df[i].mean()
        sigma=df[i].std()
        
        list_mu.append(mu)
        list_sigma.append(sigma)
    
    list_param=list(zip(list_mu, list_sigma))
    df_param=pd.DataFrame(list_param).T
    df_param.columns = df.columns
        
    return df_param



