# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:27:54 2017

@author: russellding
"""

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import setting


'''
The code examine the turnover in the five portfolios

Input: portfolio, start and end date

Return: Dataframe of turnover
'''


def calculate_turnover(df1,df2):
    #print ('Entered Function')
    count=0
    total=len(df1.index.values)
    if total==0:
        return None
    for item in df1.index.values:
        if item in df2.index.values:
            count+=1
    return 1-float(count)/total

def Turnover(start_date,end_date,filename):
    #start_date=datetime.datetime(1964,1,1)
    #end_date=datetime.datetime(2016,12,1)
    
    #with open(setting.datapath_pf_weight+filename,'rb') as f:
    portfolio=pd.read_pickle(setting.datapath_pf_weight+filename)
    #print(investable_universe[datetime.datetime(1964,1,1)])
    times=pd.date_range(start=start_date+relativedelta(months=1),end=end_date,freq='MS')
    turn_over_rate=pd.DataFrame(index=times,columns=[1,2,3,4,5])
    
    for i in times:
        
        list_old=portfolio[pd.to_datetime(i-relativedelta(months=1))]
        list_new=portfolio[pd.to_datetime(i)]
        for port in range(1,6):
            turn_over_rate.loc[i,port]=calculate_turnover(list_old[port-1],list_new[port-1])
    turn_over_rate.to_excel(setting.outputpath+'Turnover Rate of '+filename[0:filename.find('.')]+'.xlsx')
    return None

def Turnover_rate(portfolio, start, end):
    """functions used for analysis
    ：portfolio(the dictionary of portfolio weight, with datetime.datetime as keys)
    :start(datetime.datetime)
    :end(datetime.datetime)
    """
    start_date = np.min(list(portfolio.keys()))
    start_date = np.max([start_date, start])
    end_date = np.max(list(portfolio.keys()))
    end_date = np.min([end, end_date])

    times=pd.date_range(start=start_date+relativedelta(months=1),end=end_date,freq='MS')
    num_port = len(portfolio[start_date])
    turn_over_rate=pd.DataFrame(index=times,columns=list(np.arange(num_port)+1))
    
    for i in times:
        list_old=portfolio[pd.to_datetime(i-relativedelta(months=1))]
        list_new=portfolio[pd.to_datetime(i)]
        for port in range(1,num_port+1):
            turn_over_rate.loc[i,port]=calculate_turnover(list_old[port-1],list_new[port-1])
    return turn_over_rate


def Turnover_fee(portfolio, start, end, fee):
    return Turnover_rate(portfolio, start, end).shift(-1) * fee


if __name__ == "__main__":
    Turnover(datetime.datetime(1964,1,1),datetime.datetime(2016,12,1),\
             'pfweight_FF_L_VOL_24W_lag_lag_False1963-07-2016-12.pkl')
