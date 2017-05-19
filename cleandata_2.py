# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 21:43:51 2017

@author: russellding


The code can be used to generate the clean data required by the 
"""

import numpy as np
import pandas as pd
import datetime
import dateutil
from dateutil.relativedelta import relativedelta

import sys
sys.path.append('C:\\project\\code')
import setting


def slice_pivot_data(m_data, today, monthdelta):
    """Slice a time period of data
    Args:
        m_data(pd.DataFrame): the "member" data, index is date, column is PERMNO
        date_col(string): the name of the date column
        today(datetime.datetime)
        monthdelta(int): if 1, means forward looking of one month;
            if -1, means look back one month
    Returns:
        sliced_data(pd.DataFrame): the data within the range of today and pastdate or future date
    """
    past_date = None
    if monthdelta < 0:
        past_date = today + dateutil.relativedelta.relativedelta(months=monthdelta)
        past_date = past_date.replace(day=1)
        future_date = today
    else:
        past_date = today
        future_date = today + dateutil.relativedelta.relativedelta(months=monthdelta)
        future_date = future_date.replace(day=1)
    index = (m_data.index >= past_date) &  (m_data.index < future_date)
    
    # a copy
    sliced_data = m_data.ix[index]
    
    return sliced_data


def sift_na(table, threshold):
    names = np.sum(table.isnull(), axis=0) > table.shape[0]*threshold
    return names.index.values[(names.values==1)]
                          
    
def clean_data(year_time, month_time, m_price, m_return, m_volume, m_status):
    '''
    The function trims out the 
    '''
    price_past_year = slice_pivot_data(m_price, datetime.datetime(year_time,month_time,1), -12)
    price_past_month = slice_pivot_data(m_price, datetime.datetime(year_time,month_time,1), -1)
    names = price_past_month.min(axis=0) < 2.5
    stock_low_price = names.index.values[(names.values==True)]
    # obtain the list of stocks with NA in price for more than 20% in a year/month
    stock_price_na_year = sift_na(price_past_year, 0.2)
    stock_price_na_month = sift_na(price_past_month, 0.2)


    return_past_year = slice_pivot_data(m_return, datetime.datetime(year_time,month_time,1), -12)
    return_past_month = slice_pivot_data(m_return, datetime.datetime(year_time,month_time,1), -1)
    #obtain the list of stocks with NA in return for more than 20% in a year/month
    stock_return_na_year = sift_na(return_past_year, 0.2)
    stock_return_na_month = sift_na(return_past_month, 0.2)


    volume_past_month = slice_pivot_data(m_volume, datetime.datetime(year_time,month_time,1), -1)
    # filter out stocks with 0 in volume for more than 20% in a month
    stock_volume_0_month=volume_past_month.columns[(np.sum(volume_past_month == 0.0, axis=0)
                                            > volume_past_month.shape[0]*0.23)]

    status_past_year = slice_pivot_data(m_status, datetime.datetime(year_time,month_time,1), -12)
    status_past_month = slice_pivot_data(m_status, datetime.datetime(year_time,month_time,1), -1)
    stock_status_na_year = sift_na(status_past_year, 0.2)
    stock_status_na_month = sift_na(status_past_month, 0.2)

    na_year=set(list(stock_status_na_year)+list(stock_price_na_year)+list(stock_return_na_year))
    na_month=set(list(stock_status_na_month)+list(stock_price_na_month)+list(stock_return_na_month)
                +list(stock_volume_0_month))
    
    na_all=na_year.union(na_month)
    na_all=na_all.union(set(list(stock_low_price)))

    # na_all = set(stock_price_na_month)
    stk_names = price_past_month.columns
    stk_names=stk_names[~stk_names.isin(list(na_all))]
    
    return np.array(stk_names.unique())


def create_invest_universe(inpath, outpath, start_date, end_date):
    """Obtain the investable universe (in permno) each month, and 
    save them as a dictionay with the datetime as key
    
    """
    print('Read-in...')
    data = pd.read_hdf(inpath, key='data')
    print('Read done...')
    #Convert 'date' colum from string to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    
    #covert return series from string to numbers
    data['RET']=pd.to_numeric(data['RET'],errors='coerce')
    data['PRC']=pd.to_numeric(data['PRC'],errors='coerce')
    
    #Check Trading Stauts
    data['T_Status']=data['TRDSTAT'].apply(
        lambda x: 1.0 if x=='A' else None)

    price_temp = data.pivot_table(values='PRC', index='date', columns='PERMNO')
    # return_temp = None
    # volume_temp = None
    # status_temp = None
    return_temp = data.pivot_table(values='RET', index='date', columns='PERMNO')
    volume_temp = data.pivot_table(values='VOL', index='date', columns='PERMNO')
    status_temp = data.pivot_table(values='T_Status', index='date', columns='PERMNO',aggfunc='first')
    print('Data preclean done...')

    del data
    
    permno_record=dict()
    # the start date of next month
    nextmonth_start = start_date
    
    while nextmonth_start<=end_date:
        if nextmonth_start.month == 1:
            print(nextmonth_start.year)
        # the investable universe for next month
        invest_universe = clean_data(nextmonth_start.year,nextmonth_start.month, 
            m_price=price_temp, m_return=return_temp, 
            m_volume=volume_temp, m_status=status_temp)
        permno_record.update({nextmonth_start: invest_universe})
        nextmonth_start = nextmonth_start+relativedelta(months=1)
    
    np.save(outpath, permno_record)
    print('Data saved')    
    
    
if __name__=='__main__':
    create_invest_universe(setting.datapath+'full_data_0218_washed.h5',
                           setting.datapath+'investable_universe_0221.npy',
                           datetime.datetime(1962, 1, 1),
                           datetime.datetime(2016, 12, 31))
