# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:03:38 2017

@author: qiuyi.chen
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib as plt

data = pd.read_csv('../input/shortsell.csv', parse_dates = [0])
data = data[data['MarketArea'].str.contains('US Equity')]

P_C_mapping = pd.read_csv('../input/PCmapping.csv', parse_dates = [1])
data.rename(columns = {'CUSIP':'CUSIP_9', 'DataDate':'date'}, inplace = True)
data['CUSIP'] = data['CUSIP_9'].str[:8]
data = data.merge(P_C_mapping, on = ['date','CUSIP'], how = 'right')

#Clean short data
short_data = data.dropna(subset = ['LendableQuantity'])
short_data['QuantityOnLoan'].fillna(0, inplace = True)
short_data = short_data[short_data['LendableQuantity'] != 0]
short_data['DS_ratio'] = short_data['QuantityOnLoan']/short_data['LendableQuantity']

#Construct mapping table: date, cusip_8, cusip_9, permno. P_C_mapping is from CRSP data base for
#CUSIP vs. PERMNO
na = short_data[short_data['CUSIP'].isnull()]
short_data = short_data[~short_data['CUSIP'].isnull()]
short_data = short_data[~short_data['DS_ratio'].isnull()]

#Utilization time series:
def get_short_s(signal, short_data):
    df = short_data.ix[:,['date','PERMNO', signal]]
    df.set_index('date', inplace = True)
    df = df.groupby('PERMNO').apply(lambda x: x.groupby(pd.TimeGrouper('M')).mean())
    df.dropna(inplace = True)
    df.index = [x[1] for x in np.array(df.index)]
    df = df.to_period('M')
    df.reset_index(inplace = True)
    df.columns = ['date','PERMNO',signal]
    df.to_csv('../input/from_prepare_data/'+signal +'.csv', index = False)
    return df
    
UtilisationByQuantity = get_short_s('UtilisationByQuantity', data.dropna(subset = ['UtilisationByQuantity'])) 
DS_ratio = get_short_s('DS_ratio', short_data[short_data['DS_ratio'] <= 1])
Fee = get_short_s('IndicativeFee', data.dropna(subset = ['IndicativeFee']))


#output 8 digit cusip, in short data from markit, cusip is 9 digits, so we need to convert
'''
cusip_8 = P_C_mapping['CUSIP'].unique()
cusip_8 = cusip_8.astype(str)
thefile = open('cusip_crsp.txt', 'w')
for item in cusip_8:
  thefile.write("%s\n" % item)
thefile.close()
'''

