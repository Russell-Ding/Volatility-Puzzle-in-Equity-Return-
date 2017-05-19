import pandas as pd
import datetime
import numpy as np

import setting
from portfolio_analysis import  review_pf_return
import io_tool

# Read the pre-defined signal and calculate return based on that

def read_in_signal(name):
    """Read in signal from data prepared"""
    m_data = pd.read_csv(setting.datapath_prepared+name+'.csv')
    m_data['date'] = pd.to_datetime(m_data['date'])
    m_data_pivot =pd.pivot_table(m_data,columns='PERMNO',index='date',values=name)
    return m_data_pivot



return_pivot=read_in_signal('M_RET')
pf_start = datetime.datetime(1965, 2, 1)
pf_end = datetime.datetime(2016, 12, 31)   
#Amend the file name if necessary
pfweight = io_tool.load_pickle_obj(setting.datapath_pf_weight, 
	'pfweight_FF_M_VOL_lag_False1965-02-2016-12')
return_series=review_pf_return(pf_start, pf_end, pfweight, return_pivot)

#print(return_series.head())

return_series.to_excel('return.xlsx')

return_series['1-5']=return_series['1']-return_series['5']

return_series['1-5'].cumsum()
