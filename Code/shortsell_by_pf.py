# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:12:21 2017

@author: qiuyi.chen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
plt.style.use('ggplot')


'''
Plot settings
'''
plt.rcParams['figure.figsize'] = (12,8)
# font
plt.rcParams['font.sans-serif']=['Fira Sans OT']
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'medium'

#load portfolio weighting dictionary
with open('..\input\portfolio_weight\pfweight_M_VOL_lag_False1963-07-2016-12.pkl', 'rb') as f:
    pfweight_TT_M_VOL = pickle.load(f)

with open('..\input\portfolio_weight\pfweight_FF_M_VOL_lag_False1963-07-2016-12.pkl', 'rb') as f:
    pfweight_FF_M_VOL = pickle.load(f)
    
fee = pd.read_csv('../input/from_prepare_data/IndicativeFee.csv', parse_dates = ['date'])

def pf_ss(fee, pfweight):
    #Cross sectional plotting
    result = []
    count_ = []
    for date,fee_date in fee.groupby('date'):
        pf = pfweight[date]
        for i in range(5):
            df = pf[i].to_frame().merge(fee_date, left_index = True, right_on = ['PERMNO'])
            pf_fee = df['IndicativeFee'].multiply(df['weight']).sum()
            result.append([date, pf_fee, i])
            count_.append([date, len(df), i])
    
    result_df = pd.DataFrame(result, columns = ['date','wa_fee','portfolio'])
    count_ = pd.DataFrame(count_, columns = ['date','count','portfolio'])
    return result_df, count_

shortfee_pf, count_ = pf_ss(fee, pfweight_TT_M_VOL)
shortfee_pf_idio, count_idio = pf_ss(fee, pfweight_FF_M_VOL)
shortfee_pf.to_csv('short fee by portfolio_total vol.csv')
shortfee_pf_idio.to_csv('short fee by portfolio_idio vol.csv')
#Plot
def plt_ss(df_ss, type_):
    colors_ = cm.rainbow(np.linspace(0, 1, 5))
    for pf, df in df_ss.groupby('portfolio'):
        plt.scatter(df.ix[:,0].as_matrix(), df.ix[:,1].as_matrix(), c = colors_[pf])
    
    plt.legend(['portfolio 1','portfolio 2','portfolio 3','portfolio 4','portfolio 5'], loc = 4)
    plt.ylabel('count')
    #plt.title('Weighted-Average Short Fee of each Portfolio ('+ type_ + ')')
    plt.savefig('..\output\Plot\shortfee_by_portfolio_' + type_ + '.png')
    plt.close('all')

plt_ss(shortfee_pf, 'total vol')
plt_ss(shortfee_pf_idio, 'idio vol')
plt_ss(count_, 'total vol short fee count')
fee_ts = fee.pivot(index = 'date', columns = 'PERMNO', values = 'IndicativeFee')

