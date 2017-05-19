# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import scipy

from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import matplotlib.cm as cm
plt.style.use('ggplot')

from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

import datetime as dt
import pickle

'''
Plot settings
'''
plt.rcParams['figure.figsize'] = (15,10)
# font
plt.rcParams['font.sans-serif']=['Fira Sans OT']
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'small'

'''
'''

#load portfolio weighting dictionary
with open('..\input\portfolio_weight\pfweight_M_VOL_lag_False1963-07-2016-12.pkl', 'rb') as f:
    pfweight_TT_M_VOL = pickle.load(f)
 
with open('..\input\portfolio_weight\pfweight_FF_M_VOL_lag_False1963-07-2016-12.pkl', 'rb') as f:
    pfweight_FF_M_VOL = pickle.load(f)

## Total Vol    
#Read volatility and annualize it
tt_VOL_list = ['M_VOL','L_VOL_12M','L_VOL_18M','L_VOL_24M','L_VOL_24W','L_VOL_48W','L_VOL_72W']
tt_VOL = {}

ff_VOL_list = ['FF_M_VOL','FF_L_VOL_12M','FF_L_VOL_18M','FF_L_VOL_24M','FF_L_VOL_24W','FF_L_VOL_48W','FF_L_VOL_72W']
ff_VOL = {}



#Annualize short term volatility (average monthly vol over the past 12 months)
def read_vol(list_):
    dict_ = {}
    for item in list_:
        df = pd.read_csv('../input/from_prepare_data/' +item +'.csv', usecols = ['date','PERMNO',item],parse_dates = ['date'])
        if item[-1] == 'L':
             df[item] = df[item]*np.sqrt(252)             
        elif item[-1] == 'M':
             df[item] = df[item]*np.sqrt(12)
        elif item[-1] == 'W':
             df[item] = df[item]*np.sqrt(52)
        dict_[item] = df
    return dict_

tt_VOL = read_vol(tt_VOL_list)
ff_VOL = read_vol(ff_VOL_list)

#define regression function
def reggr(df, lt_vol):
    
    df = df.dropna()
    beta = np.nan
    alpha = np.nan
    X = df.ix[:,2].to_frame()
    X = sm.add_constant(X)
    Y = df[lt_vol].to_frame()
    try:
        model = sm.OLS(Y,X,hasconst = True)
        results = model.fit()
    except:
        print (lt_vol + 'regression error')
    else:
        if len(df) > 10:
            alpha = results.params[0]
            alpha_t = results.tvalues[0]
            beta = results.params[1]
            beta_t = results.tvalues[1]
    return [alpha, alpha_t, beta, beta_t]

## Function to plot relationship between short term and long term vol
#interesting_keys = (keys[-1], keys[-2])
#subdict = {x: pfweight[x] for x in interesting_keys if x in pfweight}

def xsec(lt_vol_name, lt_vol, M_VOL, pfweight):
    #Cross sectional plotting
    keys = list(pfweight.keys())
    keys.sort()
    result = []
    agg = M_VOL.merge(lt_vol, on = ['date','PERMNO'])
    
    agg = agg[(agg['date'] >= keys[0]) & (agg['date'] <= keys[-1])]

    for date, df in agg.groupby('date'):
        pf = pfweight[date]
        for i in range(5):
            pf_vol = df[df['PERMNO'].isin(pf[i].index)]
            corr = pf_vol[lt_vol_name].corr(pf_vol.ix[:,2])
            result.append(np.concatenate(([date,corr, i], reggr(pf_vol,lt_vol_name))))
        print (date)
    
    result_df = pd.DataFrame(result, columns = ['date','correlation','portfolio','alpha','alpha_t','beta','beta_t'])
    #plt_corr(result_df, lt_vol_name)
    #plt_beta(result_df, lt_vol_name)
    return result_df       
      
#plot correlation
def plt_corr(result_df, lt_vol_name):
    for i, df in result_df.groupby('portfolio'):
        plt.plot(df['date'],df['correlation'], label = 'portfolio '+ str(i+1))
    
    plt.title('Correlation between ' + lt_vol_name + ' and short term vol')
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base = 5))
    plt.gcf().autofmt_xdate()
    
    plt.savefig('..\output\Plot\Correlation '+lt_vol_name +'.png')
    plt.close('all')

#plot beta history
def plt_beta(result_df, lt_vol_name):
    plt.close('all')
    for i, df in result_df.groupby('portfolio'):
        plt.plot(df['date'],df['beta'], label = 'portfolio '+ str(i+1))
    
    plt.title('Beta of '+ lt_vol_name + ' regression on short term vol')
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base = 5))
    plt.gcf().autofmt_xdate()
    
    plt.savefig('..\output\Plot\Beta '+ lt_vol_name + '.png')
    plt.close('all')
    
#Total long/short vol  
result_TT_48W = xsec('L_VOL_48W', tt_VOL['L_VOL_48W'], tt_VOL['M_VOL'], pfweight_TT_M_VOL) 
result_FF_48W = xsec('FF_L_VOL_48W', ff_VOL['FF_L_VOL_48W'], ff_VOL['FF_M_VOL'], pfweight_FF_M_VOL)

result_TT_24W = xsec('L_VOL_24W', tt_VOL['L_VOL_24W'], tt_VOL['M_VOL'], pfweight_TT_M_VOL)  
result_FF_24W = xsec('FF_L_VOL_24W', ff_VOL['FF_L_VOL_24W'], ff_VOL['FF_M_VOL'], pfweight_FF_M_VOL)

result_TT_72W = xsec('L_VOL_72W', tt_VOL['L_VOL_72W'], tt_VOL['M_VOL'], pfweight_TT_M_VOL)  
result_FF_72W = xsec('FF_L_VOL_72W', ff_VOL['FF_L_VOL_72W'], ff_VOL['FF_M_VOL'], pfweight_FF_M_VOL)

result_TT_12M = xsec('L_VOL_12M', tt_VOL['L_VOL_12M'], tt_VOL['M_VOL'], pfweight_TT_M_VOL)  
result_FF_12M = xsec('FF_L_VOL_12M', ff_VOL['FF_L_VOL_12M'], ff_VOL['FF_M_VOL'], pfweight_FF_M_VOL)

result_TT_18M = xsec('L_VOL_18M', tt_VOL['L_VOL_18M'], tt_VOL['M_VOL'], pfweight_TT_M_VOL)  
result_FF_18M = xsec('FF_L_VOL_18M', ff_VOL['FF_L_VOL_18M'], ff_VOL['FF_M_VOL'], pfweight_FF_M_VOL)

result_TT_24M = xsec('L_VOL_24M', tt_VOL['L_VOL_24M'], tt_VOL['M_VOL'], pfweight_TT_M_VOL)  
result_FF_24M = xsec('FF_L_VOL_24M', ff_VOL['FF_L_VOL_24M'], ff_VOL['FF_M_VOL'], pfweight_FF_M_VOL)

for item in result_list:
    print (item, globals()[item].correlation.min())
    

stat = []
diff = []
for i in range(6):
    pd1 = globals()[result_list[2*i]]
    pd2 = globals()[result_list[2*i+1]]
    stat.append([result_list[2*i],pd1.correlation.median(), pd1.correlation.mean(), pd1.correlation.std()])
    stat.append([result_list[2*i+1],pd2.correlation.median(), pd2.correlation.mean(), pd2.correlation.std()])
    p_value = scipy.stats.ttest_ind(pd1.correlation, pd2.correlation, axis=0, equal_var=False)[0]
    diff.append([result_list[2*i] +'-'+ result_list[2*i+1], pd1.correlation.mean() - pd2.correlation.mean(), p_value])
    
stat = pd.DataFrame(stat, columns = ['long term vol', 'median', 'mean', 'std'])
diff = pd.DataFrame(diff, columns = ['vol','mean diff', 't_stat/null is identical mean'])
stat.to_csv('..\output\corr_stat.csv')
diff.to_csv('..\output\p_value_corr_diff.csv')

## Plot histogram
fig, ax = plt.subplots(3,2)
ax = ax.ravel()
result_list = ['result_TT_24W', 'result_FF_24W','result_TT_12M', 'result_FF_12M','result_TT_48W', 'result_FF_48W', 'result_TT_18M', 'result_FF_18M','result_TT_72W', 'result_FF_72W','result_TT_24M', 'result_FF_24M']

for i in range(6):
    ax[i].hist(globals()[result_list[2*i]].correlation, alpha = 0.3, label = result_list[2*i], bins = 100)
    ax[i].hist(globals()[result_list[2*i + 1]].correlation, alpha = 0.3, label = result_list[2*i + 1], bins = 100)
    ax[i].set_title(result_list[2*i][-3:] + ' histogram')
    ax[i].legend(loc = 'best')
    ax[i].set_xlim([-0.1,1])
    ax[i].set_ylim([0,150])
    ax[i].set_xlabel('Correlation')

plt.tight_layout()
plt.savefig('..\output\Plot\correlation histogram.png',dpi=1000)

## Plot time series of correlation
fig, ax = plt.subplots(3,2)
ax = ax.ravel()
result_list = ['result_TT_24W', 'result_FF_24W','result_TT_12M', 'result_FF_12M','result_TT_48W', 'result_FF_48W', 'result_TT_18M', 'result_FF_18M','result_TT_72W', 'result_FF_72W','result_TT_24M', 'result_FF_24M']

for i in range(6):
    x_axis = globals()[result_list[i*2]].date
    ax[i].plot(globals()[result_list[2*i]].correlation, label = result_list[2*i])
    ax[i].plot(globals()[result_list[2*i + 1]].correlation, label = result_list[2*i + 1])
    ax[i].set_title(result_list[2*i][-3:] + ' Time Series')
    ax[i].legend(loc = 'best')
    #ax[i].set_xlim([-0.1,1])
    #ax[i].set_ylim([0,150])
    ax[i].set_xlabel('Correlation Time Series')
    

plt.tight_layout()
plt.savefig('..\output\Plot\correlation time series.png',dpi=1000)


## Plot beta/correlation time series
fig, ax = plt.subplots(3,2)
ax = ax.ravel()
result_list = ['result_TT_24W', 'result_FF_24W','result_TT_12M', 'result_FF_12M','result_TT_48W', 'result_FF_48W', 'result_TT_18M', 'result_FF_18M','result_TT_72W', 'result_FF_72W','result_TT_24M', 'result_FF_24M']

for i in range(6):
    x_axis = globals()[result_list[i*2]].date
    ax[i].plot(x_axis,globals()[result_list[i*2]].beta.rolling(window = 5).mean(), label = result_list[i*2][-6:-4])
    ax[i].plot(x_axis,globals()[result_list[i*2+1]].beta.rolling(window = 5).mean(), label = result_list[i*2+1][-6:-4])
    ax[i].set_title(result_list[i*2][-3:])
    ax[i].legend(loc = 1)
    #ax[i].set_xlim([-0.1,1])
    ax[i].set_ylim([0, 0.6])
    ax[i].set_ylabel('Beta')

plt.tight_layout()
plt.savefig('..\output\Plot\\beta.png',dpi=1000)

#Plot mean of long term vol for each portfolio
def plot_lt_vol_mean(pfweight, M_VOL, lt_vol):
    
    colors_ = cm.rainbow(np.linspace(0, 1, 5))
    keys = list(pfweight.keys())
    keys.sort()
    agg = M_VOL.ix[:,['date','PERMNO','M_VOL']].merge(globals()[lt_vol], on = ['date','PERMNO'])    
    agg = agg[(agg['date'] >= keys[0]) & (agg['date'] <= keys[-1])]
    
    for date, df in agg.groupby('date'):
        pf = pfweight[date]
        for i in range(5):
            pf_vol = df[df['PERMNO'].isin(pf[i].index)]
            plt.scatter(date, pf_vol[lt_vol].mean(), c = colors_[i])
    
    plt.show()
    plt.legend(['portfolio 1','portfolio 2','portfolio 3','portfolio 4','portfolio 5'])
    plt.savefig('Average' + lt_vol + ' for each portfolio')

#scatter plot long term vol based on portfolio
plot_lt_vol_mean(pfweight_TT_M_VOL, M_VOL, lt_vol)


'''
def plot_lt_vol_scatter(pfweight, M_VOL, lt_vol):
    
    colors_ = cm.rainbow(np.linspace(0, 1, 5))
    keys = list(pfweight.keys())
    keys.sort()
    agg = M_VOL.ix[:,['date','PERMNO','M_VOL']].merge(globals()[lt_vol], on = ['date','PERMNO'])    
    agg = agg[(agg['date'] >= keys[0]) & (agg['date'] <= keys[-1])]
    
    for date, df in agg.groupby('date'):
        pf = pfweight[date]
        for i in range(5):
            pf_vol = df[df['PERMNO'].isin(pf[i].index)]
            plt.scatter([date]*len(pf_vol), pf_vol[lt_vol], c = colors_[i])
    
    plt.show()

plot_lt_vol_scatter(pfweight, M_VOL, lt_vol)
'''

    ''' 
    X_sec_ols = pd.DataFrame(result, columns = ['date','beta','correlation'])
    X_sec_ols['date'] =[dt.datetime.strptime(d,'%Y-%m').date() for d in X_sec_ols['date'].values]
    
    
    plt.subplot(211)
    plt.plot(X_sec_ols['date'],X_sec_ols['beta'])
    plt.title('Cross Sectional Regression Beta (' + lt_vol + ' vs. short term)')
    
    plt.subplot(212)
    plt.plot(X_sec_ols['date'],X_sec_ols['correlation'],'r')
    plt.title('Correlation between '+ lt_vol + ' and short term vol')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base = 5))
    plt.gcf().autofmt_xdate()
    plt.show()
    
    plt.savefig('c:\\project\\output\\'+'cross sectional' + lt_vol + '.png')
    
    #Time series regression for each permno
    result = []
    for PERMNO, df in agg.groupby('PERMNO'):
        corr = df[lt_vol].corr(df['M_VOL'])
        result.append([PERMNO, reggr(df), corr])
    
    ts_ols = pd.DataFrame(result, columns = ['PERMNO','beta','correlation'])
    M_VOL_level = M_VOL.groupby('PERMNO')['M_VOL'].mean()
    ts_ols = pd.merge(M_VOL_level.to_frame().reset_index(), ts_ols, on = 'PERMNO' )
    ts_ols.sort_values(by = 'M_VOL', inplace = True)
    ts_ols = ts_ols.dropna()
    ts_ols.reset_index(inplace = True, drop = True)
    
    plt.subplot(311)
    plt.scatter(ts_ols.index,ts_ols['beta'])
    plt.title('Time series regression by Permno ('+ lt_vol + ' vs. short term)')
    
    plt.subplot(312)
    plt.scatter(ts_ols.index,ts_ols['correlation'])
    plt.plot(pd.rolling_mean(ts_ols['correlation'],window = 2000))
    plt.title('Correlation between '+ lt_vol + ' and short term vol')
    
    plt.subplot(313)
    plt.plot(ts_ols['M_VOL'],'r')
    plt.title('Average monthly return volatility')
    plt.tight_layout()
    plt.show()
    
    plt.savefig('c:\\project\\output\\'+ 'time series regression' + lt_vol + '.png')
    
    plt.scatter(agg.M_VOL, agg[lt_vol])
    plt.xlim([0,15])
    plt.ylim([0,15])

'''
'''
3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
date_sel = agg.date.unique()[::100]
agg_test = agg[agg['date'].isin(date_sel)]

zs = range(len(agg_test.date.unique()))
verts = []

for date, df in agg_test.groupby('date'):
    xs = df.L_VOL_12M.values
    ys = df.M_VOL_A.values
    verts.append(list(zip(xs, ys)))
    
poly = PolyCollection(verts)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 1.5)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 10)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 2.5)

plt.show()
'''
