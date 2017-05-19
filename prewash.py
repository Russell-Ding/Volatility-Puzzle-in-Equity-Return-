# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:38:33 2017

We filter out the RET column of the CRSP stock daily data
when value is 'B'.


@author: shihong
"""

import numpy as np
import pandas as pd
# for importing personalized functions
import sys
sys.path.append('C:\\Project\code')
import setting


def prewash_data(data, name):
	"""
	prewash and save the data to pickle
	data(pd.DataFrame): from the CRSP
	name(str): the name of pickle file

	"""
	data.loc[:,'date'] = pd.to_datetime(data['date'], format='%Y%m%d')
	data = data.ix[data['RET'] != 'B']
	# 'C' is the first date of new listing, with no return, set it to zero
	data.loc[:, 'RET'] = data['RET'].apply(lambda x: 0.0 if x=='C' else x)
	data.loc[:, 'RET'] = pd.to_numeric(data['RET'])
	
	# negative sign in Price means that there is no closing price, and the price
	# is set as the negative of the average bid ask. So, we change it
	# to its absolute value
	data.loc[:, 'PRC'] = pd.to_numeric(data['PRC'])
	data.loc[:, 'PRC'] = np.abs(data['PRC'])

	# clean out 4 stocks with the same stock prices over time, 
	data=data.ix[~data['PERMNO'].isin([82217, 31377, 78372, 12274])]

	data = data.reset_index(drop=True)
	data.to_hdf(setting.datapath+name+'.h5', key='data')


if __name__ == '__main__':

	inpath = setting.datapath + 'full_data_processed.csv'
	#inpath = setting.datapath + 'NYSE 3000 data.csv'
	data = pd.read_csv(inpath)

	prewash_data(data, 'full_data_processed2')
