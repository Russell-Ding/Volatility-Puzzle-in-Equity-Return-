# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:21:49 2017

@author: qiuyi.chen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

import datetime as dt

routine='c:\\project\\input\\'
month_ret = pd.read_csv(routine + 'Monthly Return CRSP.csv')
month_ret.loc[month_ret['RET'] == 'C','RET'] = 0
month_ret.loc[month_ret['RET'] == 'B','RET'] = 0
month_ret.dropna(inplace = True)
month_ret['RET'] = pd.to_numeric(month_ret['RET'])

month_ret.sort_values(by = 'date')
