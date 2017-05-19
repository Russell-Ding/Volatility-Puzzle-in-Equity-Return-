"""Functions used to form portfolios"""
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta

import sys
sys.path.append('C:\\project\\code')
import setting
import io_tool



def add_lag_signal(signal_current, signal_name):
    """Add the lag of signal to current dataframe
    Args:
        signal_current(pd.DataFrame): with column of date, PERMNO, and signal_name;
            the order of column names may be different
        signa_name(string)
    Return:
        signal_merge(pd.DataFrame): with columns of date, PERMNO, signal, signal_lag;
            the order of column names may be different
    """
    signal_lag = signal_current.pivot_table(index='date', 
                                            columns='PERMNO', values=signal_name).shift(1)
    signal_lag = signal_lag.unstack().reset_index()
    signal_lag.columns = ['PERMNO', 'date', signal_name+'_lag']
    signal_merge = pd.merge(signal_current, signal_lag,
                            left_on=['PERMNO', 'date'], right_on=['PERMNO', 'date'])
    return signal_merge


def merge_signal_df(signal_list):
    """To merge a list of pandas dataframe into one
    Args:
        signal_list(list of pd.DataFrame): each dataframe has columns of 
            'date', 'PERMNO', and its own signal name
    Return:
    	signal_df(pd.DataFrame): with columns of 'date',
    	 'PERMNO', 'signal1', 'signal2'
    """
    signal_df = pd.merge(signal_list[0], signal_list[1],
                         left_on=['PERMNO', 'date'], right_on=['PERMNO', 'date'])
    
    for i in range(2, len(signal_list)):
        signal_df = pd.merge(signal_df, signal_list[i],
                             left_on=['PERMNO', 'date'], right_on=['PERMNO', 'date'])
        
    return signal_df
    

def extract_daily_signal(signal_list, signal_col, datestr, investable_name):
    """The function to extract and organize multiple daily signals for calculation
    Args:
        signal_list(list of pandas.DataFrame): each dataframe has 
            index of 'date', and columns of 'PERMNO', and value of a type of signal
        signal_col(list of string): the list of names of the signals
        datestr(string): the date of the month, e.g. '1995-10' 
        investable_name(np.array): the array element is the PERMNO of the stocks
    Return:
        aggregate_info(pd.DataFrame): index is the names of the signals,
           columns are the permno of the signals
    """
    current_signal = []
    for i in range(len(signal_col)):
        current_signal.append(signal_list[i].ix[datestr].to_frame())
        current_signal[i].columns = [signal_col[i]]

    aggregate_info = pd.concat(current_signal, axis=1)      
    aggregate_info = aggregate_info.dropna(axis=0)

    chosen_name = set(aggregate_info.index).intersection(set(investable_name))
    aggregate_info = aggregate_info.loc[chosen_name, :]
    
    return aggregate_info


def split_portfolio(signal_series, nyse_list):
    # from small to large sorted portfolio
    # Intriduce NYSE stocks as thresholds to split the whole stock universe
    
    nyse_signal = signal_series[signal_series.index.isin(nyse_list)]
    
    # Obtain the threshold    
    p80 = nyse_signal.quantile(q=0.8)
    p60 = nyse_signal.quantile(q=0.6)
    p40 = nyse_signal.quantile(q=0.4)
    p20 = nyse_signal.quantile(q=0.2)

    # Obtain the five portfolios
    l1_stock = signal_series[signal_series < p20]
    l2_stock = signal_series[(signal_series < p40) & (signal_series >= p20)]
    l3_stock = signal_series[(signal_series < p60) & (signal_series >= p40)]
    l4_stock = signal_series[(signal_series < p80) & (signal_series >= p60)]
    l5_stock = signal_series[signal_series >= p80]

    return [l1_stock.index.values, l2_stock.index.values, \
            l3_stock.index.values, l4_stock.index.values, l5_stock.index.values]


def get_pf_weight(signal_df, signal_col, mktcap_col, investable_universe, nyse_stk,
                  startdate, enddate, breakbyNYSE=True,value_weight=True):
    """
    Obtain the historical weights of portfolio according to given signals
    Args:
        signal_df(pd.DataFrame): with columns of 'date', 'PERMNO', 'signal'
        signal_col(str or list of str): the names of the signals
            mktcap_col(str or [str]): the column name of market cap info in the signal_df
        investable_universe(dictionary of {datetime.datetime: np.array}): np.array
            records the PERMNO of investable stocks that month
        nyse_stk(np.array): the PERMNO of stock with NYSE as primary exchange
        startdate(datetime.datetime): the month we start investing (assumed from the 
		beginning of the month)
        enddate(datetime.datetime): we only invest before the start of 
            the month of enddate
        breakbyNYSE(bool): whether use NYSE stocks as quantile portfolio's breakpoints
        value_weight(bool): whether the portfolio is value weighted, otherwise
            equal-weighted
    Return:
        monthly_portfolio(dictionary {datetime.datetime:  list of pd.Series})
        each pd.Series has PERMNO as index and stock weight as values
        the datetime.datettime object is set at the start of the month
    Note:
        when the length of signal_col is equal to 1, we would do single sort
         of portfolio (5 quintiles), from small to large
         when the length of signal_col is larger than 1, we would use the first
         two signals to do double sorts of portfolio (5x5)
    """
    if not isinstance(signal_col, list):
        signal_col = [signal_col]
    if not isinstance(mktcap_col, list):
        mktcap_col = [mktcap_col]

    startdate.replace(day = 1)
    enddate.replace(day=1)

    # --- extract each signal ---
    signal_list = []
    for i in range(len(signal_col)):
        signal_list.append(signal_df.pivot_table(index='date', columns='PERMNO', values=signal_col[i]))
        
    # names of data we need to extract daily
    col_needed = signal_col
    # if value weighted and the original signal_col doesn't contain last month mktcap, append it
    if value_weight and (mktcap_col[0] not in signal_col):
        signal_list.append(signal_df.pivot_table(index='date', columns='PERMNO', values=mktcap_col[0]))
        col_needed = signal_col + mktcap_col

    # --- portfolio formation ---
    current = startdate
    monthly_portfolio = {}

    while current < enddate:

        # --- get signal ---
        datestr = str(current)[:7]
        name = investable_universe[current]
        aggregate_info = extract_daily_signal(signal_list, col_needed,
                                              datestr, name)
        
        # --- get the stocks in each portfolio ---
        if breakbyNYSE:
            breakpoint_stk = list(set(nyse_stk).intersection(set(aggregate_info.index.values)))
        else:
            breakpoint_stk = list(aggregate_info.index.values)
        five_portfolio_permno = split_portfolio(aggregate_info[signal_col[0]],
                                                breakpoint_stk)
        
        # single sort
        if len(signal_col) == 1:  
            port_list = five_portfolio_permno
        # double sort
        else:
            five_portfolio_permno2 = split_portfolio(aggregate_info[signal_col[1]],
                                                     aggregate_info.index.values)
            port_list = []
            for i in range(len(five_portfolio_permno)):
                for j in range(len(five_portfolio_permno2)):
                    port_list.append(list(
                            set(five_portfolio_permno[i]).intersection(set(five_portfolio_permno2[j]))
                        ))
        
        # --- obtain stk weight ---
        pf_weight = []
        for i in range(len(port_list)):
            stk = aggregate_info.ix[port_list[i]]
            # if the portfolio is empty, the result would be Series([])
            if value_weight:
                pf_weight.append(pd.Series(stk[mktcap_col[0]]/np.sum(stk[mktcap_col[0]]),
                 name='weight'))
            # equal weight
            else:
                pf_weight.append(pd.Series(1.0/stk.shape[0], index=stk.index, name='weight'))

        monthly_portfolio[current] = pf_weight
        current += dateutil.relativedelta.relativedelta(months=1)
    return monthly_portfolio


def form_portfolio(files, investable_universe_npy, nyse_permno_xlsx, 
    list_execute=[['M_VOL_lag']], nyse_break=True,
    startdate=datetime.datetime(1964, 1, 1), enddate=datetime.datetime(2000, 1, 1),
    value_weight=True):
    
    signal_df_list = []
    
    for i in range(len(files)):
        signal_df_list.append(pd.read_csv(setting.datapath_prepared+files[i]+'.csv'))
        signal_df_list[i] = add_lag_signal(signal_df_list[i], files[i])
        # add an additional lag
        signal_df_list[i] = add_lag_signal(signal_df_list[i], files[i]+'_lag')

    
    signal_df = merge_signal_df(signal_df_list)
    
    investable_universe=np.load(setting.datapath+investable_universe_npy).item()
    nyse_stk = pd.read_excel(setting.datapath+nyse_permno_xlsx)
    nyse_stk = np.squeeze(nyse_stk.values)

    for i in range(len(list_execute)):
        history_weight = get_pf_weight(signal_df, list_execute[i], 'MKTCAP_M_lag',
            investable_universe, nyse_stk, startdate=startdate,
            enddate=enddate, breakbyNYSE=nyse_break, value_weight=value_weight)
        # get the name of signals used
        name = str()
        for j in range(len(list_execute[i])):
            name = name +'_'+ list_execute[i][j]
        if value_weight:
            file_name = 'pfweight'+name+'_'+str(nyse_break)+str(startdate)[:7]+'-'+str(enddate)[:7]
        else:
            file_name = 'pfweight'+name+'_'+str(nyse_break)+str(startdate)[:7]+'-'+str(enddate)[:7]+'equal'
        io_tool.save_pickle_obj(history_weight, setting.datapath_pf_weight, 
            file_name)
        print('Portfolio formed for '+name)


if __name__ == '__main__':
    # files =  ['M_RET', 'MKTCAP_M', 'M_VOL', 
    # 'L_VOL_12M', 'L_VOL_18M', 'L_VOL_24M']
    files = ['M_VOL', 'MKTCAP_M']

    form_portfolio(files, investable_universe_npy='investable_universe_test.npy',
        nyse_permno_xlsx='NYSE PERMNO.xlsx', list_execute=[['M_VOL_lag']],
        nyse_break=True)
