import pandas as pd
import numpy as np
import statsmodels.api as sm
import time


def _load_raw_data(raw_csv_filename='../input/full_data_processed.csv'):
    print('Start reading CSV...')
    data = pd.read_csv(
        raw_csv_filename,
        usecols=['date', 'RET', 'PRC', 'PERMNO', 'SHROUT'],
        dtype={
            'date': str,
            'RET': object,
            'PRC': float,
            'PERMNO': int,
            'SHROUT': float
        }
    )
    print('Done reading CSV...')

    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data['RET'] = pd.to_numeric(data['RET'], errors='coerce')
    data['PRC'] = pd.to_numeric(data['PRC'], errors='coerce')
    return data


def _get_daily_return(data):
    daily_return = data.pivot(index='date', columns='PERMNO', values='RET')
    return daily_return


def _get_weekly_return(daily_return):
    weekly_return = (1 + daily_return).resample('W').prod()  # return 1 if missing
    weekly_return = weekly_return.to_period('W')
    return weekly_return


def _get_monthly_return(daily_return):
    monthly_return = (1 + daily_return).resample('M').prod()  # return 1 if missing
    monthly_return = monthly_return.to_period('M')
    return monthly_return


def _get_daily_vol_one_month_window(daily_return):
    monthly_vol = daily_return.resample('M').std()  # NaN if there is no sample
    monthly_vol = monthly_vol.to_period('M')
    return monthly_vol


def _get_long_vol_from_weekly_return(weekly_return, nweeks=12):
    longvol = weekly_return.rolling(nweeks).std()
    longvol[longvol == 0] = np.NaN
    return longvol


def _get_long_vol_from_monthly_return(monthly_return, nmonths=12):
    longvol = monthly_return.rolling(nmonths).std()
    longvol[longvol == 0] = np.NaN
    return longvol


def _get_weekly_mktcap(data):
    mktcap = data[['date', 'PERMNO']]
    mktcap = mktcap.assign(MKTCAP=data['PRC'] * data['SHROUT'])
    mktcap = mktcap.pivot(index='date', columns='PERMNO', values='MKTCAP')
    mktcap = mktcap.resample('W').last()
    mktcap = mktcap.to_period('W')
    return mktcap


def _get_monthly_mktcap(data):
    mktcap = data[['date', 'PERMNO']]
    mktcap = mktcap.assign(MKTCAP=data['PRC'] * data['SHROUT'])
    mktcap = mktcap.pivot(index='date', columns='PERMNO', values='MKTCAP')
    mktcap = mktcap.resample('M').last()
    mktcap = mktcap.to_period('M')
    return mktcap


def main_total_vol(data):
    print('Total-Vol: Formatting data...')
    daily_return = _get_daily_return(data)
    daily_return.stack(dropna=True).to_csv('../input/from_prepare_data/D_RET.csv', header=['D_RET'])

    weekly_return = _get_weekly_return(daily_return)
    weekly_return.stack(dropna=True).to_csv('../input/from_prepare_data/W_RET.csv', header=['M_RET'])

    '''
    weekly_mktcap = _get_weekly_mktcap(data)
    weekly_mktcap.stack(dropna=True).to_csv('../input/from_prepare_data/MKTCAP_W.csv', header=['MKTCAP_W'])

    monthly_mktcap = _get_monthly_mktcap(data)
    monthly_mktcap.stack(dropna=True).to_csv('../input/from_prepare_data/MKTCAP_M.csv', header=['MKTCAP_M'])

    monthly_return = _get_monthly_return(daily_return)
    monthly_return.stack(dropna=True).to_csv('../input/from_prepare_data/M_RET.csv', header=['M_RET'])

    monthly_vol = _get_daily_vol_one_month_window(daily_return)
    monthly_vol.stack(dropna=True).to_csv('../input/from_prepare_data/M_VOL.csv', header=['M_VOL'])
    '''
    monthly_return = None

    long_vol_24w = _get_long_vol_from_weekly_return(weekly_return,24)
    long_vol_24w.stack(dropna=True).to_csv('../input/from_prepare_data/L_VOL_24W.csv', header=['L_VOL_24W'])

    long_vol_48w = _get_long_vol_from_weekly_return(weekly_return, 48)
    long_vol_48w.stack(dropna=True).to_csv('../input/from_prepare_data/L_VOL_48W.csv', header=['L_VOL_48W'])

    long_vol_72w = _get_long_vol_from_weekly_return(weekly_return, 72)
    long_vol_72w.stack(dropna=True).to_csv('../input/from_prepare_data/L_VOL_72W.csv', header=['L_VOL_72W'])

    '''
    long_vol_12m = _get_long_vol_from_monthly_return(monthly_return)
    long_vol_12m.stack(dropna=True).to_csv('../input/from_prepare_data/L_VOL_12M.csv', header=['L_VOL_12M'])

    long_vol_18m = _get_long_vol_from_monthly_return(monthly_return, 18)
    long_vol_18m.stack(dropna=True).to_csv('../input/from_prepare_data/L_VOL_18M.csv', header=['L_VOL_18M'])

    long_vol_24m = _get_long_vol_from_monthly_return(monthly_return, 24)
    long_vol_24m.stack(dropna=True).to_csv('../input/from_prepare_data/L_VOL_24M.csv', header=['L_VOL_24M'])
    '''

    return daily_return, weekly_return, monthly_return


def _ff_reg_short(joined_data_one_stock, mins_sample_size=10):
    '''
    :param joined_data_one_stock: [return, const, RF, MKTRF, HML, SMB]
    :param mins_sample_size:
    :return: std, alpha

    pandas issue: pd.DataFrame.rolling passes np.ndarray whereas
    groupby.apply passes DataFrame. This is inconsisent!
    '''
    y = joined_data_one_stock.iloc[:,0]
    if (~y.isnull()).sum() < mins_sample_size:
        return np.nan
    X = joined_data_one_stock.iloc[:,1:]
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()
    resid_var = np.sum(np.square(results.resid)) / (len(results.resid) - 4)
    resid_vol = np.sqrt(resid_var)
    return resid_vol


def _get_short_ff_vol(daily_data):
    num_permnos = daily_data.shape[1]
    counter = 1
    short_ff_vol = pd.DataFrame()
    for permno in daily_return:
        if counter%100 == 0:
            print(
                'Daily FF regression for PERMNO', permno, 
                '(', counter, '/', num_permnos, ')...'
            )
        joined_data = pd.concat(
            [
                daily_data[permno],
                daily_data[['const', 'MKT_RF', 'HML', 'SMB']]
            ],
            axis=1, 
            join='inner'
        )
        joined_data = joined_data.dropna()
        grouped = joined_data.groupby(pd.TimeGrouper(freq='M'))
        resid_vol = grouped.apply(_ff_reg_short)
        resid_vol = resid_vol.rename(permno)
        short_ff_vol = pd.concat([short_ff_vol, resid_vol], axis=1)
        counter += 1
    short_ff_vol.columns = short_ff_vol.columns.rename('PERMNO')
    short_ff_vol = short_ff_vol.to_period('M')
    return short_ff_vol


def main_ff_vol_short(daily_return):
    ff = pd.read_csv('../input/famafrench.csv')
    ff['date'] = pd.to_datetime(ff['date'], format='%Y%m%d')
    ff.set_index('date', inplace=True)
    ff.columns = ['MKT_RF', 'SMB', 'HML', 'RF', 'UMD']
    daily_data = pd.concat(
        [daily_return, ff[['MKT_RF', 'SMB', 'HML', 'RF']]], 
        axis=1, 
        join='inner'
        )
    daily_data['const'] = 1
    # get excessive returns
    daily_data[daily_return.columns] = (
        daily_data[daily_return.columns].sub(daily_data['RF'], axis=0)
    )
    
    short_ff_vol = _get_short_ff_vol(daily_data)
    short_ff_vol.stack(dropna=True).to_csv('../input/from_prepare_data/FF_M_VOL.csv', header=['FF_M_VOL'])


def _ff_reg_long(one_stock_return, ff_factors, min_sample_size):
    y = one_stock_return
    if (~y.isnull()).sum() < min_sample_size:
        return np.nan
    X = ff_factors
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()
    resid_var = np.sum(np.square(results.resid)) / (len(results.resid) - 4)
    resid_vol = np.sqrt(resid_var)
    return resid_vol


def _get_long_ff_vol(return_data, num_periods=12):
    global counter
    counter = 0

    permno_list = [
        x for x in return_data.columns
        if x not in ['const', 'RF', 'MKT_RF', 'HML', 'SMB']
        ]

    X = return_data[['const', 'MKT_RF', 'HML', 'SMB']] # copy
    ylist = return_data[permno_list]
    long_ff_vol = pd.DataFrame([])
    for index1 in range(num_periods, len(return_data)+1):
        print(
            num_periods,
            'periods Long-vol regression',
            index1 - num_periods + 1,
            '/',
            len(return_data) - num_periods
        )
        index0 = index1 - num_periods
        X_slice = X.iloc[index0:index1,:]
        ylist_slice = ylist.iloc[index0:index1,:]
        vol_one_period = ylist_slice.apply(
            _ff_reg_long,
            axis=0,
            ff_factors=X_slice,
            min_sample_size=10
        )
        vol_one_period = vol_one_period.to_frame(return_data.index[index1 - 1]).transpose()
        long_ff_vol = pd.concat([long_ff_vol, vol_one_period], axis=0)
    long_ff_vol.columns = long_ff_vol.columns.rename('PERMNO')
    return long_ff_vol


def main_ff_vol_long_weekly_return(weekly_return):
    ff = pd.read_csv('../input/famafrench.csv')
    ff['date'] = pd.to_datetime(ff['date'], format='%Y%m%d')
    ff.set_index('date', inplace=True)
    ff.columns = ['MKT_RF', 'SMB', 'HML', 'RF']
    weekly_ff = ff.resample('W').mean()
    weekly_ff = weekly_ff.to_period('W')
    weekly_data = pd.concat(
        [weekly_return, weekly_ff[['MKT_RF', 'SMB', 'HML', 'RF']]],
        axis=1,
        join='inner'
    )
    weekly_data['const'] = 1

    weekly_data[weekly_return.columns] = (
        weekly_data[weekly_return.columns].sub(weekly_data['RF'], axis=0)
    )

    long_ff_vol_24w = _get_long_ff_vol(weekly_data, num_periods=24)
    long_ff_vol_24w.stack(dropna=True).to_csv('../input/from_prepare_data/FF_L_VOL_24W.csv', header=['FF_L_VOL_24W'])
    del long_ff_vol_24w

    long_ff_vol_48w = _get_long_ff_vol(weekly_data, num_periods=48)
    long_ff_vol_48w.stack(dropna=True).to_csv('../input/from_prepare_data/FF_L_VOL_48W.csv', header=['FF_L_VOL_48W'])
    del long_ff_vol_48w

    long_ff_vol_72w = _get_long_ff_vol(weekly_data, num_periods=72)
    long_ff_vol_72w.stack(dropna=True).to_csv('../input/from_prepare_data/FF_L_VOL_72W.csv', header=['FF_L_VOL_72W'])
    del long_ff_vol_72w
    print('Done...')


def main_ff_vol_long_monthly_return(monthly_return):
    monthly_ff = pd.read_csv('../input/famafrench_mthly.csv')
    monthly_ff['date'] = pd.to_datetime(monthly_ff['date'], format='%Y%m')
    monthly_ff.set_index('date', inplace=True)
    monthly_ff.columns = ['MKT_RF', 'SMB', 'HML', 'RF']
    monthly_ff = monthly_ff.to_period('M')
    monthly_data = pd.concat(
        [monthly_return, monthly_ff[['MKT_RF', 'SMB', 'HML', 'RF']]], 
        axis=1, 
        join='inner'
        )
    monthly_data['const'] = 1

    monthly_data[monthly_return.columns] = (
        monthly_data[monthly_return.columns].sub(monthly_data['RF'], axis=0)
    )

    long_ff_vol_12m = _get_long_ff_vol(monthly_data, num_periods=12)
    long_ff_vol_12m.stack(dropna=True).to_csv('../input/from_prepare_data/FF_L_VOL_12M.csv', header=['FF_L_VOL_12M'])
    del long_ff_vol_12m

    long_ff_vol_18m = _get_long_ff_vol(monthly_data, num_periods=18)
    long_ff_vol_18m.stack(dropna=True).to_csv('../input/from_prepare_data/FF_L_VOL_18M.csv', header=['FF_L_VOL_18M'])
    del long_ff_vol_18m

    long_ff_vol_24m = _get_long_ff_vol(monthly_data, num_periods=24)
    long_ff_vol_24m.stack(dropna=True).to_csv('../input/from_prepare_data/FF_L_VOL_24M.csv', header=['FF_L_VOL_24M'])
    del long_ff_vol_24m
    print('Done...')


if __name__ == '__main__':
    t0 = time.time()
    use_full_data = True
    if use_full_data:
        print('Read h5...')
        data = pd.read_hdf('../input/full_data_0218_washed.h5', key='data')
        print('Done reading h5...')
    else:
        data = _load_raw_data('../input/stock300.csv')
    t1 = time.time()
    print(t1 - t0, "seconds...")

    daily_return, weekly_return, t = main_total_vol(data)

    #main_ff_vol_short(daily_return)
    del daily_return

    main_ff_vol_long_weekly_return(weekly_return)

    #main_ff_vol_long_monthly_return(monthly_return)

    t4 = time.time()
    print(t4 - t0, "seconds in total...")