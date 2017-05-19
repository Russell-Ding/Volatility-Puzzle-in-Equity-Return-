import numpy as np
import pandas as pd
import datetime
import sys
sys.path.append('C:\\project\\code')


from dateutil.relativedelta import relativedelta


def review_pf_return(begindate,enddate,pf_weight, return_pivot):
    
    today = begindate
    five_port=pf_weight[today]
    number=len(five_port)
    monthly_pf_return = pd.DataFrame(columns=range(1,number+1))
    while(today<enddate):
        
        five_port=pf_weight[today]
        
        current_month_return=pd.DataFrame(index=[today],columns=range(1,number+1))
        current_month_return=pd.DataFrame(index=[today],columns=range(1,number+1))
        
        for i in range(0,number):
            weight=five_port[i]

            if len(weight)==0:
                current_month_return.iloc[0,i]=0
                print('skipped'+str(today)+'on '+str(i)+'th portfolio')
                continue

            # select data and convert series to dataframe
            return_df=pd.DataFrame(
                return_pivot.loc[datetime.datetime(today.year, today.month, 1),:]
                )
            # # drop the higher level column
            # return_df = return_df.xs('M_RET', axis=1, drop_level=True)
            # return_df = return_df.transpose()

            stock_list=list(weight.index.values)
            return_df=return_df.ix[return_df.index.isin(stock_list)]

            portfolio=pd.concat([return_df, weight],axis=1).dropna()
            portfolio['Return']=portfolio[today]*portfolio['weight']

            current_month_return.iloc[0,i]=portfolio['Return'].sum()
        
        monthly_pf_return=pd.concat([monthly_pf_return,current_month_return])
        monthly_pf_return=pd.DataFrame(np.array(monthly_pf_return, dtype=float),
            index=monthly_pf_return.index, columns=monthly_pf_return.columns)
        today+=relativedelta(months=1)
    return monthly_pf_return.to_period('M')


def review_pf_signal(begindate, enddate, pf_weight, signal_pivot):
    """
    :pf_weight(pd.DataFrame)
    :signal_pivot(pd.DataFrame): index='date', column='PERMNO', values='MKTCAP'
    """
    today = begindate
    five_port=pf_weight[today]
    number=len(five_port)
    monthly_pf_sig = pd.DataFrame(columns=range(1,number+1))
    
    while(today<enddate):

        five_port=pf_weight[today]  # update portfolio weight
        current_month_signal=pd.DataFrame(index=[today],columns=range(1, number+1))
        
        for i in range(0,number):
            weight=five_port[i]
            
            #print (weight)
            if len(weight)==0:
                current_month_signal.iloc[0,i]=None
                print('skipped'+str(today)+'on '+str(i)+'th portfolio')
                continue
            
            signal_df = pd.DataFrame(
                signal_pivot.loc[datetime.datetime(today.year, today.month,1),:]
                )
            
            # signal_df=signal_pivot.loc[str(today.year)+'-'+str(today.month),:]
            # signal_df=signal_df.iloc[0]
            # signal_df=signal_df.reset_index()
            # signal_df=signal_df.set_index('PERMNO')

            stock_list=list(weight.index.values)
            signal_df=signal_df.ix[signal_df.index.isin(stock_list)]

            portfolio=pd.concat([signal_df, weight],axis=1).dropna()
            portfolio['Signal']=portfolio[today]*portfolio['weight']/np.sum(portfolio['weight'])

            current_month_signal.iloc[0,i]=portfolio['Signal'].sum()

        monthly_pf_sig=pd.concat([monthly_pf_sig,current_month_signal])
        today+=relativedelta(months=1)

    monthly_pf_sig=pd.DataFrame(np.array(monthly_pf_sig, dtype=float),
        index=monthly_pf_sig.index, columns=monthly_pf_sig.columns)
    return monthly_pf_sig.to_period('M')
