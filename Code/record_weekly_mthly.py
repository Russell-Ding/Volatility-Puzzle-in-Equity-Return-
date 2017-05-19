
import pandas as pd
import sys
sys.path.append('C:\\project\\code')
import setting

def record_weekly_to_mthly(data_pivot):
    data_pivot = data_pivot.copy(deep=True)
    new_index = []
    for x in data_pivot.index.values:
        new_index.append(pd.to_datetime(x[-10:]))
    data_pivot.index = new_index
    data_pivot.index.name = 'date'
    data_pivot = data_pivot.resample('M').last().to_period('M')
    return data_pivot


if __name__ == '__main__':
	# file_list = ['L_VOL_12W', 'L_VOL_24W', 'L_VOL_36W', 'L_VOL_48W', 'L_VOL_72W']
	file_list = ['FF_L_VOL_24W']
	for i in range(len(file_list)):
		file_input = setting.datapath_prepared+file_list[i]+'.csv'
		data = pd.read_csv(file_input)
		data.columns = ['date', 'PERMNO', file_list[i]]
		data_pivot = data.pivot_table(index='date', columns='PERMNO', values=file_list[i])
		data_pivot = record_weekly_to_mthly(data_pivot)
		data_pivot.stack(dropna=True).to_csv(file_input, header=[file_list[i]])    
