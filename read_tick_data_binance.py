import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numba as nb
import time
import datetime
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist
import os
import scipy.stats as st
import glob
import pickle as pickle
def cumsum(df):
    df['volume'] = np.cumsum(abs(df['size'].fillna(0)))
    df['amount'] = np.cumsum(df['price'].fillna(0) * abs(df['size'].fillna(0)))
    return df

def read_pickle(path):
    with open(path, "rb") as f:
        obj_again = pickle.load(f)
    return obj_again
    # print("反序列化后的对象为{}".format(obj_again))

def vwap10s(df):
    q = abs(df['volume']).values
    p = df['price'].values
    return df.assign(vwap10s=(p * q).cumsum() / q.cumsum())
#%%
start = time.time()
date = []
symbol = 'bnbusdt'
year = 2023
# def read_pickle_data(symbol)
# platform = 'binance-futures_pickle'
depth = 'book_snapshot_25'
trade = 'trades'
depth_dir_path = ('/home/data_store/dc_data/%s/binance-futures/%s/'%(symbol, depth))
trade_dir_path = ('/home/data_store/dc_data/%s/binance-futures/%s/'%(symbol, trade))

file_extension = '*.csv.gz'
depth_files = sorted(glob.glob(os.path.join(depth_dir_path, '**', file_extension), recursive=True), key=os.path.getmtime)
trade_files = sorted(glob.glob(os.path.join(trade_dir_path, '**', file_extension), recursive=True), key=os.path.getmtime)

start_date = datetime.datetime(year, 1, 1)
end_date = datetime.datetime(year, 11, 8)
all_data = pd.DataFrame()
filtered_files = []
for depth in depth_files:
    depth_file_name = os.path.basename(depth)
    depth_file_date_str = depth_file_name.split('_')[0]
    for trade in trade_files:
        trade_file_name = os.path.basename(trade)
        trade_file_date_str = trade_file_name.split('_')[0]  # Assuming the date is in the format 'YYYY-MM-DD'
        if depth_file_date_str == trade_file_date_str:
            file_date = datetime.datetime.strptime(depth_file_date_str, '%Y-%m-%d')
            if start_date <= file_date <= end_date:
            # if file_date >= start_date:
                print(file_date)
                # depth_df = read_pickle(depth)
                depth_df = pd.read_csv(depth, compression='gzip')
                depth_df = depth_df.iloc[:,2:44]
                depth_df['timestamp'] =  (depth_df['timestamp']//1000)//100*100+99
                depth_df = depth_df.sort_values(by='timestamp', ascending=True)
                depth_cols = ['closetime', 'local_timestamp',
                              'ask_price1','ask_size1', 'bid_price1', 'bid_size1','ask_price2','ask_size2', 'bid_price2', 'bid_size2',
                              'ask_price3','ask_size3', 'bid_price3', 'bid_size3','ask_price4','ask_size4', 'bid_price4', 'bid_size4',
                              'ask_price5','ask_size5', 'bid_price5', 'bid_size5','ask_price6','ask_size6', 'bid_price6', 'bid_size6',
                              'ask_price7','ask_size7', 'bid_price7', 'bid_size7','ask_price8','ask_size8', 'bid_price8', 'bid_size8',
                              'ask_price9','ask_size9', 'bid_price9', 'bid_size9','ask_price10','ask_size10', 'bid_price10', 'bid_size10']
                depth_df.columns = depth_cols
                # deoth_df = reduce_mem_usage(depth_df[0])
                del depth_cols, depth_df['local_timestamp']
                # trade_df = read_pickle(trade)
                trade_df = pd.read_csv(trade, compression='gzip')
                trade_df = trade_df.iloc[:, 2:]
                trade_df = trade_df.sort_values(by='id', ascending=True)
                trade_df['timestamp'] = (trade_df['timestamp'] // 1000) // 100 * 100 + 99
                trade_cols = ['closetime','local_timestamp', 'id', 'side', 'price', 'size']
                trade_df.columns = trade_cols
                # trade_df['datetime'] = pd.to_datetime(trade_df['closetime'] + 28800000, unit='ms')
                trade_df['datetime'] = pd.to_datetime(trade_df['closetime'], unit='ms')
                trade_df['size'] = np.where(trade_df['side'] == 'sell', (-1) * trade_df['size'], trade_df['size'])
                trade_df = trade_df.set_index('datetime').groupby(pd.Grouper(freq='1D')).apply(cumsum)
                # trade_df = trade_df.loc[:, ['closetime', 'price', 'size']]
                del trade_df['local_timestamp'], trade_df['id'], trade_df['side']
                trade_df = trade_df.reset_index(drop=True)
                # trade_df = reduce_mem_usage(trade_df[0])
                del trade_cols
                data_merge = pd.merge(depth_df, trade_df, how='outer', on='closetime')
                # data_merge = pd.merge_asof(depth_df, trade_df, on='datetime', tolerance=pd.Timedelta('2000ms'))
                data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
                data_merge = data_merge.set_index('datetime').groupby(pd.Grouper(freq='1s')).apply('last')
                #
                filtered_files.append(data_merge)
                all_data = pd.concat(filtered_files)
                # all_data = reduce_mem_usage(all_data)[0]

# all_data = all_data.set_index('datetime').groupby(pd.Grouper(freq='1s')).apply('last')
end = time.time()
print('Total Time = %s' % (end - start))
#
def trans_pickle(output_path, pickle_name, df):
    with open(os.path.join(output_path, pickle_name), "wb") as file:
        pickle.dump(df, file)
    print("save {} file done".format(pickle_name))
trans_pickle(output_path='/home/data_crypto/binance/%s/tick/'%(symbol), pickle_name='%s_tick_%s.pkl'%(symbol,year), df=all_data)