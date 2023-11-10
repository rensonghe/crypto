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
from HFT_factor_3 import add_factor_process
# from HFT_factor_2 import add_factor_process
# from HFT_factor_4 import add_factor_process
def cumsum(df):
    df['volume'] = np.cumsum(abs(df['size'].fillna(0)))
    df['amount'] = np.cumsum(df['price'].fillna(0) * abs(df['size'].fillna(0)))
    return df

def read_pickle(path):
    with open(path, "rb") as f:
        obj_again = pickle.load(f)
    return obj_again
    # print("反序列化后的对象为{}".format(obj_again))

# def vwap(df, second):
#     q = abs(df['size'].fillna(0))
#     p = df['price'].fillna(0)
#     for i in range(0,second+1):
#         df['vwap_5s'] = (p*q + p.shift(-1)*q.shift(-1) + p.shift(-2)*q.shift(-2) + p.shift(-3)*q.shift(-3)+ p.shift(-4)*q.shift(-4))/(q+q.shift(-1) +q.shift(-2) +q.shift(-3) +q.shift(-4))
#     return df['vwap_5s']
def vwap(df, second):
    amount = df['amount'].fillna(method='ffill')
    volume = df['volume'].fillna(method='ffill')
    df = (df['amount'].shift(-(second-1)) - df['amount'])/(df['volume'].shift(-(second-1))-df['volume'])
    return df
def twap(df, second):
    price = df['price'].fillna(method='ffill')
    df['twap'] += [price.shift(-(i)) for i in range(0, second)][0]
    return df['twap']
#%%
symbol = 'solusdt'
year = 2023
all_data = read_pickle('/home/data_crypto/binance/%s/tick/%s_tick_%s.pkl'%(symbol,symbol,year))
all_data = all_data.sort_values(by='closetime', ascending=True)
min = 40

#%%
all_data = all_data.dropna(subset=['ask_price1'])
all_data = all_data.dropna(subset=['closetime'])
all_data = all_data.sort_values(by='closetime', ascending=True)
trade = all_data.loc[:, ['closetime', 'price', 'size', 'volume', 'amount']]
depth = all_data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
         'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
         'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5', 'ask_price6', 'ask_size6', 'bid_price6', 'bid_size6',
         'ask_price7', 'ask_size7', 'bid_price7', 'bid_size7', 'ask_price8', 'ask_size8', 'bid_price8', 'bid_size8',
         'ask_price9', 'ask_size9', 'bid_price9', 'bid_size9', 'ask_price10', 'ask_size10', 'bid_price10', 'bid_size10']]
#
start = time.time()

# depth_factor = depth_factor_process(depth, rolling=60)
# trade_factor = trade_factor_process(trade, rolling=60)
add_factor = add_factor_process(depth=depth, trade=trade, min=min)
# aggre_factor = order_aggressiveness(depth, rolling=10)
end = time.time()
print('Total Time = %s' % (end - start))
#
add_factor['vwap_2s'] = (add_factor['price'].fillna(method='ffill')*abs(add_factor['size'].fillna(method='ffill'))).rolling(2).sum()/abs(add_factor['size'].fillna(method='ffill')).rolling(2).sum()
add_factor['vwap_5s'] = (add_factor['price'].fillna(method='ffill')*abs(add_factor['size'].fillna(method='ffill'))).rolling(5).sum()/abs(add_factor['size'].fillna(method='ffill')).rolling(5).sum()
add_factor['vwap_10s'] = (add_factor['price'].fillna(method='ffill')*abs(add_factor['size'].fillna(method='ffill'))).rolling(10).sum()/abs(add_factor['size'].fillna(method='ffill')).rolling(10).sum()
add_factor['vwap_30s'] = (add_factor['price'].fillna(method='ffill')*abs(add_factor['size'].fillna(method='ffill'))).rolling(30).sum()/abs(add_factor['size'].fillna(method='ffill')).rolling(30).sum()
add_factor['vwap_60s'] = (add_factor['price'].fillna(method='ffill')*abs(add_factor['size'].fillna(method='ffill'))).rolling(60).sum()/abs(add_factor['size'].fillna(method='ffill')).rolling(60).sum()
add_factor['vwap_120s'] = (add_factor['price'].fillna(method='ffill')*abs(add_factor['size'].fillna(method='ffill'))).rolling(120).sum()/abs(add_factor['size'].fillna(method='ffill')).rolling(120).sum()


# add_factor['vwap_2s_f'] = vwap(add_factor, second=2)
# add_factor['vwap_5s_f'] = vwap(add_factor, second=5)
# add_factor['vwap_10s_f'] = vwap(add_factor, second=10)
# add_factor['vwap_30s_f'] = vwap(add_factor, second=30)
# add_factor['vwap_60s_f'] = vwap(add_factor, second=60)
# add_factor['vwap_120s_f'] = vwap(add_factor, second=120)

del all_data, depth, trade
#

# volume/dollar bar
def dollar_bars(df, dv_column, m):
    '''
    compute dollar bars

    # args
        df: pd.DataFrame()
        dv_column: name for dollar volume data
        m: int(), threshold value for dollars
    # returns
        idx: list of indices
    '''
    t = df[dv_column].astype(int)
    ts = 0
    idx = []
    # for i, x in enumerate(tqdm(t)):
    for i in tqdm(range(1, len(t))):
        if t[i] - t[i - 1] >= m:
            # print(t[i])
            idx.append(i)
            continue
        # ts += x
        # if ts >= m:
        #     idx.append(i)
        #     ts = 0
        # continue
    return idx

def dollar_bar_df(df, dv_column, m):
    idx = dollar_bars(df, dv_column, m)
    # print(df.iloc[idx])
    return df.iloc[idx].drop_duplicates()
#
add_factor['amount'] = add_factor['amount'].fillna(method='ffill')
def trans_pickle(output_path, pickle_name, df):
    with open(os.path.join(output_path, pickle_name), "wb") as file:
        pickle.dump(df, file)
    print("save {} file done".format(pickle_name))
#
dollar = [110000,120000,130000,140000]
# dollar = [60000, 70000]
# dollar = [1100000,1200000,1300000,1400000]
# dollar = [90000,100000]
# dollar = [120000]
for i in dollar:
    add_factor = add_factor.iloc[3:,:]
    data = dollar_bar_df(add_factor, 'amount',int(i))
    if year == 2023:
        Jan = data[(data.index>='2023-01-01')&(data.index<='2023-01-30')]
        print(len(Jan))
        del Jan
        Feb = data[(data.index >= '2023-02-01') & (data.index <= '2023-02-28')]
        print(len(Feb))
        del Feb
        Mar = data[(data.index >= '2023-03-01') & (data.index <= '2023-03-30')]
        print(len(Mar))
        del Mar
        Apr = data[(data.index >= '2023-04-01') & (data.index <= '2023-04-30')]
        print(len(Apr))
        del Apr
        May = data[(data.index >= '2023-05-01') & (data.index <= '2023-05-30')]
        print(len(May))
        del May
        Jun = data[(data.index >= '2023-06-01') & (data.index <= '2023-06-30')]
        print(len(Jun))
        del Jun
        Jul = data[(data.index >= '2023-07-01') & (data.index <= '2023-07-30')]
        print(len(Jul))
        del Jul
        Aug = data[(data.index >= '2023-08-01') & (data.index <= '2023-08-30')]
        print(len(Aug))
        del Aug
        Sep = data[(data.index >= '2023-09-01') & (data.index <= '2023-09-30')]
        print(len(Sep))
        del Sep
        Oct = data[(data.index >= '2023-10-01') & (data.index <= '2023-10-30')]
        print(len(Oct))
        del Oct
    # trans_pickle(output_path='/home/xianglake/songhe/binance/%s/tick_factor/'%(symbol), pickle_name='%s_tick_factor_%s_%smin_%s_Sep.pkl'%(symbol, i, min, year), df=data)
    trans_pickle(output_path='/home/data_crypto/binance/%s/tick_factor/' % (symbol),
                 pickle_name='%s_tick_factor3_%s_%smin_%s.pkl' % (symbol, i, min, year), df=data)

# del add_factor

# trans_pickle(output_path='/home/xianglake/songhe/binance/xrpusdt/tick_factor/', pickle_name='xrpusdt_tick_factor_100_000.pkl', df=data_100)
#%%
import joblib
base_path = '/home/xianglake/songhe/crypto_saved_model/'
model_side_0 = joblib.load('{}/{}/{}_lightGBM_side_0.pkl'.format(base_path, symbol, symbol))
model_side_1 = joblib.load('{}/{}/{}_lightGBM_side_1.pkl'.format(base_path, symbol, symbol))
model_side_2 = joblib.load('{}/{}/{}_lightGBM_side_2.pkl'.format(base_path, symbol, symbol))
model_side_3 = joblib.load('{}/{}/{}_lightGBM_side_3.pkl'.format(base_path, symbol, symbol))
model_side_4 = joblib.load('{}/{}/{}_lightGBM_side_4.pkl'.format(base_path, symbol, symbol))
model_out_0 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, symbol, symbol))
model_out_1 = joblib.load('{}/{}/{}_lightGBM_out_1.pkl'.format(base_path, symbol, symbol))
model_out_2 = joblib.load('{}/{}/{}_lightGBM_out_2.pkl'.format(base_path, symbol, symbol))
model_out_3 = joblib.load('{}/{}/{}_lightGBM_out_3.pkl'.format(base_path, symbol, symbol))
model_out_4 = joblib.load('{}/{}/{}_lightGBM_out_4.pkl'.format(base_path, symbol, symbol))
test = data.copy()
test_set = test.iloc[:,5:105]
X_test = np.array(test_set)
y_pred_side_0 = model_side_0.predict(X_test, num_iteration=model_side_0.best_iteration)
y_pred_side_1 = model_side_1.predict(X_test, num_iteration=model_side_1.best_iteration)
y_pred_side_2 = model_side_2.predict(X_test, num_iteration=model_side_2.best_iteration)
y_pred_side_3 = model_side_3.predict(X_test, num_iteration=model_side_3.best_iteration)
y_pred_side_4 = model_side_4.predict(X_test, num_iteration=model_side_4.best_iteration)
# y_pred_side = (y_pred_side_0 + y_pred_side_1 + y_pred_side_2 + y_pred_side_3 +
#                y_pred_side_4 ) / 5
y_pred_side = (y_pred_side_0 * 0.03+ y_pred_side_1 * 0.05+ y_pred_side_2 * 0.07+ y_pred_side_3 * 0.2 +
               y_pred_side_4 * 0.65) / 5
predict = pd.DataFrame(y_pred_side,columns=['predict'])
signal = test.reset_index()
signal['predict'] = predict['predict']

signal_1 = signal[signal['predict']>=0.11953897141214208]
signal_0 = signal[signal['predict']<=0.0785620576425637]

signal_1['side'] = 'buy'
signal_0['side'] = 'sell'
signal_df = pd.concat([signal_1, signal_0],axis=0)
signal_df = signal_df.sort_values(by='closetime', ascending=True)
signal_df = signal_df.set_index('datetime')
signal_df_only = signal_df.loc[:,['closetime','vwap_2s', 'price','predict','side']]
#%%
test_set_ = signal_df.iloc[:,5:105]
X_test_ = np.array(test_set_)
y_pred_out_0 = model_out_0.predict(X_test_, num_iteration=model_out_0.best_iteration)
y_pred_out_1 = model_out_1.predict(X_test_, num_iteration=model_out_1.best_iteration)
y_pred_out_2 = model_out_2.predict(X_test_, num_iteration=model_out_2.best_iteration)
y_pred_out_3 = model_out_3.predict(X_test_, num_iteration=model_out_3.best_iteration)
y_pred_out_4 = model_out_4.predict(X_test_, num_iteration=model_out_4.best_iteration)
# y_pred_out = (y_pred_out_0 + y_pred_out_1 + y_pred_out_2 + y_pred_out_3 + y_pred_out_4) / 5
y_pred_out = (y_pred_out_0*0.03 + y_pred_out_1*0.05 + y_pred_out_2*0.07 + y_pred_out_3*0.2 + y_pred_out_4*0.65) / 5

secondary_predict = pd.DataFrame(y_pred_out, columns=['out'])
signal_df_ = signal_df.reset_index()
signal_df_['out'] = secondary_predict['out']
signal_df_only_ = signal_df_.loc[:,['datetime','closetime','vwap_2s', 'price','predict','side','out']]
# signal_df_['out_pctrank'] = secondary_predict['out_pctrank']
# final_df = signal_df_[signal_df_['out']>=np.percentile(secondary_y_pred_train[-50000:], out_threshold)]
final_df = signal_df_[signal_df_['out']>=0.12351501197360165]
final_df = final_df.loc[:,['datetime','closetime','vwap_2s','price','predict','side','out']]
# #%%
# final_df = final_df.set_index('datetime').groupby(pd.Grouper(freq='10s')).apply('first')
# final_df = final_df.dropna()
# final_df = final_df.reset_index(drop=True)
# final_df['datetime'] = pd.to_datetime(final_df['closetime'] + 28800000, unit='ms')
#%%
final_df = final_df.set_index('datetime')
final_df['closetime'] = final_df['closetime'].astype('int')
final_df.to_csv(
        '/home/xianglake/songhe/crypto_backtest/{}/binance_{}_20231101_1107_{}bar_{}_ST1.0_20231107_filter_95_{}_{}.csv'.format(
            symbol, symbol, 5, 'vwap_2s', 80, i))