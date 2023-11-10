# %%
import csv, os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

exchange = 'binance'
symbol = 'dogeusdt'
time_span = '1min'
slippage = 0.0001
commission_rate = 0.0002
pos_rate = 1
digit = 100000

capital = 100000
capital_value = 100000
split = 5
max_margin = 1

p = 'vwap_2s'
amount = 90000
bar = '5'
out_threshold = '50'

pf_p = 0.007
pf_l = 0.005
start_date = '2023-09-01'
end_date = '2023-10-01'
path_signal = '/home/xianglake/songhe/crypto_backtest/%s/%s_%s_20230901_0930_%sbar_%s_ST1.0_20231011_filter_%s_%s.csv' % (
symbol, exchange, symbol, bar, p, out_threshold, amount)

# 加载行情数据
data_path = '/home/data_crypto/dc_bar/%s/' % (symbol)
# path_quote='/home/data_crypto/dc_data_bar/xrpusdt/%s_%s_%s.csv' % (symbol, time_span,date)
# quote_data = pd.read_csv(path_quote)

# 获取指定日期范围内的文件名列表
files = [f for f in os.listdir(data_path) if f.endswith('.csv') and start_date <= f[:10] < end_date]

# 初始化 DataFrame
quote_data = pd.DataFrame()

# 逐一读取数据文件，并追加到 DataFrame
for file in files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)
    quote_data = pd.concat([quote_data, df], ignore_index=True)

# 重置索引
quote_data = quote_data.sort_values(by='closetime', ascending=True)
quote_data.reset_index(drop=True, inplace=True)
quote_data.drop(columns=['Unnamed: 0'], inplace=True)

print('行情数据加载完成，quote_data:')
print(quote_data)

# 加载信号路径
# path_signal='/home/xianglake/songhe/crypto_backtest/xrpusdt/binance_xrpusdt_20230301_0330_15bar_vwap_60s_ST1.0_20230907_filter_80_130000.csv'
# signal_data = pd.read_csv(path_signal)
signal_data = {}  # 加载信号文件
with open(path_signal, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        signal_data.setdefault(int(row[1]), row)

# quote_data['datetime'] = pd.to_datetime(quote_data['datetime'])
print('信号数据加载完成，signal_data:{}'.format(signal_data))
#
k = 0
net_position = 0  # 净头寸，带方向
place_order_size = 0
net_position_series = []
value_series = []
commission_series = []
balance_series = []
profit_series = []

place_order_price_series = []
place_order_qty_series = []
place_order_side_series = []
place_order_size_series = []

deal_order_price_series = []
deal_order_qty_series = []
deal_order_side_series = []
deal_order_size_series = []
deal_order_time_series = []

print("Current time:", datetime.now())
for index, quote_row in quote_data.iterrows():
    closetime_value = int(quote_row['closetime'])
    datetime_value = pd.to_datetime(closetime_value + 28800000, unit='ms')
    # print('=======debug==========quote_data.closetime:{}'.format(closetime_value))
    # if index > 15:
    #     break
    # 撮合成交
    commission = 0
    # profit=net_position*(quote_data['close'][i-1]-quote_data['close'][i-1].shift(1)).fillna(0)
    for k in range(0, len(place_order_price_series)):
        # 开盘价撮合
        # 多单
        if place_order_side_series[k] == 1 and place_order_price_series[k] >= quote_row['open']:
            deal_order_price_series.append(quote_row['open'])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(closetime_value)

            net_position = net_position + place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * quote_row['open'] * commission_rate
            capital = capital - place_order_side_series[k] * quote_row['open'] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * quote_row['open'] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(datetime_value) + 'closetime' + str(closetime_value) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(quote_row['open']) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))

            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

            continue
        # 空单
        if place_order_side_series[k] == -1 and place_order_price_series[k] <= quote_row['open']:
            deal_order_price_series.append(quote_row['open'])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(closetime_value)

            net_position = net_position - place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * quote_row['open'] * commission_rate
            capital = capital - place_order_side_series[k] * quote_row['open'] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * quote_row['open'] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(datetime_value) + 'closetime' + str(closetime_value) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(quote_row['open']) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))

            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

            continue

        # 中间限定价撮合
        # 多单
        if place_order_side_series[k] == 1 and place_order_price_series[k] >= min(quote_row['open'],
                                                                                  quote_row['high'],
                                                                                  quote_row['low'],
                                                                                  quote_row['close']) \
                and place_order_price_series[k] <= max(quote_row['open'], quote_row['high'],
                                                       quote_row['low'], quote_row['close']):
            deal_order_price_series.append(place_order_price_series[k])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(closetime_value)

            net_position = net_position + place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            capital = capital - place_order_side_series[k] * place_order_price_series[k] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(datetime_value) + 'closetime' + str(closetime_value) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(place_order_price_series[k]) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))
            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

        # 空单
        if place_order_side_series[k] == -1 and place_order_price_series[k] >= min(quote_row['open'],
                                                                                   quote_row['high'],
                                                                                   quote_row['low'],
                                                                                   quote_row['close']) \
                and place_order_price_series[k] <= max(quote_row['open'], quote_row['high'],
                                                       quote_row['low'], quote_row['close']):
            deal_order_price_series.append(place_order_price_series[k])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(closetime_value)

            net_position = net_position - place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            capital = capital - place_order_side_series[k] * place_order_price_series[k] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(datetime_value) + 'closetime' + str(closetime_value) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(place_order_price_series[k]) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))
            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

    # 从挂单列中删除成交的订单
    place_order_price_series = [x for x in place_order_price_series if x != 0]
    place_order_qty_series = [x for x in place_order_qty_series if x != 0]
    place_order_side_series = [x for x in place_order_side_series if x != 0]
    place_order_size_series = [x for x in place_order_size_series if x != 0]

    net_position_series.append(net_position)
    value_series.append(net_position * quote_row['close'])

    balance = capital + net_position * quote_row['close']
    balance_series.append(balance)

    if index >= 1:
        profit = balance_series[index] - balance_series[index - 1]
    else:
        profit = 0

    profit_series.append(profit)
    commission_series.append(commission)

    qtt = 0
    cnt = 0
    # 成交均价计算
    if net_position != 0:
        for l in range(len(deal_order_qty_series) - 1, -1, -1):
            cnt = cnt + 1
            qtt = qtt + deal_order_qty_series[l]
            if int(qtt) >= int(abs(net_position)):
                # avg_deal_price=np.mean(deal_order_price_series[-1:-cnt])
                avg_deal_price = np.dot(deal_order_price_series[-1:-(cnt + 1):-1],
                                        deal_order_qty_series[-1:-(cnt + 1):-1]) / sum(
                    deal_order_qty_series[-1:-(cnt + 1):-1])
                break
        # for l in range(len(deal_order_qty_series)-1,-1,-1):
        #     qtt=qtt+deal_order_qty_series[l]
        #
        #     if int(qtt)>=int(abs(net_position)):
        #         if sum(deal_order_qty_series[len(deal_order_qty_series)-1:l+1])+qtt-abs(net_position)==0:
        #             avg_deal_price=deal_order_price_series[-1]
        #         else:
        #             avg_deal_price=(np.dot(deal_order_qty_series[len(deal_order_qty_series)-1:l+1], deal_order_price_series[len(deal_order_qty_series)-1:l+1])
        #             +(qtt-abs(net_position))*deal_order_price_series[l])/(sum(deal_order_qty_series[len(deal_order_qty_series)-1:l+1])+qtt-abs(net_position))
        #
        #         break

        # print('avg_deal_price:'+str(avg_deal_price))
        net_profit_rate = (quote_row['close'] / avg_deal_price - 1) * (net_position / abs(net_position))

        # 止损止盈单逻辑
        if net_position == 0:
            if net_profit_rate >= pf_p:
                if net_position < 0: place_order_side = 1
                if net_position > 0: place_order_side = -1
                place_order_price = round(
                    (quote_row['close'] + quote_row['close'] * place_order_side * slippage) * digit) / digit
                place_order_qty = abs(net_position)
                place_order_size = place_order_side * place_order_qty
                # 撤单
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []

                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('time' + str(datetime_value) + 'place stop profit order，side:' + str(
                    place_order_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(abs(net_position)))
                continue
                # 止盈bar不做其他信号操作

            if net_profit_rate <= -pf_l:
                if net_position < 0: place_order_side = 1
                if net_position > 0: place_order_side = -1
                place_order_price = round(
                    (quote_row['close'] + quote_row['close'] * place_order_side * slippage) * digit) / digit
                place_order_qty = abs(net_position)
                place_order_size = place_order_side * place_order_qty

                # 撤单
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []

                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print(
                    'time' + str(datetime_value) + 'place stop loss order，side:' + str(place_order_side) + ' price:' + str(
                        place_order_price) + ' quantity:' + str(abs(net_position)))
                continue
        if net_position != 0:
            print('有加仓')
            if net_profit_rate >= pf_p-0.002:
                if net_position < 0: place_order_side = 1
                if net_position > 0: place_order_side = -1
                place_order_price = round(
                    (quote_row['close'] + quote_row['close'] * place_order_side * slippage) * digit) / digit
                place_order_qty = abs(net_position)
                place_order_size = place_order_side * place_order_qty
                # 撤单
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []

                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('time' + str(datetime_value) + 'place stop profit order，side:' + str(
                    place_order_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(abs(net_position)))
                continue
                # 止盈bar不做其他信号操作

            if net_profit_rate <= -pf_l -0.002:
                if net_position < 0: place_order_side = 1
                if net_position > 0: place_order_side = -1
                place_order_price = round(
                    (quote_row['close'] + quote_row['close'] * place_order_side * slippage) * digit) / digit
                place_order_qty = abs(net_position)
                place_order_size = place_order_side * place_order_qty

                # 撤单
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []

                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print(
                    'time' + str(datetime_value) + 'place stop loss order，side:' + str(
                        place_order_side) + ' price:' + str(
                        place_order_price) + ' quantity:' + str(abs(net_position)))
                continue
            # 止损bar不做其他信号操作
        # if i>=1 and balance_series[i]!=balance_series[i-1]:
        #     print('balance:'+str(balance))
    # 信号挂单及平单
    k = 0
    match_signal = signal_data.get(closetime_value)
    if match_signal is not None:
        signal_datetime = match_signal[0]
        signal_closetime = match_signal[1]
        signal_vwapv_120s = float(match_signal[2])
        signal_price = float(match_signal[3])
        signal_predict = match_signal[4]
        signal_target = match_signal[5]
        signal_side = match_signal[6]
        # signal_datetime = match_signal[0]
        # signal_closetime = match_signal[1]
        # signal_vwapv_120s = float(match_signal[2])
        # signal_price = float(match_signal[3])
        # signal_predict = match_signal[4]
        # # signal_target = match_signal[5]
        # signal_side = match_signal[5]

        # k=k+1
        # 统计有多少笔挂单
        if signal_side == 'sell': place_order_side = -1
        if signal_side == 'buy': place_order_side = 1

        place_order_price = round(
            (signal_vwapv_120s + signal_vwapv_120s * place_order_side * slippage) * digit) / digit

        place_order_qty = min(balance, capital_value) * pos_rate / split / place_order_price
        place_order_size = place_order_side * place_order_qty

        # 平仓
        if place_order_side * net_position < 0:
            # 发现一根bar方向不相同的信号，先撤单
            if k == 0:  # 是否是第一个与持仓反向的订单
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []
                print('cancel negative side order when net_position !=0')
                # if place_order_side_series[k]*net_position<0:
                # 再挂平仓单
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(abs(net_position))
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(abs(net_position) * place_order_side)
                print('-------------------------')
                print('place negative side close postion order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(abs(net_position)) + 'time' + str(datetime_value))
            k = k + 1
        if net_position == 0 and place_order_side_series != [] and place_order_side != place_order_side_series[-1]:
            # 当出现一根bar里有相反的信号，将所有挂单先撤销
            place_order_price_series = []
            place_order_qty_series = []
            place_order_side_series = []
            place_order_size_series = []
            print('cancel negative side order when net_position ==0')
        if place_order_size_series != []:
            if abs(np.dot(place_order_size_series, place_order_price_series) + \
                   net_position * quote_row['close']) <= min(balance, capital_value) * pos_rate:
                # 当仓位不超限时，下单
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('------------------')
                print('place order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(place_order_qty) + 'time' + str(datetime_value))
            else:
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('------------------')
                print('place order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(place_order_qty) + 'time' + str(datetime_value))

        else:
            if abs(net_position * quote_row['close']) < min(balance, capital_value) * pos_rate:
                # 当仓位不超限时，下单
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('------------------')
                print('place order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(place_order_qty) + 'time' + str(datetime_value))

# 交易统计及结果记录
print("Current time:", datetime.now())

# pnl_result = pd.concat(
#     [quote_data['datetime'], quote_data['open'], quote_data['high'], quote_data['low'], quote_data['close'],
#      pd.Series(balance_series), pd.Series(profit_series), pd.Series(net_position_series), pd.Series(commission_series)],
#     axis=1)
# pnl_result.to_csv('/home/xianglake/yiliang/log/xrp_pnl_withstop.csv',
#                   index=True, sep=',')
# trades_result = pd.concat(
#     [pd.Series(deal_order_time_series), pd.Series(deal_order_size_series), pd.Series(deal_order_price_series)], axis=1)
# trades_result.to_csv('/home/xianglake/yiliang/log/xrp_trades_withstop.csv',
#                      index=True, sep=',')
#
pnl_result = pd.concat(
    [quote_data['closetime'], quote_data['open'], quote_data['high'], quote_data['low'], quote_data['close'],
     pd.Series(balance_series), pd.Series(profit_series), pd.Series(net_position_series), pd.Series(commission_series)],
    axis=1)
cols = ['closetime', 'open', 'high', 'low', 'close', 'balance_series', 'profit_series', 'net_position_series',
        'commission_series']
pnl_result.columns = cols
final_pnl = pnl_result.copy()
final_pnl['datetime'] = pd.to_datetime(final_pnl['closetime'] + 28800000, unit='ms')
final_pnl = final_pnl.set_index('datetime')

r_a = final_pnl.iloc[-1, 5] / final_pnl.iloc[0, 5]
df_aa = pd.DataFrame({
    'col1': final_pnl.iloc[::60, 1],
    'col2': final_pnl.iloc[::60, 5]
})
daily_returns = df_aa['col2'].resample('D').last().pct_change()
non_zero_returns = daily_returns[daily_returns != 0]
sharpe_ratio = non_zero_returns.mean() / non_zero_returns.std()


def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    cumulative_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - cumulative_max
    max_drawdown = drawdown.min()
    return max_drawdown


df_aa['daily_return'] = df_aa['col2'].pct_change()
mdd = max_drawdown(df_aa['daily_return'])

print('------------------  Trading Result:             --------------------')
print('------------------  Total  Retrun:   ' + str(r_a) + ' --------------------')
print('------------------  Maximum Drawdown:   ' + str(mdd) + ' --------------------')
print('------------------  Sharpe  Ratio:   ' + str(sharpe_ratio) + ' --------------------')
final_pnl[['balance_series']].plot()
# plt.legend('')
plt.title('%s pnl' % str(start_date)[:7])
plt.show()
pnl_result.to_csv('/home/xianglake/yiliang/log/%s_pnl_withstop_%s_%s_%s_%s_%s_%sp_%sl_sec_capital.csv' % (
symbol, start_date, p, out_threshold, amount, slippage, pf_p, pf_l),
                  index=True, sep=',')
trades_result = pd.concat(
    [pd.Series(deal_order_time_series), pd.Series(deal_order_size_series), pd.Series(deal_order_price_series)], axis=1)
trades_result.to_csv('/home/xianglake/yiliang/log/%s_trades_withstop_%s_%s_%s_%s_%s_%sp_%sl_sec_capital.csv' % (
symbol, start_date, p, out_threshold, amount, slippage, pf_p, pf_l),
                     index=True, sep=',')


#

def multiply_and_sum(df):
    return (df.iloc[:, 1] * df.iloc[:, 2]).sum()


trades_result['size'] = trades_result.iloc[:, 1].cumsum()
trades_result['new_size'] = trades_result['size'].where(trades_result['size'].abs() >= 0.02, 0)
total_trading_count = (trades_result['new_size'] != 0).sum()
trading_count = (trades_result['new_size'] == 0).sum()
trades_result['group'] = (trades_result['new_size'] == 0).shift(fill_value=False).cumsum()

result = trades_result.groupby('group').apply(multiply_and_sum)
p_l = -result

greater_than_zero_mean = p_l[p_l > 0].mean()
greater_than_zero_cnt = p_l[p_l > 0].count()
less_than_zero_mean = p_l[p_l < 0].mean()
less_than_zero_cnt = p_l[p_l < 0].count()
p_l_odds = -greater_than_zero_mean / less_than_zero_mean
p_l_pctg = greater_than_zero_cnt / (less_than_zero_cnt + greater_than_zero_cnt)
print('------------------trading summary by month ------------------')
# print('year_month',year_month)
print('trading_cnt', trading_count)
print('total_trading_count', total_trading_count)  # 含加仓
print('win%', p_l_pctg)  # 含加仓
print('profit_by_cnt/loss_by_cnt', p_l_odds)  # 含加仓