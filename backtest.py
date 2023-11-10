# coding:utf-8
#%%
import csv, os
import pandas as pd
import numpy as np
from datetime import datetime

# exchange = 'binance'
# symbol = 'xrpusdt'
# time_span = '1min'
# slippage = 0.0004
# commission_rate = 0.0002
#
# capital = 1000000
# # value=1000000
# split = 5
# max_margin = 1.6
#
# path_quote = '/home/xianglake/data/dc_data_bar/xrpusdt/%s_%s.csv' % (symbol, time_span)
# quote_data = pd.read_csv(path_quote)
#
# path_signal = '/home/xianglake/songhe/crypto_backtest/xrpusdt/%s_%s_20230101_0131_10bar_vwap120s_ST1.0_20230807_filter_50_99sec.csv' % (
# exchange, symbol)

exchange='binance'
symbol='xrpusdt'
time_span='1min'
slippage=0.0005
commission_rate=0.0002

capital=100000
# value=1000000
split=5
max_margin=1.6

p = 'vwap_30s'
amount = '150000'
bar = '10'
out_threshold = '80'

start_date = '2022-03-01'
end_date = '2022-03-30'


# 加载行情数据
data_path = '/home/data_crypto/dc_bar/xrpusdt/'
# path_quote='/home/data_crypto/dc_data_bar/xrpusdt/%s_%s_%s.csv' % (symbol, time_span,date)
# quote_data = pd.read_csv(path_quote)

# 获取指定日期范围内的文件名列表
files = [f for f in os.listdir(data_path) if f.endswith('.csv') and start_date <= f[:10] <= end_date]

# 初始化 DataFrame
quote_data = pd.DataFrame()

# 逐一读取数据文件，并追加到 DataFrame
for file in files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)
    quote_data = quote_data.append(df)

# 重置索引
quote_data.reset_index(drop=True, inplace=True)


# 加载信号路径
path_signal='/home/xianglake/songhe/crypto_backtest/xrpusdt/%s_%s_20230301_0330_%sbar_%s_ST1.0_20230906_filter_%s_%s.csv' % (exchange, symbol,bar,p, out_threshold,amount)
# path_signal='/home/xianglake/songhe/crypto_backtest/xrpusdt/%s_%s_20230401_0430_%sbar_%s_ST2.0_20230905_pctrank_%s_%s_99sec.csv' % (exchange, symbol,bar,p, out_threshold,amount)
# signal_data = pd.read_csv(path_signal)
signal_data = {}  # 加载信号文件
with open(path_signal, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        signal_data.setdefault(int(row[1]), row)

# quote_data['datetime'] = pd.to_datetime(quote_data['datetime'])

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

pf_p = 0.02
pf_l = 0.005

print("Current time:", datetime.now())
for i in range(0, len(quote_data)):  # 类似onbar
    # 撮合成交
    commission = 0
    # profit=net_position*(quote_data['close'][i-1]-quote_data['close'][i-1].shift(1)).fillna(0)
    for k in range(0, len(place_order_price_series)):

        # 开盘价撮合
        # 多单
        if place_order_side_series[k] == 1 and place_order_price_series[k] >= quote_data['open'][i]:
            deal_order_price_series.append(quote_data['open'][i])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['closetime'][i])

            net_position = net_position + place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * quote_data['open'][i] * commission_rate
            capital = capital - place_order_side_series[k] * quote_data['open'][i] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * quote_data['open'][i] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(quote_data['closetime'][i]) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(quote_data['open'][i]) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))

            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

            continue
        # 空单
        if place_order_side_series[k] == -1 and place_order_price_series[k] <= quote_data['open'][i]:
            deal_order_price_series.append(quote_data['open'][i])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['closetime'][i])

            net_position = net_position - place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * quote_data['open'][i] * commission_rate
            capital = capital - place_order_side_series[k] * quote_data['open'][i] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * quote_data['open'][i] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(quote_data['closetime'][i]) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(quote_data['open'][i]) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))

            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

            continue

        # 中间限定价撮合
        # 多单
        if place_order_side_series[k] == 1 and place_order_price_series[k] >= min(quote_data['open'][i],
                                                                                  quote_data['high'][i],
                                                                                  quote_data['low'][i],
                                                                                  quote_data['close'][i]) \
                and place_order_price_series[k] <= max(quote_data['open'][i], quote_data['high'][i],
                                                       quote_data['low'][i], quote_data['close'][i]):
            deal_order_price_series.append(place_order_price_series[k])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['closetime'][i])

            net_position = net_position + place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            capital = capital - place_order_side_series[k] * place_order_price_series[k] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(quote_data['closetime'][i]) + 'deal，side:' + str(
                place_order_side_series[k]) + ' price:' + str(place_order_price_series[k]) + ' quantity:' + str(
                place_order_qty_series[k]) + ' commision:' + str(commission))
            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

        # 空单
        if place_order_side_series[k] == -1 and place_order_price_series[k] >= min(quote_data['open'][i],
                                                                                   quote_data['high'][i],
                                                                                   quote_data['low'][i],
                                                                                   quote_data['close'][i]) \
                and place_order_price_series[k] <= max(quote_data['open'][i], quote_data['high'][i],
                                                       quote_data['low'][i], quote_data['close'][i]):
            deal_order_price_series.append(place_order_price_series[k])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['closetime'][i])

            net_position = net_position - place_order_qty_series[k]
            commission = commission + place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            capital = capital - place_order_side_series[k] * place_order_price_series[k] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            print('balance:' + str(balance))
            print('time' + str(quote_data['closetime'][i]) + 'deal，side:' + str(
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
    value_series.append(net_position * quote_data['close'][i])

    balance = capital + net_position * quote_data['close'][i]
    balance_series.append(balance)

    if i >= 1:
        profit = balance_series[i] - balance_series[i - 1]
    else:
        profit = 0

    profit_series.append(profit)
    commission_series.append(commission)

    qtt = 0
    # 成交均价计算
    if net_position != 0:
        for l in range(len(deal_order_qty_series) - 1, -1, -1):
            qtt = qtt + deal_order_qty_series[l]
            if qtt >= abs(net_position):
                avg_deal_price = (np.dot(deal_order_qty_series[len(deal_order_qty_series) - 1:l + 1],
                                         deal_order_price_series[len(deal_order_qty_series) - 1:l + 1])
                                  + (qtt - abs(net_position)) * deal_order_price_series[l]) / (
                                             sum(deal_order_qty_series[
                                                 len(deal_order_qty_series) - 1:l + 1]) + qtt - abs(net_position))
                break
        # print('avg_deal_price:'+str(avg_deal_price))
        net_profit_rate = (quote_data['close'][i] / avg_deal_price - 1) * (net_position / abs(net_position))
        # print('net_profit_rate:' + str(net_profit_rate))
        # 止盈止损挂单
        if net_profit_rate >= pf_p:
            if net_position < 0: place_order_side = 1
            if net_position > 0: place_order_side = -1
            place_order_price = round(
                (quote_data['close'][i] + quote_data['close'][i] * place_order_side * slippage) * 10000) / 10000
            place_order_qty = abs(net_position)

            place_order_price_series.append(place_order_price)
            place_order_qty_series.append(place_order_qty)
            place_order_side_series.append(place_order_side)
            place_order_size_series.append(place_order_size)
            print('place stop profit order，side:' + str(place_order_side) + ' price:' + str(
                place_order_price) + ' quantity:' + str(abs(net_position)))
            continue
            # 止盈bar不做其他信号操作

        if net_profit_rate <= -pf_l:
            if net_position < 0: place_order_side = 1
            if net_position > 0: place_order_side = -1
            place_order_price = round(
                (quote_data['close'][i] + quote_data['close'][i] * place_order_side * slippage) * 10000) / 10000
            place_order_qty = abs(net_position)

            place_order_price_series.append(place_order_price)
            place_order_qty_series.append(place_order_qty)
            place_order_side_series.append(place_order_side)
            place_order_size_series.append(place_order_size)
            print('place stop loss order，side:' + str(place_order_side) + ' price:' + str(
                place_order_price) + ' quantity:' + str(abs(net_position)))
            continue
            # 止损bar不做其他信号操作
        # if i>=1 and balance_series[i]!=balance_series[i-1]:
        #     print('balance:'+str(balance))
    # 信号挂单及平单
    k = 0
    match_signal = signal_data.get(quote_data['closetime'][i])
    if match_signal is not None:
        signal_datetime = match_signal[0]
        signal_closetime = match_signal[1]
        signal_vwapv_120s = float(match_signal[2])
        signal_price = float(match_signal[3])
        signal_predict = match_signal[4]
        signal_target = match_signal[5]
        signal_side = match_signal[6]

        # k=k+1
        # 统计有多少笔挂单
        if signal_side == 'sell': place_order_side = -1
        if signal_side == 'buy': place_order_side = 1
        place_order_price = round(
            (signal_vwapv_120s + signal_vwapv_120s * place_order_side * slippage) * 10000) / 10000
        place_order_qty = balance * max_margin / split / place_order_price
        place_order_size = place_order_side * place_order_qty

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
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('place negative side close postion order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(abs(net_position)))
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
                   net_position * quote_data['close'][i]) <= balance * max_margin:
                # 当仓位不超限时，下单
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('place order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(place_order_qty))
            else:
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series = []
                place_order_size_series = []
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('place order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(place_order_qty))

        else:
            if abs(net_position * quote_data['close'][i]) < balance * max_margin:
                # 当仓位不超限时，下单
                place_order_price_series.append(place_order_price)
                place_order_qty_series.append(place_order_qty)
                place_order_side_series.append(place_order_side)
                place_order_size_series.append(place_order_size)
                print('place order, side:' + str(signal_side) + ' price:' + str(
                    place_order_price) + ' quantity:' + str(place_order_qty))

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
pnl_result=pd.concat([quote_data['closetime'],quote_data['open'],quote_data['high'],quote_data['low'],quote_data['close'],pd.Series(balance_series),pd.Series(profit_series),pd.Series(net_position_series),pd.Series(commission_series)],axis=1)
# pnl_result.to_csv('/home/xianglake/yiliang/log/xrp_pnl_withstop_%s_%s_%s_%s.csv'%(date, p,out_threshold,amount),
#             index=True, sep=',')
# trades_result=pd.concat([pd.Series(deal_order_time_series),pd.Series(deal_order_size_series),pd.Series(deal_order_price_series)],axis=1)
# trades_result.to_csv('/home/xianglake/yiliang/log/xrp_trades_withstop_%s_%s_%s_%s.csv'%(date, p,out_threshold,amount),
#             index=True, sep=',')