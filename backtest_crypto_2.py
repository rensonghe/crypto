
#coding:utf-8
#%%
import pandas as pd
import numpy as np
import math
from datetime import datetime


exchange='binance'
symbol='xrpusdt'
time_span='1min'
slippage=0.0006
commission_rate=0.0002

capital=100000
# value=1000000
split=5
max_margin=1.6

p = 'vwap_60s'
amount = '130000'
bar = '15'
out_threshold = '80'
date = '202303'
path_quote='/home/data_crypto/dc_data_bar/xrpusdt/%s_%s_%s.csv' % (symbol, time_span,date)
quote_data = pd.read_csv(path_quote)
path_signal='/home/xianglake/songhe/crypto_backtest/xrpusdt/%s_%s_20230301_0330_%sbar_%s_ST1.0_20230906_filter_%s_%s_99sec.csv' % (exchange, symbol,bar,p, out_threshold,amount)
# path_signal='/home/xianglake/songhe/crypto_backtest/xrpusdt/%s_%s_20230401_0430_%sbar_%s_ST2.0_20230905_pctrank_%s_%s_99sec.csv' % (exchange, symbol,bar,p, out_threshold,amount)
signal_data = pd.read_csv(path_signal)
# print(signal_data)

quote_data['datetime']= pd.to_datetime(quote_data['datetime'])
signal_data['datetime']= pd.to_datetime(signal_data['datetime'])

k=0
net_position=0 # 净头寸，带方向
place_order_size=0
avg_deal_price = 0
net_position_series=[]
value_series=[]
commission_series=[]
balance_series=[]
profit_series=[]

place_order_price_series=[]
place_order_qty_series=[]
place_order_side_series=[]
place_order_size_series=[]

deal_order_price_series=[]
deal_order_qty_series=[]
deal_order_side_series=[]
deal_order_size_series=[]
deal_order_time_series=[]

pf_p=0.02
pf_l=0.005

print("Current time:", datetime.now())
for i in range(0,len(quote_data)):   # 类似onbar

    # 撮合成交
    commission=0
    # profit=net_position*(quote_data['close'][i-1]-quote_data['close'][i-1].shift(1)).fillna(0)
    for k in range(0,len(place_order_price_series)):

    # 开盘价撮合
    #多单
        if place_order_side_series[k]==1 and place_order_price_series[k]>=quote_data['open'][i]:
            deal_order_price_series.append(quote_data['open'][i])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['datetime'][i])

            net_position=net_position+place_order_qty_series[k]
            commission=commission+place_order_qty_series[k]*quote_data['open'][i]*commission_rate
            capital=capital - place_order_side_series[k]*quote_data['open'][i]*place_order_qty_series[k] - place_order_qty_series[k]*quote_data['open'][i]*commission_rate
            print('balance:' + str(balance))
            print('avg_deal_price:', avg_deal_price)
            print('time' + str(quote_data['datetime'][i])+'deal，side:' + str(place_order_side_series[k]) + ' price:' + str(quote_data['open'][i]) + ' quantity:' + str(place_order_qty_series[k])+ ' commision:' + str(commission))

            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] =0
            place_order_size_series[k] =0

            continue
    # 空单
        if place_order_side_series[k]==-1 and place_order_price_series[k]<=quote_data['open'][i]:
            deal_order_price_series.append(quote_data['open'][i])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['datetime'][i])


            net_position = net_position - place_order_qty_series[k]
            commission = commission+place_order_qty_series[k] * quote_data['open'][i] * commission_rate
            capital = capital - place_order_side_series[k] * quote_data['open'][i] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * quote_data['open'][i] * commission_rate
            print('balance:' + str(balance))
            print('avg_deal_price:', avg_deal_price)
            print('time' + str(quote_data['datetime'][i])+'deal，side:' + str(place_order_side_series[k]) + ' price:' + str(quote_data['open'][i]) + ' quantity:' + str(place_order_qty_series[k])+ ' commision:' + str(commission))

            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

            continue

    # 中间限定价撮合
    # 多单
        if place_order_side_series[k] == 1 and place_order_price_series[k] >= min(quote_data['open'][i],quote_data['high'][i],quote_data['low'][i],quote_data['close'][i])\
            and place_order_price_series[k] <= max(quote_data['open'][i],quote_data['high'][i],quote_data['low'][i],quote_data['close'][i]):
            deal_order_price_series.append(place_order_price_series[k])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['datetime'][i])

            net_position = net_position + place_order_qty_series[k]
            commission = commission+place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            capital = capital - place_order_side_series[k] * place_order_price_series[k] * place_order_qty_series[k] - \
                      place_order_qty_series[k]  * place_order_price_series[k] * commission_rate
            print('balance:' + str(balance))
            print('avg_deal_price:', avg_deal_price)
            print('time' + str(quote_data['datetime'][i])+'deal，side:' + str(place_order_side_series[k]) + ' price:' + str(place_order_price_series[k]) + ' quantity:' + str(place_order_qty_series[k])+ ' commision:' + str(commission))
            place_order_price_series[k] = 0
            place_order_qty_series[k] = 0
            place_order_side_series[k] = 0
            place_order_size_series[k] = 0

    # 空单
        if place_order_side_series[k] == -1 and place_order_price_series[k] >= min(quote_data['open'][i],quote_data['high'][i],quote_data['low'][i],quote_data['close'][i])\
            and place_order_price_series[k] <= max(quote_data['open'][i],quote_data['high'][i],quote_data['low'][i],quote_data['close'][i]):
            deal_order_price_series.append(place_order_price_series[k])
            deal_order_qty_series.append(place_order_qty_series[k])
            deal_order_side_series.append(place_order_side_series[k])
            deal_order_size_series.append(place_order_size_series[k])
            deal_order_time_series.append(quote_data['datetime'][i])

            net_position = net_position - place_order_qty_series[k]
            commission = commission+place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            capital = capital - place_order_side_series[k] * place_order_price_series[k] * place_order_qty_series[k] - \
                      place_order_qty_series[k] * place_order_price_series[k] * commission_rate
            print('balance:' + str(balance))
            print('avg_deal_price:',avg_deal_price)
            print('time' + str(quote_data['datetime'][i])+'deal，side:' + str(place_order_side_series[k]) + ' price:' + str(place_order_price_series[k]) + ' quantity:' + str(place_order_qty_series[k])+ ' commision:' + str(commission))
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
    value_series.append(net_position*quote_data['close'][i])

    balance = capital+net_position*quote_data['close'][i]
    balance_series.append(balance)

    if i>=1:
        profit=balance_series[i]-balance_series[i-1]
    else: profit=0

    profit_series.append(profit)
    commission_series.append(commission)

    qtt=0
    # 成交均价计算
    if net_position!=0:

        for l in range(len(deal_order_qty_series)-1,-1,-1):
            qtt=qtt+deal_order_qty_series[l]

            if int(qtt)>=int(abs(net_position)):

                avg_deal_price=(np.dot(deal_order_qty_series[len(deal_order_qty_series)-1:l+1], deal_order_price_series[len(deal_order_qty_series)-1:l+1])
                +(qtt-abs(net_position))*deal_order_price_series[l])/(sum(deal_order_qty_series[len(deal_order_qty_series)-1:l+1])+qtt-abs(net_position))

                break
        # print('avg_deal_price:'+str(avg_deal_price))
        net_profit_rate = (quote_data['close'][i] / avg_deal_price - 1)*(net_position/abs(net_position))
        # print('net_profit_rate:' + str(net_profit_rate))
    # 止盈止损挂单
        if net_profit_rate>=pf_p:
            if net_position<0: place_order_side = 1
            if net_position>0: place_order_side = -1
            place_order_price = round((quote_data['close'][i] + quote_data['close'][i] * place_order_side * slippage)*10000)/10000
            place_order_qty=abs(net_position)

            place_order_price_series.append(place_order_price)
            place_order_qty_series.append(place_order_qty)
            place_order_side_series.append(place_order_side)
            place_order_size_series.append(place_order_size)
            print('place stop profit order，side:' + str(place_order_side) + ' price:' + str(place_order_price) + ' quantity:' + str(abs(net_position)))
            continue
            # 止盈bar不做其他信号操作

        if net_profit_rate<=-pf_l:
            if net_position<0: place_order_side = 1
            if net_position>0: place_order_side = -1
            place_order_price = round((quote_data['close'][i] + quote_data['close'][i] * place_order_side * slippage)*10000)/10000
            place_order_qty=abs(net_position)

            place_order_price_series.append(place_order_price)
            place_order_qty_series.append(place_order_qty)
            place_order_side_series.append(place_order_side)
            place_order_size_series.append(place_order_size)
            print('place stop loss order，side:' + str(place_order_side) + ' price:' + str(place_order_price) + ' quantity:' + str(abs(net_position)))
            continue
            # 止损bar不做其他信号操作
        # if i>=1 and balance_series[i]!=balance_series[i-1]:
        #     print('balance:'+str(balance))
    # 信号挂单及平单
    k=0
    for j in range(0,len(signal_data)):
        if quote_data['close_time'][i]==signal_data['closetime'][j]:
            #k=k+1
            # 统计有多少笔挂单
            if signal_data['side'][j]=='sell': place_order_side=-1
            if signal_data['side'][j]== 'buy': place_order_side = 1
            place_order_price=round((signal_data[p][j] + signal_data[p][j] * place_order_side * slippage)*10000)/10000
            place_order_qty=balance*max_margin/split/place_order_price
            place_order_size=place_order_side*place_order_qty

            if place_order_side*net_position<0:
            # 发现一根bar方向不相同的信号，先撤单
                if k==0: # 是否是第一个与持仓反向的订单
                    place_order_price_series = []
                    place_order_qty_series = []
                    place_order_side_series=[]
                    place_order_size_series = []
                    print('cancel negative side order when net_position !=0')
                # if place_order_side_series[k]*net_position<0:
                # 再挂平仓单
                    place_order_price_series.append(place_order_price)
                    place_order_qty_series.append(place_order_qty)
                    place_order_side_series.append(place_order_side)
                    place_order_size_series.append(place_order_size)
                    print('place negative side close postion order, side:'+str(signal_data['side'][j])+' price:'+str(place_order_price)+' quantity:'+str(abs(net_position)))
                k=k+1
            if net_position==0 and place_order_side_series!=[] and place_order_side != place_order_side_series[-1] :
            # 当出现一根bar里有相反的信号，将所有挂单先撤销
                place_order_price_series = []
                place_order_qty_series = []
                place_order_side_series=[]
                place_order_size_series = []
                print('cancel negative side order when net_position ==0')
            if place_order_size_series!=[]:
                if abs(np.dot(place_order_size_series,place_order_price_series)+ \
                        net_position*quote_data['close'][i]) <=balance*max_margin:
                    # 当仓位不超限时，下单
                    place_order_price_series.append(place_order_price)
                    place_order_qty_series.append(place_order_qty)
                    place_order_side_series.append(place_order_side)
                    place_order_size_series.append(place_order_size)
                    print('place order, side:'+str(signal_data['side'][j])+' price:'+str(place_order_price)+' quantity:'+str(place_order_qty))
                else:
                    place_order_price_series = []
                    place_order_qty_series = []
                    place_order_side_series = []
                    place_order_size_series = []
                    place_order_price_series.append(place_order_price)
                    place_order_qty_series.append(place_order_qty)
                    place_order_side_series.append(place_order_side)
                    place_order_size_series.append(place_order_size)
                    print('place order, side:'+str(signal_data['side'][j])+' price:'+str(place_order_price)+' quantity:'+str(place_order_qty))

            else:
                if abs(net_position*quote_data['close'][i]) <balance*max_margin:
                    # 当仓位不超限时，下单
                    place_order_price_series.append(place_order_price)
                    place_order_qty_series.append(place_order_qty)
                    place_order_side_series.append(place_order_side)
                    place_order_size_series.append(place_order_size)
                    print('place order, side:'+str(signal_data['side'][j])+' price:'+str(place_order_price)+' quantity:'+str(place_order_qty))

# 交易统计及结果记录
print("Current time:", datetime.now())

pnl_result=pd.concat([quote_data['datetime'],quote_data['open'],quote_data['high'],quote_data['low'],quote_data['close'],pd.Series(balance_series),pd.Series(profit_series),pd.Series(net_position_series),pd.Series(commission_series)],axis=1)
pnl_result.to_csv('/home/xianglake/yiliang/log/xrp_pnl_withstop_%s_%s_%s_%s.csv'%(date, p,out_threshold,amount),
            index=True, sep=',')
trades_result=pd.concat([pd.Series(deal_order_time_series),pd.Series(deal_order_size_series),pd.Series(deal_order_price_series)],axis=1)
trades_result.to_csv('/home/xianglake/yiliang/log/xrp_trades_withstop_%s_%s_%s_%s.csv'%(date, p,out_threshold,amount),
            index=True, sep=',')
