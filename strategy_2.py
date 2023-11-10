import redis, joblib
import numpy as np  # linear algebra
import pandas as pd
from loguru import logger

from HFT_factor_online import *
from functions import um_trade_cancel_open_orders, um_trade_new_limit_order

import warnings         # 消除groupby()函数警告
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_data(name, symbol, depth, trade, open_orders, balance, positions, strategy_arg):

    old_sec = strategy_arg['old_sec']
    strategy_depth = strategy_arg['strategy_depth']
    strategy_trade = strategy_arg['strategy_trade']
    closetime = depth['trading_time'] // 100 * 100 + 99
    depth_dict = {'closetime': depth['trading_time'] // 100 * 100 + 99,
                  'ask_price1': depth['asks'][0][0], 'ask_size1': depth['asks'][0][1],
                  'bid_price1': depth['bids'][0][0],
                  'bid_size1': depth['bids'][0][1],
                  'ask_price2': depth['asks'][1][0], 'ask_size2': depth['asks'][1][1],
                  'bid_price2': depth['bids'][1][0],
                  'bid_size2': depth['bids'][1][1],
                  'ask_price3': depth['asks'][2][0], 'ask_size3': depth['asks'][2][1],
                  'bid_price3': depth['bids'][2][0],
                  'bid_size3': depth['bids'][2][1],
                  'ask_price4': depth['asks'][3][0], 'ask_size4': depth['asks'][3][1],
                  'bid_price4': depth['bids'][3][0],
                  'bid_size4': depth['bids'][3][1],
                  'ask_price5': depth['asks'][4][0], 'ask_size5': depth['asks'][4][1],
                  'bid_price5': depth['bids'][4][0],
                  'bid_size5': depth['bids'][4][1],
                  'ask_price6': depth['asks'][5][0], 'ask_size6': depth['asks'][5][1],
                  'bid_price6': depth['bids'][5][0],
                  'bid_size6': depth['bids'][5][1],
                  'ask_price7': depth['asks'][6][0], 'ask_size7': depth['asks'][6][1],
                  'bid_price7': depth['bids'][6][0],
                  'bid_size7': depth['bids'][6][1],
                  'ask_price8': depth['asks'][7][0], 'ask_size8': depth['asks'][7][1],
                  'bid_price8': depth['bids'][7][0],
                  'bid_size8': depth['bids'][7][1],
                  'ask_price9': depth['asks'][8][0], 'ask_size9': depth['asks'][8][1],
                  'bid_price9': depth['bids'][8][0],
                  'bid_size9': depth['bids'][8][1],
                  'ask_price10': depth['asks'][9][0], 'ask_size10': depth['asks'][9][1],
                  'bid_price10': depth['bids'][9][0], 'bid_size10': depth['bids'][9][1],
                  }
    strategy_depth.append(depth_dict)

    trade_dict = {'closetime': trade['trading_time'] // 100 * 100 + 99,
                  'price': trade['price'], 'size': trade['quantity'], 'is_maker': trade['is_maker'],
                  'volume': trade['daily_volume'], 'amount': trade['daily_amount']
                  }
    strategy_trade.append(trade_dict)
    closetime = depth['trading_time'] // 100 * 100 + 99

    time_10 = int(closetime / 1000)
    # print(time_10, 'now time')
    # print(strategy_depth[-1]['closetime'], 'last closetime')
    # print(strategy_depth[0]['closetime'], 'first closetime')
    # print(time_10 - last_time)
    interval_time = 60000 * 45  # 提前储存40分钟数据用于计算因子
    if strategy_depth[-1]['closetime'] - strategy_depth[0]['closetime'] > interval_time and time_10 - last_time > 0.999:
        # print(strategy_depth[-1]['closetime'] - strategy_depth[0]['closetime'])
        last_time = time_10
        len_depth = int(len(strategy_depth) * 0.99)
        diff_time = strategy_depth[-1]['closetime'] - strategy_depth[-len_depth]['closetime']
        if diff_time > interval_time:
            strategy_depth = strategy_depth[-len_depth:]
        len_trade = int(len(strategy_trade) * 0.99)
        if strategy_trade[-1]['closetime'] - strategy_trade[-len_trade]['closetime'] > interval_time:
            strategy_trade = strategy_trade[-len_trade:]

        df_depth = pd.DataFrame(strategy_depth)
        df_trade = pd.DataFrame(strategy_trade)
        # print(df_depth, 'depth----------')
        # print(df_trade, 'trade----------')
        # df_trade['datetime'] = pd.to_datetime(df_trade['closetime'] + 28800000, unit='ms')
        # df_trade = df_trade.groupby(pd.Grouper(key='datetime', freq='30s', group_keys=True)).apply(vwap_30s)
        # df_trade = df_trade.set_index('datetime').groupby(pd.Grouper(freq='1D'), group_keys=True).apply(cumsum)
        df_trade['size'] = np.where(df_trade['is_maker'] == False, (-1) * df_trade['size'], df_trade['size'])
        # trade_df = trade_df.loc[:, ['closetime', 'price', 'size']]
        del df_trade['is_maker']
        # trade_df = trade_df.set_index('datetime').groupby(pd.Grouper(freq='1D'), group_keys=True).apply(cumsum)
        # df_trade = df_trade.reset_index(drop=True)
        # 100ms数据trade和depth合并
        data_merge = pd.merge(df_depth, df_trade, on='closetime', how='outer')
        data_merge = data_merge.sort_values(by='closetime', ascending=True)
        data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
        data_merge['sec'] = data_merge['datetime'].dt.second
        # print(data_merge)
        closetime_sec = time.localtime(closetime / 1000).tm_sec
        # print(closetime_sec,'closetime_sec')
        # print(data_merge['sec'].iloc[-1],'sec')
        # print(old_sec,'old_sec')
        if closetime_sec != old_sec:
            if data_merge['sec'].iloc[-1] != data_merge['sec'].iloc[-2]:
                old_sec = closetime_sec
                tick1 = data_merge.iloc[:-1, :]
                print(tick1,'------tick1---------')
        # 取这一秒内最后一条切片为这个1s的点
        tick1s = tick1.set_index('datetime').groupby(pd.Grouper(freq='1000ms'), group_keys=True).apply('last')



    return tick1s

def strategy_sub(name, symbol, depth, trade, open_orders, balance, positions, strategy_arg):
    # strategy_depth = strategy_arg['strategy_depth']
    # strategy_trade = strategy_arg['strategy_trade']
    y_pred_out_list = strategy_arg['y_pred_out_list']
    y_pred_side_list = strategy_arg['y_pred_side_list']
    last_time = strategy_arg['last_time']
    tick1 = strategy_arg['tick1']
    old_sec = strategy_arg['old_sec']
    kill_time = strategy_arg['kill_time']

    split_count = 5
    place_rate = 2 / 10000
    capital = 200
    pos_rate = 0.3  # 持仓比例

    base_path = '/home/ubuntu/binance-market/crypto_saved_model'

    threshold = 120000
    side_long = 0.6063621242380169
    side_short = 0.3836363736141947
    out = 0.7820945850293652

    model_side_0 = joblib.load('{}/{}/{}_lightGBM_side_0.pkl'.format(base_path, symbol, symbol))
    model_side_1 = joblib.load('{}/{}/{}_lightGBM_side_1.pkl'.format(base_path, symbol, symbol))
    model_side_2 = joblib.load('{}/{}/{}_lightGBM_side_2.pkl'.format(base_path, symbol, symbol))
    model_side_3 = joblib.load('{}/{}/{}_lightGBM_side_3.pkl'.format(base_path, symbol, symbol))
    model_side_4 = joblib.load('{}/{}/{}_lightGBM_side_4.pkl'.format(base_path, symbol, symbol))
    model_out_0 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, symbol, symbol))
    model_out_1 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, symbol, symbol))
    model_out_2 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, symbol, symbol))
    model_out_3 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, symbol, symbol))
    model_out_4 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, symbol, symbol))

    closetime = depth['trading_time'] // 100 * 100 + 99
    # depth_dict = {'closetime': depth['trading_time'] // 100 * 100 + 99,
    #               'ask_price1': depth['asks'][0][0], 'ask_size1': depth['asks'][0][1], 'bid_price1': depth['bids'][0][0],
    #               'bid_size1': depth['bids'][0][1],
    #               'ask_price2': depth['asks'][1][0], 'ask_size2': depth['asks'][1][1], 'bid_price2': depth['bids'][1][0],
    #               'bid_size2': depth['bids'][1][1],
    #               'ask_price3': depth['asks'][2][0], 'ask_size3': depth['asks'][2][1], 'bid_price3': depth['bids'][2][0],
    #               'bid_size3': depth['bids'][2][1],
    #               'ask_price4': depth['asks'][3][0], 'ask_size4': depth['asks'][3][1], 'bid_price4': depth['bids'][3][0],
    #               'bid_size4': depth['bids'][3][1],
    #               'ask_price5': depth['asks'][4][0], 'ask_size5': depth['asks'][4][1], 'bid_price5': depth['bids'][4][0],
    #               'bid_size5': depth['bids'][4][1],
    #               'ask_price6': depth['asks'][5][0], 'ask_size6': depth['asks'][5][1], 'bid_price6': depth['bids'][5][0],
    #               'bid_size6': depth['bids'][5][1],
    #               'ask_price7': depth['asks'][6][0], 'ask_size7': depth['asks'][6][1], 'bid_price7': depth['bids'][6][0],
    #               'bid_size7': depth['bids'][6][1],
    #               'ask_price8': depth['asks'][7][0], 'ask_size8': depth['asks'][7][1], 'bid_price8': depth['bids'][7][0],
    #               'bid_size8': depth['bids'][7][1],
    #               'ask_price9': depth['asks'][8][0], 'ask_size9': depth['asks'][8][1], 'bid_price9': depth['bids'][8][0],
    #               'bid_size9': depth['bids'][8][1],
    #               'ask_price10': depth['asks'][9][0], 'ask_size10': depth['asks'][9][1],
    #               'bid_price10': depth['bids'][9][0], 'bid_size10': depth['bids'][9][1],
    #               }
    # strategy_depth.append(depth_dict)
    #
    # trade_dict = {'closetime': trade['trading_time'] // 100 * 100 + 99,
    #               'price': trade['price'], 'size': trade['quantity'], 'is_maker': trade['is_maker'],
    #               'volume': trade['daily_volume'], 'amount': trade['daily_amount']
    #               }
    # strategy_trade.append(trade_dict)

    # print(strategy_depth, '------depth---------')
    # print(closetime, 'closetime')
    # if closetime:

    symbol_name = symbol.upper()
    pos_try = positions.get(symbol_name)
    if pos_try is not None:
        pos = positions[symbol_name]['position_amount']
        pos_entry_price = positions[symbol_name]['entry_price']
    else:
        pos = 0.0
        pos_entry_price = 0.0
        ask_price_1 = depth['asks'][0][0]
        bid_price_1 = depth['bids'][0][0]
        # 止盈止损
        if int(closetime / 1000) - int(kill_time / 1000) > 60 * 1:
            # 多仓止盈止损
            if pos > 0:
                # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
                pf = float(bid_price_1 / abs(pos_entry_price) )- 1
                con1 = 0
                if pf > 0.009:
                    con1 = 1
                    msg_ = f'stop long position profit---close price:{bid_price_1}---size:{pos}---time:{closetime}---symbol:{symbol_name}'
                    logger.info(msg_)
                elif pf <= -0.005:
                    con1 = 1
                    msg_ = f'stop long position loss---close price:{bid_price_1}---size:{pos}---time:{closetime}---symbol:{symbol_name}'
                    logger.info(msg_)
                if con1 == 1:
                    # print('-------------离场时间-----------------',
                    # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                    um_trade_cancel_open_orders(name, symbol)
                    kill_time = closetime
                    um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=abs(pos), price=bid_price_1)

            # 空仓止盈止损
            if pos < 0:
                # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
                # self.cancel_all(closetime=bar.closetime / 1000)
                pf = 1 - float(ask_price_1 / abs(pos_entry_price))
                con1 = 0
                if pf > 0.009:
                    con1 = 1
                    msg_ = f'stop short position profit---close price:{ask_price_1}---size:{pos}---time:{closetime}---symbol:{symbol_name}'
                    logger.info(msg_)
                elif pf <= -0.005:
                    con1 = 1
                    # print('-------------空头止损离场-------------', '品种:',self.model_symbol)
                    msg_ = f'stop short position loss---close price:{ask_price_1}---size:{pos}---time:{closetime}---symbol:{symbol_name}'
                    logger.info(msg_)
                if con1 == 1:
                    # print('-------------离场时间-----------------',
                    # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                    um_trade_cancel_open_orders(name, symbol)
                    kill_time = closetime
                    um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=abs(pos), price=ask_price_1)


        # print(closetime)
        # print(strategy_depth[-1], '----depth---')
        # time_10 = int(closetime / 1000)
        # # print(time_10, 'now time')
        # # print(strategy_depth[-1]['closetime'], 'last closetime')
        # # print(strategy_depth[0]['closetime'], 'first closetime')
        # # print(time_10 - last_time)
        # interval_time = 60000 * 45  # 提前储存40分钟数据用于计算因子
        # if strategy_depth[-1]['closetime'] - strategy_depth[0]['closetime'] > interval_time and time_10 - last_time > 0.999:
        #     # print(strategy_depth[-1]['closetime'] - strategy_depth[0]['closetime'])
        #     last_time = time_10
        #     len_depth = int(len(strategy_depth) * 0.99)
        #     diff_time = strategy_depth[-1]['closetime'] - strategy_depth[-len_depth]['closetime']
        #     if diff_time > interval_time:
        #         strategy_depth = strategy_depth[-len_depth:]
        #     len_trade = int(len(strategy_trade) * 0.99)
        #     if strategy_trade[-1]['closetime'] - strategy_trade[-len_trade]['closetime'] > interval_time:
        #         strategy_trade = strategy_trade[-len_trade:]
        #
        #     df_depth = pd.DataFrame(strategy_depth)
        #     df_trade = pd.DataFrame(strategy_trade)
        #     # print(df_depth, 'depth----------')
        #     # print(df_trade, 'trade----------')
        #     # df_trade['datetime'] = pd.to_datetime(df_trade['closetime'] + 28800000, unit='ms')
        #     # df_trade = df_trade.groupby(pd.Grouper(key='datetime', freq='30s', group_keys=True)).apply(vwap_30s)
        #     # df_trade = df_trade.set_index('datetime').groupby(pd.Grouper(freq='1D'), group_keys=True).apply(cumsum)
        #     df_trade['size'] = np.where(df_trade['is_maker'] == False, (-1) * df_trade['size'], df_trade['size'])
        #     # trade_df = trade_df.loc[:, ['closetime', 'price', 'size']]
        #     del df_trade['is_maker']
        #     # trade_df = trade_df.set_index('datetime').groupby(pd.Grouper(freq='1D'), group_keys=True).apply(cumsum)
        #     # df_trade = df_trade.reset_index(drop=True)
        #     # 100ms数据trade和depth合并
        #     data_merge = pd.merge(df_depth, df_trade, on='closetime', how='outer')
        #     data_merge = data_merge.sort_values(by='closetime', ascending=True)
        #     data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
        #     # data_merge['sec'] = data_merge['datetime'].dt.second
        #     # print(data_merge)
        #     # closetime_sec = time.localtime(closetime / 1000).tm_sec
        #     # print(closetime_sec,'closetime_sec')
        #     # print(data_merge['sec'].iloc[-1],'sec')
        #     # print(old_sec,'old_sec')
        #     # if closetime_sec != old_sec:
        #     #     if data_merge['sec'].iloc[-1] != data_merge['sec'].iloc[-2]:
        #     #         old_sec = closetime_sec
        #     #         tick1 = data_merge.iloc[:-1, :]
        #     #         print(tick1,'------tick1---------')
        #     # 取这一秒内最后一条切片为这个1s的点
        #     tick1s = data_merge.set_index('datetime').groupby(pd.Grouper(freq='1000ms'), group_keys=True).apply('last')
            # print(tick1s, '--------1s data---------')
            # print(tick1s.info())
            tick1s = tick1s.drop_duplicates(subset=['closetime'], keep='last')
            tick1s = tick1s.dropna(subset=['ask_price1'])
            trade_df = tick1s.loc[:, ['closetime', 'price', 'size', 'volume', 'amount']]
            depth_df = tick1s.loc[:, ['closetime',
                                   'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2',
                                   'bid_price2', 'bid_size2',
                                   'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
                                   'bid_price4', 'bid_size4',
                                   'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5', 'ask_price6', 'ask_size6',
                                   'bid_price6', 'bid_size6',
                                   'ask_price7', 'ask_size7', 'bid_price7', 'bid_size7', 'ask_price8', 'ask_size8',
                                   'bid_price8', 'bid_size8',
                                   'ask_price9', 'ask_size9', 'bid_price9', 'bid_size9', 'ask_price10', 'ask_size10',
                                   'bid_price10', 'bid_size10']]
            # 计算因子
            factor = add_factor_process(depth=depth_df, trade=trade_df, min=20)
            # factor['datetime'] = pd.to_datetime(factor['closetime'] + 28800000, unit='ms')
            # 计算 5s vwap price
            factor['vwap_5s'] = (factor['price'].fillna(0) * abs(factor['size'].fillna(0))).rolling(5).sum() / abs(factor['size'].fillna(0)).rolling(5).sum()
            factor['amount'] = factor['amount'].fillna(method='ffill')
            # if time.time() - self.strategy_time > 30:
            #     print('每十分钟打印一次阈值:',factor['turnover'].iloc[-1] - factor['turnover'].iloc[-2],'时间:',tick.datetime, self.model_symbol)
            #     self.strategy_time = time.time()
            if factor['amount'].iloc[-1] - factor['amount'].iloc[-2] >= threshold:
                print('bar采样触发阈值时间:', closetime, '品种:',symbol)
                signal = factor.iloc[-1:, :]
                X_test = np.array(signal.iloc[:, 5:90]).reshape(1, -1)

                y_pred_side_0 = model_side_0.predict(X_test, num_iteration=model_side_0.best_iteration)
                y_pred_side_1 = model_side_1.predict(X_test, num_iteration=model_side_1.best_iteration)
                y_pred_side_2 = model_side_2.predict(X_test, num_iteration=model_side_2.best_iteration)
                y_pred_side_3 = model_side_3.predict(X_test, num_iteration=model_side_3.best_iteration)
                y_pred_side_4 = model_side_4.predict(X_test, num_iteration=model_side_4.best_iteration)
                y_pred_side = (y_pred_side_0[0] + y_pred_side_1[0] + y_pred_side_2[0] + y_pred_side_3[0] +
                               y_pred_side_4[0]) / 5
                y_pred_side_list.append([y_pred_side])
                msg_ = f'批式方向信号:{y_pred_side_list[-1]}--time:{closetime}---symbol:{symbol}'
                logger.info(msg_)

                y_pred_side_df = pd.DataFrame(y_pred_side_list, columns=['predict'])

                if y_pred_side_df['predict'].iloc[-1] > side_long or y_pred_side_df['predict'].iloc[-1] < side_short:
                    y_pred_out_0 = model_out_0.predict(X_test, num_iteration=model_out_0.best_iteration)
                    y_pred_out_1 = model_out_1.predict(X_test, num_iteration=model_out_1.best_iteration)
                    y_pred_out_2 = model_out_2.predict(X_test, num_iteration=model_out_2.best_iteration)
                    y_pred_out_3 = model_out_3.predict(X_test, num_iteration=model_out_3.best_iteration)
                    y_pred_out_4 = model_out_4.predict(X_test, num_iteration=model_out_4.best_iteration)
                    y_pred_out = (y_pred_out_0[0] + y_pred_out_1[0] + y_pred_out_2[0] + y_pred_out_3[0] + y_pred_out_4[0]) / 5
                    y_pred_out_list.append([y_pred_out])
                    y_pred_out_df = pd.DataFrame(y_pred_out_list, columns=['out'])
                    msg_ = f'入场信号:{y_pred_out_list[-1]}-----time:{closetime}---symbol:{symbol}'
                    logger.info(msg_)

                    # 策略逻辑
                    symbol_name = symbol.upper()
                    pos_try = positions.get(symbol_name)
                    if pos_try is not None:
                        pos = positions[symbol_name]['position_amount']
                        pos_entry_price = positions[symbol_name]['entry_price']
                    else:
                        pos = 0.0
                        pos_entry_price = 0.0

                    price = factor['vwap_5s'].iloc[-1]  # 挂单价格
                    position_value = pos * price  # 持仓金额
                    place_value = capital * pos_rate / split_count  # 挂单金额
                    buy_size = round(place_value / depth['asks'][0][0], 8)  # 买单量
                    sell_size = round(place_value / depth['bids'][0][0], 8)  # 卖单量
                    max_limited_order_value = capital * pos_rate  # 最大挂单金额

                    # 计算挂单金额
                    limit_orders_values = 0.0
                    final_values = 0.0
                    for key in open_orders.keys():
                        limit_orders_values += float(open_orders[key]['price']) * abs(float(open_orders[key]['cur_qty']))
                    # 持仓金额+挂单金额

                    final_values = limit_orders_values + pos_entry_price * pos

                    # 平多仓
                    if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                            y_pred_out_df['out'].iloc[-1]) >= out and pos > 0:
                        # print('-------------平仓之前撤销所有订单-------------')
                        um_trade_cancel_open_orders(name, symbol)
                        # print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                        bid_price_3 = depth['bids'][2][0]
                        msg_ = f'下空单平多仓---平仓价格:{price*(1-place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                        logger.info(msg_)
                        um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=abs(pos), price=price*(1-place_rate))
                    # 平空仓
                    if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(y_pred_out_df['out'].iloc[-1]) >= out and pos < 0:
                        # print('-------------平仓之前撤销所有订单-------------')
                        um_trade_cancel_open_orders(name, symbol)
                        # print('-----------------------------下多单平空仓----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                        ask_price_3 = depth['asks'][2][0]
                        msg_ = f'下多单平空仓---平仓价格:{price*(1+place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                        logger.info(msg_)
                        um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=abs(pos), price=price*(1+place_rate))


                    # 开空仓
                    if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float( y_pred_out_df['out'].iloc[-1]) >= out \
                            and position_value >= -pos_rate * capital * (1 - 1 / split_count):
                        if len(open_orders) > 0:
                            last_order = list(open_orders.keys())[-1]
                            if open_orders[last_order]['cur_qty'] > 0:
                                um_trade_cancel_open_orders(name, symbol)
                        # print('--------------开空仓----------------', '品种:',self.model_symbol)
                        if max_limited_order_value <= final_values * 1.0001:
                            um_trade_cancel_open_orders(name, symbol)
                        msg_ = f'开空仓---开仓价格:{price * (1 - place_rate)}---size:{sell_size}---time:{closetime}---symbol:{symbol}'
                        logger.info(msg_)
                        um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=sell_size, price=price*(1-place_rate))
                        # self.fill_order_time = tick.closetime

                    # 开多仓
                    if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                            y_pred_out_df['out'].iloc[-1]) >= out and position_value <= pos_rate * capital * (
                            1 - 1 / split_count):
                        # 如果此时有挂空单，全部撤掉
                        if len(open_orders) > 0:
                            last_order = list(open_orders.keys())[-1]
                            if open_orders[last_order]['cur_qty'] < 0:
                                um_trade_cancel_open_orders(name, symbol)
                        # print('--------------开多仓----------------', '品种:',self.model_symbol)
                        if max_limited_order_value <= final_values * 1.0001:
                            um_trade_cancel_open_orders(name, symbol)
                        msg_ = f'开多仓---开仓价格:{price * (1 + place_rate)}---size:{buy_size}---time:{closetime}---symbol:{symbol}'
                        logger.info(msg_)

                        um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=buy_size, price=price*(1+place_rate))

                        # self.fill_order_time = tick.closetime
                        # return
                    # else:
                        # 保持仓位
                        # return

    # =====逻辑部分结束，循环外传参=====
    strategy_depth = strategy_arg['strategy_depth']
    strategy_trade = strategy_arg['strategy_trade']
    y_pred_out_list = strategy_arg['y_pred_out_list']
    y_pred_side_list = strategy_arg['y_pred_side_list']
    last_time = strategy_arg['last_time']
    tick1 = strategy_arg['tick1']
    old_sec = strategy_arg['old_sec']
    kill_time = strategy_arg['kill_time']

    strategy_arg_update = {
        'strategy_depth': strategy_depth,
        'strategy_trade': strategy_trade,
        'y_pred_out_list': y_pred_out_list,
        'y_pred_side_list': y_pred_side_list,
        'last_time': last_time,
        'tick1': tick1,
        'old_sec': old_sec,
        'kill_time': kill_time,
    }

    return strategy_arg_update
