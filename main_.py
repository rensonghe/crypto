#!/usr/bin/env python
import logging, time
from loguru import logger
from datetime import datetime
from binance.lib.utils import config_logging
from binance.um_futures import UMFutures
from apscheduler.schedulers.background import BackgroundScheduler
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from functions import test_get_key, um_trade_get_orders, um_trade_cancel_open_orders, um_trade_account, \
    um_trade_change_leverage, get_symbol_daily_from_redis
import pandas as pd
from HFT_factor_online import *
import joblib


config_logging(logging, logging.DEBUG)

symbol = 'btcusdt'
trade = {'symbol': 'null', 'price': 0.0, 'quantity': 0.0, 'trading_time': 0, 'is_maker': None, 'daily_volume': 0.0, 'daily_amount': 0.0}
depth = {'trading_time': 0, 'bids': [], 'asks': []}
symbol_daily = {'daily_volume': 0.0, 'daily_amount': 0.0}
balance = {'USDT': {'asset': 'USDT', 'wallet_balance': '199.93564633', 'cross_wallet': '199.93564633', 'balance_change': '0', 'trading_time': '(13位时间戳)'}}
position = {'ETHUSDT': {'symbol': 'ETHUSDT', 'position_amount': 0.200, 'entry_price': 1640.91000000,
                        'cumulative_realized': 0.16890000, 'unrealized': -0.00536252, 'margin_type': 'cross',
                        'isolated_wallet': 0, 'position_side': 'BOTH', 'monetary_unit': 'USDT',
                        'average entry price': 1640.91000000, 'break_even_price': 1706.08216, 'trading_time': '(13位时间戳)'}}

# ===========================

def start_wb_market():
    my_client = UMFuturesWebsocketClient()

    my_client.start()
    my_client.partial_book_depth(symbol=symbol, id=1, level=10, speed=100, callback=message_depth,)
    logger.info('行情depth已启动')

    # redis拉取symbol_daily
    redis_volume, redis_amount, is_complete = get_symbol_daily_from_redis(symbol)
    if is_complete is True:
        global symbol_daily
        symbol_daily['daily_volume'] = redis_volume
        symbol_daily['daily_amount'] = redis_amount
        logger.info('redis拉取symbol_daily成功，symbol_daily:{}'.format(symbol_daily))
    else:
        logger.error('symbol_daily数据不完整，不能启动策略')
        return None

    my_client.agg_trade(symbol=symbol, id=2, callback=message_trade,)
    logger.info('行情trade已启动')

def message_depth(message):
    global depth
    trading_time = message.get('T')
    if trading_time is not None:
        depth['trading_time'] = trading_time

        # 确保本条一定有且更新
        bids = message.get('b')
        if bids is not None:
            depth['bids'] = bids

        asks = message.get('a')
        if asks is not None:
            depth['asks'] = asks

def message_trade(message):
    global trade, symbol_daily
    # 首条再次读取symbol_daily，确保绝对同步
    fir = message.get('id')
    if fir is not None:
        redis_volume, redis_amount, is_complete = get_symbol_daily_from_redis(symbol)
        symbol_daily['daily_volume'] = redis_volume
        symbol_daily['daily_amount'] = redis_amount
        logger.info('再次刷新成功，symbol_daily:{}'.format(symbol_daily))

    # 非首条交易处理
    p = message.get('p')
    if p is not None:
        trade['symbol'] = message.get('s')
        price = float(message.get('p'))
        quantity = float(message.get('q'))
        trade['price'] = price
        trade['quantity'] = quantity
        trade['trading_time'] = message.get('T')
        trade['is_maker'] = message.get('m')

        # 变更symbol_daily处理
        update_daily_volume = round(symbol_daily['daily_volume'] + quantity, 8)
        update_daily_amount = round(symbol_daily['daily_amount'] + (quantity * price), 8)
        trade['daily_volume'] = update_daily_volume
        trade['daily_amount'] = update_daily_amount
        symbol_daily['daily_volume'] = update_daily_volume
        symbol_daily['daily_amount'] = update_daily_amount


def initial():
    logger.info('初始化策略开始执行')

    # 查看账户所有挂单，撤销所有挂单
    pre_pos = um_trade_get_orders(name, symbol)
    if pre_pos is True:
        if um_trade_cancel_open_orders(name, symbol) is False:
            logger.info("挂单未撤成功，策略终止")
            return None

    # 获取账户资产v2
    org_acc = um_trade_account

    # 判断是否有持仓，已有持仓则报错，终止策略

    # 查询持仓模式，将其变为单向持仓

    # 调整杠杆数额
    um_trade_change_leverage(name, leverage)

    # 启动策略
    sub()


    logger.info('初始化策略结束执行')


def message_account(message):
    t = message.get('e')
    if t is not None:
        # 过滤事件类型
        if t == 'ORDER_TRADE_UPDATE':
            logger.info('收到订单推送，时间：{t}'.format(t=message.get('E')))
            push_order = message.get('o')
            # 过滤交易对
            if push_order['s'] == symbol.upper():
                global open_orders
                if push_order.get('x') == 'TRADE':
                    if push_order['X'] == 'PARTIALLY_FILLED':      # 部成处理
                        trade_ord_id = int(push_order['i'])
                        order = open_orders.get(trade_ord_id)
                        order['cur_qty'] -= float(push_order['l'])
                        open_orders[trade_ord_id] = order
                        logger.info("订单部成{}".format(order))
                    elif push_order['X'] == 'FILLED':              # 全成处理
                        trade_ord_id = int(push_order['i'])
                        logger.info("订单全成{}".format(open_orders[trade_ord_id]))
                        del open_orders[trade_ord_id]
                    else:
                        logger.error('未知成交类型,push_order:{}'.format(push_order))
                elif push_order.get('x') == 'NEW':
                    # 强平订单发生(目前只给日志，不给处理)
                    client_ord_id = str(push_order['c'])
                    if 'autoclose' in client_ord_id:
                        logger.error("发生强平订单！push_order:{}".format(push_order))
                    elif 'adl_autoclose' in client_ord_id:
                        logger.error("订单发生ADL，push_order:{}".format(push_order))
                    else:
                        # 新订单发生 (限价和市价情况是一样的，市价也会产生new和trade推送)
                        order_id = int(push_order['i'])
                        new_order = {"order_id": order_id, "client_ord_id": client_ord_id, "dir": push_order['S'],
                                     "org_qty": float(push_order['q']), "cur_qty": float(push_order['q']),
                                     "price": float(push_order['p']), "work_time": int(push_order['T'])}
                        open_orders.setdefault(order_id, new_order)
                        logger.info("新增订单{}".format(new_order))
                elif push_order.get('x') == 'CANCELED':
                    cancel_ord_id = int(push_order['i'])
                    logger.info("订单已撤销{}".format(open_orders[cancel_ord_id]))
                    del open_orders[cancel_ord_id]
                elif push_order.get('x') == 'CALCULATED':
                    # 订单ADL或爆仓，参考撤单处理
                    ord_id = int(push_order['i'])
                    logger.info("订单ADL或爆仓,order_id:{}".format(ord_id))
                    if open_orders.get(ord_id) is not None:
                        logger.info("本地open_orders销毁该order{}".format(open_orders[ord_id]))
                        del open_orders[ord_id]
                elif push_order.get('x') == 'EXPIRED':
                    # 失效，参考撤单处理
                    ord_id = int(push_order['i'])
                    logger.info("订单失效{}".format(open_orders[ord_id]))
                    del open_orders[ord_id]
                else:
                    logger.info('未知的交易类型推送,push_order：{}'.format(push_order))
            else:
                logger.info('非本交易对交易推送,symbol：{s}'.format(s=push_order['s']))
        elif t == 'ACCOUNT_UPDATE':
            # 账户持仓更新（未限制交易对）
            logger.info('收到账户变动推送时间：{t}'.format(t=message.get('E')))
            push_account = message.get('a')
            logger.info('更新balance/position内容，事件原因：{t}'.format(t=push_account['m']))
            global my_balance, my_position
            push_balance = push_account.get('B')
            push_position = push_account.get('P')
            if push_balance is not None and len(push_balance) != 0:
                # 余额覆盖更新
                my_balance = push_balance
                logger.info('余额已更新，当前my_balance:{}'.format(my_balance))
            if push_position is not None and len(push_balance) != 0:
                # 持仓覆盖更新
                my_position = push_position
                logger.info('持仓已更新，当前my_position:{}'.format(my_balance))
        else:
            logger.info('未知推送：{t}'.format(t=message))


def start_wb_account():
    logger.info('开启WebsocketClient监听')
    global listen_key
    listen_key = get_listen_key()
    logger.info("Receving listen key : {}".format(listen_key))
    ws_client = UMFuturesWebsocketClient()
    ws_client.start()
    ws_client.user_data(listen_key=listen_key, id=3, callback=message_account)


def task():
    logger.info('开始启动定时任务')

    job_defaults = {'max_instances': 99}
    sched = BackgroundScheduler(timezone='MST', job_defaults=job_defaults)
    # 添加延长listen_key任务，时间间隔为50分钟
    sched.add_job(renew_listen_key, 'interval', minutes=50, id='renew_listen_key')
    # 添加新建listen_key任务，时间间隔为20小时
    # sched.add_job(new_listen_key, 'interval', hours=20, id='new_listen_key')

    sched.start()

    logger.info('启动定时任务结束')

def renew_listen_key():
    if try_renew_listen_key(listen_key) is True:
        logger.info('定时任务try_renew_listen_key触发时间为{t},本次延长后的listen_key为：{l}'.format
                     (t=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), l=listen_key))
    else:
        logger.info('ERROR!定时任务try_renew_listen_key触发时间为{t},延长失败'.format
                     (t=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), l=listen_key))


def strategy_sub():
    # 行情信息
    global depth
    depth = {'trading_time':0, "bid": [["29244.60", "3.965"], ["29244.50", "0.002"], ["29244.40", "0.001"]], "ask":[["29244.70", "28.569"], ["29244.80", "2.478"], ["29244.90", "0.837"]]}

    global trade
    trade = {'symbol': 'null', 'price': 0.0, 'quantity': 0.0, 'trading_time': 0, 'is_maker': None, 'daily_volume': 0.0, 'daily_amount': 0.0}

    # 持仓信息
    global open_orders
    open_orders = {3456789876: {"order_id": 3456789876, "client_ord_id": "whd_02", "dir": 'BUY', "org_qty": 500.24 ,"cur_qty": 244.32, "price": 0.343, "work_time": '(13位时间戳)'}}

    global balance
    balance = {'USDT': {'asset': 'USDT', 'wallet_balance': '199.93564633', 'cross_wallet': '199.93564633',
                        'balance_change': '0', 'trading_time': '(13位时间戳)'}}
    global position
    position = {'ETHUSDT': {'symbol': 'ETHUSDT', 'position_amount': 0.200, 'entry_price': 1640.91000000,
                            'cumulative_realized': 0.16890000, 'unrealized': -0.00536252, 'margin_type': 'cross',
                            'isolated_wallet': 0, 'position_side': 'BOTH', 'monetary_unit': 'USDT',
                            'average entry price': 1640.91000000, 'break_even_price': 1706.08216,
                            'trading_time': '(13位时间戳)'}}

    split_count = 5
    place_rate = 2 / 10000
    capital = 100
    pos_rate = 0.3  # 持仓比例

    depth = []
    trade = []
    y_pred_side_list = []
    y_pred_out_list = []



    last_time = int(time.time())
    old_sec = 0
    kill_time = 0

    threshold = 130000
    side_long = 0.9
    side_short = 0.1
    out = 0.8

    base_path = '/tmp/strdt/deployment/songhe/'

    symbol = 'ETHUSDT'

    model_side_0 = joblib.load('{}/{}/{}_lightGBM_side_0.pkl'.format(base_path,symbol, symbol))
    model_side_1 = joblib.load('{}/{}/{}_lightGBM_side_1.pkl'.format(base_path,symbol, symbol))
    model_side_2 = joblib.load('{}/{}/{}_lightGBM_side_2.pkl'.format(base_path,symbol, symbol))
    model_side_3 = joblib.load('{}/{}/{}_lightGBM_side_3.pkl'.format(base_path,symbol, symbol))
    model_side_4 = joblib.load('{}/{}/{}_lightGBM_side_4.pkl'.format(base_path,symbol, symbol))
    model_out_0 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path,symbol, symbol))
    model_out_1 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path,symbol, symbol))
    model_out_2 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path,symbol, symbol))
    model_out_3 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path,symbol, symbol))
    model_out_4 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path,symbol, symbol))

    closetime = depth['trading_time']//100 *100 +99
    depth_dict = {'closetime': depth['trading_time']//100 *100 +99,
                  'ask_price1': depth['ask'][0][0],'ask_size1': depth['ask'][0][1],'bid_price1': depth['bid'][0][0],'bid_size1': depth['bid'][0][1],
                  'ask_price2': depth['ask'][1][0], 'ask_size2': depth['ask'][1][1], 'bid_price2': depth['bid'][1][0],'bid_size2': depth['bid'][1][1],
                  'ask_price3': depth['ask'][2][0], 'ask_size3': depth['ask'][2][1], 'bid_price3': depth['bid'][2][0],'bid_size3': depth['bid'][2][1],
                  'ask_price4': depth['ask'][3][0], 'ask_size4': depth['ask'][3][1], 'bid_price4': depth['bid'][3][0],'bid_size4': depth['bid'][3][1],
                  'ask_price5': depth['ask'][4][0], 'ask_size5': depth['ask'][4][1], 'bid_price5': depth['bid'][4][0],'bid_size5': depth['bid'][4][1],
                  'ask_price6': depth['ask'][5][0], 'ask_size6': depth['ask'][5][1], 'bid_price6': depth['bid'][5][0],'bid_size6': depth['bid'][5][1],
                  'ask_price7': depth['ask'][6][0], 'ask_size7': depth['ask'][6][1], 'bid_price7': depth['bid'][6][0],'bid_size7': depth['bid'][6][1],
                  'ask_price8': depth['ask'][7][0], 'ask_size8': depth['ask'][7][1], 'bid_price8': depth['bid'][7][0],'bid_size8': depth['bid'][7][1],
                  'ask_price9': depth['ask'][8][0], 'ask_size9': depth['ask'][8][1], 'bid_price9': depth['bid'][8][0],'bid_size9': depth['bid'][8][1],
                  'ask_price10': depth['ask'][9][0], 'ask_size10': depth['ask'][9][1], 'bid_price10': depth['bid'][9][0],'bid_size10': depth['bid'][9][1],
    }
    depth.append(depth_dict)



    trade_dict = {'closetime': trade['trading_time']//100 *100 +99,
                  'price':trade['price'], 'size': trade['quantity'], 'is_maker': trade['is_maker'], 'volume': trade['daily_volume'], 'amount': trade['daily_amount']

    }
    trade.append(trade_dict)

    if closetime:
        time_10 = int(closetime / 1000)
        interval_time = 60000 * 40  # 提前储存40分钟数据用于计算因子
        if depth[-1]['closetime'] - depth[0][
            'closetime'] > interval_time and time_10 - last_time > 0.999:
            last_time = time_10
            len_depth = int(len(depth) * 0.99)
            diff_time = depth[-1]['closetime'] - depth[-len_depth]['closetime']
            if diff_time > interval_time:
                depth = depth[-len_depth:]
            len_trade = int(len(trade) * 0.99)
            if trade[-1]['closetime'] - trade[-len_trade]['closetime'] > interval_time:
                trade = trade[-len_trade:]

            df_depth = pd.DataFrame(depth)
            df_trade = pd.DataFrame(trade)

            df_trade['datetime'] = pd.to_datetime(df_trade['closetime'] + 28800000, unit='ms')
            df_trade = df_trade.groupby(pd.Grouper(key='datetime', freq='30s')).apply(vwap_30s)
            df_trade = df_trade.set_index('datetime').groupby(pd.Grouper(freq='1D')).apply(cumsum)
            df_trade['size'] = np.where(df_trade['is_maker'] == False, (-1) * df_trade['size'], df_trade['size'])
            # trade_df = trade_df.loc[:, ['closetime', 'price', 'size']]
            del df_trade['is_maker']
            # trade_df = trade_df.set_index('datetime').groupby(pd.Grouper(freq='1D')).apply(cumsum)
            df_trade = df_trade.reset_index(drop=True)
            # 100ms数据trade和depth合并
            data_merge = pd.merge(df_depth, df_trade, on='closetime', how='outer')
            data_merge = data_merge.sort_values(by='closetime', ascending=True)
            data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
            data_merge['sec'] = data_merge['datetime'].dt.second
            closetime_sec = time.localtime(closetime / 1000).tm_sec

            if closetime_sec != old_sec:
                if data_merge['sec'].iloc[-1] != data_merge['sec'].iloc[-2]:
                    old_sec = closetime_sec
                    tick1 = data_merge.iloc[:-1, :]
            # 取这一秒内最后一条切片为这个1s的点
            tick1s = tick1.set_index('datetime').groupby(pd.Grouper(freq='1000ms')).apply('last')

            tick1s = tick1s.drop_duplicates(subset=['closetime'], keep='last')
            tick1s = tick1s.dropna(subset=['ask_price1'])
            trade = tick1s.loc[:, ['closetime', 'price', 'size', 'volume', 'amount', 'vwap_30s']]
            depth = tick1s.loc[:, ['closetime',
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
            factor = add_factor_process(depth=depth, trade=trade)
            # factor['datetime'] = pd.to_datetime(factor['closetime'] + 28800000, unit='ms')
            # 计算120s vwap price
            factor['amount'] = factor['amount'].fillna(method='ffill')
            # if time.time() - self.strategy_time > 30:
            #     print('每十分钟打印一次阈值:',factor['turnover'].iloc[-1] - factor['turnover'].iloc[-2],'时间:',tick.datetime, self.model_symbol)
            #     self.strategy_time = time.time()
            if factor['amount'].iloc[-1] - factor['amount'].iloc[-2] >= threshold:
                print('bar采样触发阈值时间:', closetime, '品种:')
                signal = factor.iloc[-1:, :]
                X_test = np.array(signal.iloc[:, 9:84]).reshape(1, -1)

                y_pred_side_0 = model_side_0.predict(X_test, num_iteration=model_side_0.best_iteration)
                y_pred_side_1 = model_side_1.predict(X_test, num_iteration=model_side_1.best_iteration)
                y_pred_side_2 = model_side_2.predict(X_test, num_iteration=model_side_2.best_iteration)
                y_pred_side_3 = model_side_3.predict(X_test, num_iteration=model_side_3.best_iteration)
                y_pred_side_4 = model_side_4.predict(X_test, num_iteration=model_side_4.best_iteration)
                y_pred_side = (y_pred_side_0[0] + y_pred_side_1[0] + y_pred_side_2[0] + y_pred_side_3[0] +
                               y_pred_side_4[0]) / 5
                y_pred_side_list.append([y_pred_side])
                msg_ = f'批式方向信号:{y_pred_side_list[-1]}--time:{closetime}---symbol:{symbol}'
                log.info(msg_)

                y_pred_side_df = pd.DataFrame(y_pred_side_list, columns=['predict'])

                if y_pred_side_df['predict'].iloc[-1] > side_long or y_pred_side_df['predict'].iloc[
                    -1] < side_short:
                    y_pred_out_0 = model_out_0.predict(X_test, num_iteration=model_out_0.best_iteration)
                    y_pred_out_1 = model_out_1.predict(X_test, num_iteration=model_out_1.best_iteration)
                    y_pred_out_2 = model_out_2.predict(X_test, num_iteration=model_out_2.best_iteration)
                    y_pred_out_3 = model_out_3.predict(X_test, num_iteration=model_out_3.best_iteration)
                    y_pred_out_4 = model_out_4.predict(X_test, num_iteration=model_out_4.best_iteration)
                    y_pred_out = (y_pred_out_0[0] + y_pred_out_1[0] + y_pred_out_2[0] + y_pred_out_3[0] +
                                  y_pred_out_4[0]) / 5
                    y_pred_out_list.append([y_pred_out])
                    y_pred_out_df = pd.DataFrame(y_pred_out_list, columns=['out'])
                    msg_ = f'入场信号:{y_pred_out_list[-1]}-----time:{closetime}---symbol:{symbol}'
                    log.info(msg_)

                    # 策略逻辑
                    pos = position[symbol]['position_amount']
                    price = factor['vwap_30s'].iloc[-1]  # 挂单价格
                    position_value = pos * price  # 持仓金额
                    place_value = capital * pos_rate / split_count  # 挂单金额
                    buy_size = round(place_value / depth['ask'][0][0], 8)  # 买单量
                    sell_size = round(place_value / depth['bid'][0][0], 8)  # 卖单量
                    max_limited_order_value = capital * pos_rate  # 最大挂单金额

                    # 计算挂单金额
                    limit_orders_values = 0
                    final_values = 0
                    for key in open_orders.keys():
                        limit_orders_values += float(open_orders[key].price) * abs(float(open_orders[key].cur_qty))
                    # 持仓金额+挂单金额
                    final_values = limit_orders_values + position[symbol]['entry_price'] * pos

                    # 平多仓
                    if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                            y_pred_out_df['out'].iloc[-1]) >= out and pos > 0:
                        # print('-------------平仓之前撤销所有订单-------------')
                        test_um_trade_cancel_open_orders()
                        # print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                        bid_price_3 = depth['bid'][2][0]
                        um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='LONG', quantity=pos, price=bid_price_3)
                        msg_ = f'下空单平多仓---平仓价格:{bid_price_3}---size:{pos}---time:{closetime}---symbol:{symbol}'
                        log.info(msg_)

                    # 平空仓
                    if float(y_pred_side_df['predict'].iloc[-1]) >= self.side_long and float(
                            y_pred_out_df['out'].iloc[-1]) >= self.out and self.pos < 0:
                        # print('-------------平仓之前撤销所有订单-------------')
                        test_um_trade_cancel_open_orders()
                        # print('-----------------------------下多单平空仓----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                        ask_price_3 = depth['ask'][2][0]
                        um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='SHORT', quantity=pos,
                                                 price=ask_price_3)
                        msg_ = f'下多单平空仓---平仓价格:{ask_price_3}---size:{pos}---time:{closetime}---symbol:{symbol}'
                        self.log.info(msg_)

                    # 开空仓
                    if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                            y_pred_out_df['out'].iloc[-1]) >= out and position_value >= -pos_rate * capital * (
                            1 - 1 / split_count):
                        if len(open_orders) > 0:
                            last_order = list(open_orders.keys())[-1]
                            if open_orders[last_order].cur_qty > 0:
                                test_um_trade_cancel_open_orders()
                        # print('--------------开空仓----------------', '品种:',self.model_symbol)
                        if max_limited_order_value <= final_values * 1.0001:
                            test_um_trade_cancel_open_orders()

                        um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='SHORT', quantity=sell_size,
                                                 price=price)
                        msg_ = f'开空仓---开仓价格:{price * (1 - place_rate)}---size:{sell_size}---time:{closetime}---symbol:{symbol}'
                        log.info(msg_)
                        # self.fill_order_time = tick.closetime

                    # 开多仓
                    if float(y_pred_side_df['predict'].iloc[-1]) >= self.side_long and float(
                            y_pred_out_df['out'].iloc[-1]) >= self.out and position_value <= pos_rate * capital * (
                            1 - 1 / split_count):
                        # 如果此时有挂空单，全部撤掉
                        if len(open_orders) > 0:
                            last_order = list(open_orders.keys())[-1]
                            if open_orders[last_order].cur_qty < 0:
                                test_um_trade_cancel_open_orders()
                        # print('--------------开多仓----------------', '品种:',self.model_symbol)
                        if max_limited_order_value <= final_values * 1.0001:
                            test_um_trade_cancel_open_orders()
                        um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='LONG', quantity=buy_size,
                                                 price=price)
                        msg_ = f'开多仓---开仓价格:{price * (1 + place_rate)}---size:{buy_size}---time:{closetime}---symbol:{symbol}'
                        log.info(msg_)
                        # self.fill_order_time = tick.closetime

                        # return





                    # 保持仓位
                    else:
                        return



























if __name__ == '__main__':
    logger.add("./log/xrpusdt_{time}.log", rotation="1 day", enqueue=True, encoding='utf-8')
    if test_get_key() is False:
        logger.error('未成功获取账户api-key，策略开启失败')
    else:
        logger.info('初始化depth:{}'.format(depth))
        start_wb_market()   # 启动行情
        start_wb_account()  # 启动账户信息监测
        time.sleep(1)
        # initial()           # 初始化策略
        task()              # 启动所有定时任务
        logger.info('循环启动账户监听线程')
        while True:
            time.sleep(3600)
            logger.info('阻塞主线程，维持定时任务')


