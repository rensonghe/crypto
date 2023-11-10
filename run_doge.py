#!/usr/bin/env python
import pandas as pd
import numpy as np  # linear algebra
import redis, json, time, sys, gc, joblib, threading
from loguru import logger

from HFT_factor_sol import add_factor_process

from functions import um_trade_get_orders, um_trade_account, um_trade_new_market_order, cur_time, \
    um_trade_change_leverage, get_symbol_daily_from_redis, um_trade_get_position_mode, um_trade_change_position_mode, \
    create_client_order_id

from functions import um_trade_cancel_open_orders, um_trade_new_limit_order

# 设置展示所有行和列
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# from memory_profiler import profile
# import memory_profiler

import warnings         # 消除groupby()函数警告
warnings.simplefilter(action='ignore', category=FutureWarning)

from send_message import message_deal_part_order_message, \
    message_deal_all_order_message, message_new_order, message_cancel_order, send_ding

redis_client = redis.Redis(host='localhost', port=6379, db=0, password='ht@20230717')

name = 'sub_second'
symbol = 'dogeusdt'
listen_key = 'null'

lever_age = 2
trade = {}
depth = {}
balance = {}
positions = {}
open_orders = {}
symbol_daily = {}
strategy_run_flag = False

strategy_read_arg = {
    'strategy_depth': [],
    'strategy_trade': [],
    'last_time': int(time.time()),
    'old_sec': 0
}  # 初始化策略参数

strategy_factor_arg = {
    'y_pred_out_list': [],
    'y_pred_side_list': [],
    'kill_time': 0
}   # 初始化策略参数

def trade_broadcast(channel):
    global trade, symbol_daily, strategy_run_flag
    trade_sub = redis_client.pubsub()
    trade_sub.subscribe(channel)

    # 循环获取消息
    for trade_msg in trade_sub.listen():
        if trade_msg.get('type') == 'subscribe' and type(trade_msg.get('data')) is int:
            logger.info('trade redis广播连接成功。trade_msg:{}'.format(trade_msg))
            # 再次更新
            redis_volume, redis_amount, is_complete = get_symbol_daily_from_redis(redis_client, symbol)
            symbol_daily['daily_volume'] = redis_volume
            symbol_daily['daily_amount'] = redis_amount
            logger.info('再次刷新成功，symbol_daily:{}'.format(symbol_daily))

        elif trade_msg.get('type') == 'message':
            trade_record = json.loads(trade_msg['data'])
            trade['symbol'] = str(trade_record['symbol']).upper()
            trade['side'] = str(trade_record['side'])
            trade['even_time'] = int(trade_record['even_time'])
            price = float(trade_record['price'])
            quantity = float(trade_record['amount'])
            trade['price'] = price
            trade['quantity'] = quantity

            # 变更symbol_daily处理
            update_daily_volume = round(symbol_daily['daily_volume'] + quantity, 8)
            update_daily_amount = round(symbol_daily['daily_amount'] + (quantity * price), 8)
            trade['daily_volume'] = update_daily_volume
            trade['daily_amount'] = update_daily_amount
            symbol_daily['daily_volume'] = update_daily_volume
            symbol_daily['daily_amount'] = update_daily_amount

            strategy_run_flag = True

def depth_broadcast(channel):

    global depth, strategy_run_flag
    depth_sub = redis_client.pubsub()
    depth_sub.subscribe(channel)

    # 循环获取消息
    for depth_msg in depth_sub.listen():
        if depth_msg.get('type') == 'subscribe' and type(depth_msg.get('data')) is int:
            logger.info('depth redis广播连接成功。depth_msg:{}'.format(depth_msg))
        elif depth_msg.get('type') == 'message':
            depth_record = json.loads(depth_msg['data'])

            depth['even_time'] = int(depth_record['even_time'])
            depth['bids'] = [[float(bid['price']), float(bid['amount'])] for bid in depth_record['bids']]
            depth['asks'] = [[float(ask['price']), float(ask['amount'])] for ask in depth_record['asks']]

            strategy_run_flag = True


# 订阅广播频道并接收消息
def open_orders_broadcast(channel):
    op_ord_sub = redis_client.pubsub()
    op_ord_sub.subscribe(channel)

    # 循环获取消息
    for op_ord_msg in op_ord_sub.listen():
        # {'type': 'subscribe', 'pattern': None, 'channel': b'chal_ac-test_op_ord', 'data': 1}
        # {'type': 'message', 'pattern': None, 'channel': b'chal_ac-test_position', 'data': b'{}'}
        if op_ord_msg.get('type') == 'subscribe' and type(op_ord_msg.get('data')) is int:
            logger.info('open_orders redis广播连接成功。op_ord_msg:{}'.format(op_ord_msg))
        elif op_ord_msg.get('type') == 'message':
            open_orders_record = json.loads(op_ord_msg['data'])
            t = open_orders_record.get('e')
            if t is not None:
                # 过滤事件类型
                if t == 'ORDER_TRADE_UPDATE':
                    # logger.info('收到redis订单推送，even_time:{t}, local_time:{l}'.format(t=open_orders_record.get('E'), l=cur_time()))
                    push_order = open_orders_record.get('o')
                    # 过滤交易对
                    if push_order['s'] == symbol.upper():
                        global open_orders
                        if push_order.get('x') == 'TRADE':
                            if push_order['X'] == 'PARTIALLY_FILLED':  # 部成处理
                                trade_ord_id = int(push_order['i'])
                                order = open_orders.get(trade_ord_id)
                                order['cur_qty'] -= float(push_order['l'])
                                open_orders[trade_ord_id] = order
                                # 钉钉推送
                                send_ord_part_deal = threading.Thread(target=message_deal_part_order_message,
                                                                      args=(name, symbol, order))
                                send_ord_part_deal.start()
                                logger.info("订单部成{}".format(order))
                            elif push_order['X'] == 'FILLED':  # 全成处理
                                trade_ord_id = int(push_order['i'])
                                del_order = open_orders[trade_ord_id]
                                del open_orders[trade_ord_id]
                                # 钉钉推送
                                send_ord_all_deal = threading.Thread(target=message_deal_all_order_message,
                                                                     args=(name, symbol, del_order))
                                send_ord_all_deal.start()
                                logger.info("订单全成{}".format(del_order))
                            else:
                                # 钉钉推送
                                send_ding('捕获到未知成交类型推送,push_order:{}'.format(push_order))
                                logger.error('未知成交类型,push_order:{}'.format(push_order))
                        elif push_order.get('x') == 'NEW':
                            # 强平订单发生(目前只给日志，不给处理)
                            client_ord_id = str(push_order['c'])
                            if 'autoclose' in client_ord_id:
                                # 钉钉推送
                                send_ding("发生强平订单！push_order:{}".format(push_order))
                                logger.error("发生强平订单！push_order:{}".format(push_order))
                            elif 'adl_autoclose' in client_ord_id:
                                # 钉钉推送
                                send_ding("订单发生ADL，push_order:{}".format(push_order))
                                logger.error("订单发生ADL，push_order:{}".format(push_order))
                            else:
                                # 新订单发生 (限价和市价情况是一样的，市价也会产生new和trade推送)
                                order_id = int(push_order['i'])
                                new_order = {"order_id": order_id, "client_ord_id": client_ord_id, "dir": push_order['S'],
                                             "org_qty": float(push_order['q']), "cur_qty": float(push_order['q']),
                                             "price": float(push_order['p']), "work_time": int(push_order['T'])}
                                open_orders.setdefault(order_id, new_order)
                                # 钉钉推送
                                send_new_ord = threading.Thread(target=message_new_order, args=(name, symbol, new_order))
                                send_new_ord.start()
                                logger.info("新增订单{}".format(new_order))
                        elif push_order.get('x') == 'CANCELED':
                            cancel_ord_id = int(push_order['i'])
                            can_order = open_orders[cancel_ord_id]
                            del open_orders[cancel_ord_id]
                            # 钉钉推送
                            send_ord_cancel = threading.Thread(target=message_cancel_order, args=(name, symbol, can_order))
                            send_ord_cancel.start()
                            logger.info("订单已撤销{}".format(can_order))
                        elif push_order.get('x') == 'CALCULATED':
                            # 订单ADL或爆仓，参考撤单处理
                            ord_id = int(push_order['i'])
                            if open_orders.get(ord_id) is not None:
                                logger.info("本地open_orders销毁该order{}".format(open_orders[ord_id]))
                                del open_orders[ord_id]
                            # 钉钉推送
                            send_ding("订单ADL或爆仓,order_id:{}".format(ord_id))
                            logger.info("订单ADL或爆仓,order_id:{}".format(ord_id))
                        elif push_order.get('x') == 'EXPIRED':
                            # 失效，参考撤单处理
                            ord_id = int(push_order['i'])
                            del_order = open_orders[ord_id]
                            del open_orders[ord_id]
                            # 钉钉推送
                            send_ding("订单失效{}".format(del_order))
                            logger.info("订单失效{}".format(del_order))
                        else:
                            # 钉钉推送
                            send_ding('捕获到未知的交易类型推送,push_order：{}'.format(push_order))
                            logger.info('未知的交易类型推送,push_order：{}'.format(push_order))
                    else:
                        logger.info('非本交易对交易推送,symbol：{s}'.format(s=push_order['s']))
                else:
                    logger.error('在redis trade推送出现了非交易信息,请注意排查：{s}'.format(s=open_orders_record))

def balance_broadcast(channel):
    bal_sub = redis_client.pubsub()
    bal_sub.subscribe(channel)
    for bal_msg in bal_sub.listen():
        if bal_msg.get('type') == 'subscribe' and type(bal_msg.get('data')) is int:
            logger.info('open_orders redis广播连接成功。op_ord_msg:{}'.format(bal_msg))
        elif bal_msg.get('type') == 'message':
            balance_record = json.loads(bal_msg['data'])
            global balance
            balance = balance_record
            logger.info('收到redis balance推送，更新完成后的balance:{}'.format(balance))

def position_broadcast(channel):
    pos_sub = redis_client.pubsub()
    pos_sub.subscribe(channel)
    for pos_msg in pos_sub.listen():
        if pos_msg.get('type') == 'subscribe' and type(pos_msg.get('data')) is int:
            logger.info('open_orders redis广播连接成功。op_ord_msg:{}'.format(pos_msg))
        elif pos_msg.get('type') == 'message':
            positions_record = json.loads(pos_msg['data'])
            global positions
            positions = positions_record
            logger.info('收到redis positions推送，更新完成后的positions:{}'.format(positions))

def redis_connect():
    # redis拉取symbol_daily
    redis_volume, redis_amount, is_complete = get_symbol_daily_from_redis(redis_client, symbol)
    if is_complete is True:
        global symbol_daily
        symbol_daily['daily_volume'] = redis_volume
        symbol_daily['daily_amount'] = redis_amount
        logger.info('redis拉取symbol_daily成功，symbol_daily:{}'.format(symbol_daily))
        return True
    else:
        logger.error('symbol_daily数据不完整，不能启动策略。is_complete:{}'.format(is_complete))
        return False

def initial():
    logger.info('初始化策略开始执行')

    # 查看账户所有挂单，撤销所有挂单
    if len(um_trade_get_orders(name, symbol)) != 0:
        if um_trade_cancel_open_orders(name, symbol):
            logger.info("已撤下所有挂单")
        else:
            logger.error("挂单未撤成功，策略终止")
            return False


    # 查询持仓模式，将其变为单向持仓
    if um_trade_get_position_mode(name):
        if um_trade_change_position_mode(name) is False:
            logger.error('账户变更为单向持仓失败，策略终止')
            return False
        else:
            logger.info('账户已由双向变更为单向持仓')
    else:
        logger.info('账户已为单向持仓，不需变更')

    # 获取账户资产v2
    org_acc = um_trade_account(name)
    # 余额处理
    org_balance_list = org_acc.get('assets')
    global balance
    for asset_item in org_balance_list:
        # 过滤asset为0的币种
        if float(asset_item['walletBalance']) != 0.0:
            asset_name = str(asset_item['asset'])
            asset_dict = {
                'asset': asset_name,
                'wallet_balance': float(asset_item['walletBalance']),
                'cross_wallet': float(asset_item['crossWalletBalance']),
                'balance_change': 0.0,
                'trading_time': 0}
            balance.setdefault(asset_name, asset_dict)
    logger.info('账户资产更新成功，balance:{}'.format(balance))
    if len(balance) == 0:
        logger.error('账户资产为空，策略终止')
        return False

    # 持仓处理
    org_positions_list = org_acc.get('positions')
    global positions
    for pos_item in org_positions_list:
        # 过滤持仓数量(amount)为0的交易对
        if float(pos_item['positionAmt']) != 0.0:
            symbol_name = str(pos_item['symbol'])
            if bool(pos_item['isolated']) is True:
                margin_type = 'isolated'
            else:
                margin_type = 'unknow'
            pos_dict = {
                'symbol': symbol_name,
                'position_amount': float(pos_item['positionAmt']),
                'entry_price': float(pos_item['entryPrice']),          # 持仓均价
                'cumulative_realized': 0.0,
                'unrealized_profit': float(pos_item['unrealizedProfit']),
                'margin_type': margin_type,
                'isolated_wallet': 0.0,
                'position_side': 'unknow',
                'break_even_price': 0.0,
                'trading_time': 0}
            positions.setdefault(symbol_name, pos_dict)
    logger.info('持仓更新完成，positions:{}'.format(positions))

    symbol_name = symbol.upper()
    pos_try = positions.get(symbol_name)
    if pos_try is not None:
        # 存在持仓-只check此symbol的仓位
        logger.error('持仓{}不为空，判定为初始化错误，策略终止'.format(symbol_name))
        return False

    # 调整杠杆数额
    um_trade_change_leverage(name, symbol, int(lever_age))

    logger.info('初始化策略结束执行')
    return True


def ultimate():
    # 查看该交易对的持仓情况
    symbol_name = symbol.upper()
    pos_try = positions.get(symbol_name)
    if pos_try is not None:
        # 存在持仓-进行平仓(只撤此symbol的仓位)
        pos_amt = positions[symbol_name]['position_amount']
        if pos_amt < 0.0:
            logger.info('交易对:{}存在空仓pos_amt={},进行市价平空仓'.format(symbol_name, pos_amt))
            um_trade_new_market_order(name, symbol, 'BUY', 'BOTH', abs(pos_amt), ('ult' + create_client_order_id()))
        else:
            logger.info('交易对:{}存在多仓pos_amt={},进行市价平多仓'.format(symbol_name, pos_amt))
            um_trade_new_market_order(name, symbol, 'SELL', 'BOTH', abs(pos_amt), ('ult' + create_client_order_id()))

    # 查看账户所有挂单，撤销所有挂单
    if len(um_trade_get_orders(name, symbol)) != 0:
        if um_trade_cancel_open_orders(name, symbol):
            logger.info("账户已撤下所有挂单")
        else:
            logger.error("挂单未撤成功，请查看日志")
    else:
        logger.info("账户已经无任何挂单")

def read_data(name, symbol, depth, trade, open_orders, balance, positions):
    global strategy_read_arg
    old_sec = strategy_read_arg['old_sec']
    last_time = strategy_read_arg['last_time']
    strategy_depth = strategy_read_arg['strategy_depth']
    strategy_trade = strategy_read_arg['strategy_trade']

    closetime = depth['even_time'] // 100 * 100 + 99
    depth_dict = {'closetime': depth['even_time'] // 100 * 100 + 99,
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

    trade_dict = {'closetime': trade['even_time'] // 100 * 100 + 99,
                  'price': trade['price'], 'size': trade['quantity'], 'side': trade['side'],
                  'volume': trade['daily_volume'], 'amount': trade['daily_amount']
                  }

    strategy_trade.append(trade_dict)

    closetime = depth['even_time'] // 100 * 100 + 99

    time_10 = int(closetime / 1000)
    # print(time_10, 'now time')
    # print(strategy_depth[-1]['closetime'], 'last closetime')
    # print(strategy_depth[0]['closetime'], 'first closetime')
    # print(time_10 - last_time)
    interval_time = 60000 * 45 * 2  # 提前储存45分钟数据用于计算因子


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
        df_trade['size'] = np.where(df_trade['side'] == 'sell', (-1) * df_trade['size'], df_trade['size'])
        # trade_df = trade_df.loc[:, ['closetime', 'price', 'size']]
        del df_trade['side']
        # trade_df = trade_df.set_index('datetime').groupby(pd.Grouper(freq='1D'), group_keys=True).apply(cumsum)
        # df_trade = df_trade.reset_index(drop=True)
        # 100ms数据trade和depth合并
        data_merge = pd.merge(df_depth, df_trade, on='closetime', how='outer')
        data_merge = data_merge.sort_values(by='closetime', ascending=True)
        data_merge['datetime'] = pd.to_datetime(data_merge['closetime'] + 28800000, unit='ms')
        data_merge['sec'] = data_merge['datetime'].dt.second
        # logger.info(data_merge['sec'])
        # print(data_merge)
        closetime_sec = time.localtime(closetime / 1000).tm_sec
        # print(closetime_sec,'closetime_sec')
        # print(data_merge['sec'].iloc[-1],'sec')
        # print(old_sec,'old_sec')
        if closetime_sec != old_sec:
            if data_merge['sec'].iloc[-1] != data_merge['sec'].iloc[-2]:
                # 取这一秒内最后一条切片为这个1s的点
                tick1 = data_merge.iloc[:-1, :]
                # logger.info('------tick1---------:{}'.format(tick1))
                old_sec = closetime_sec
                tick1s = tick1.set_index('datetime').groupby(pd.Grouper(freq='1000ms'), group_keys=True).apply('last')
                # logger.info('---------tick1s-----------')
                # logger.info(tick1s)
                # logger.info("tick1s['amount'].iloc[-1]{}".format(tick1s['amount'].iloc[-1]))
                # 调用因子
                # s = cur_time()
                strategy(name, symbol, depth, trade, open_orders, balance, positions, tick1s)
                # e = cur_time()
                # logger.info('调用strategy函数start:{},end:{},e-s:{}'.format(s, e, e-s))

    # 更新全局变量
    strategy_read_arg['old_sec'] = old_sec
    strategy_read_arg['last_time'] = last_time
    strategy_read_arg['strategy_depth'] = strategy_depth
    strategy_read_arg['strategy_trade'] = strategy_trade

def strategy(name, symbol, depth, trade, open_orders, balance, positions, tick1s):
    global strategy_factor_arg
    y_pred_out_list = strategy_factor_arg['y_pred_out_list']
    y_pred_side_list = strategy_factor_arg['y_pred_side_list']
    kill_time = strategy_factor_arg['kill_time']

    split_count = 5
    place_rate = 3 / 10000
    capital = balance['USDT']['wallet_balance']
    pos_rate = 1  # 持仓比例

    base_path = '/home/ubuntu/binance-market/crypto_saved_model'

    threshold = 90000
    side_long = 0.12092782762534751
    side_short = 0.07501047315895487
    out = 0.15677866362435128
    model_side_0 = joblib.load('{}/{}/{}_lightGBM_side_0.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_side_1 = joblib.load('{}/{}/{}_lightGBM_side_1.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_side_2 = joblib.load('{}/{}/{}_lightGBM_side_2.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_side_3 = joblib.load('{}/{}/{}_lightGBM_side_3.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_side_4 = joblib.load('{}/{}/{}_lightGBM_side_4.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_out_0 = joblib.load('{}/{}/{}_lightGBM_out_0.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_out_1 = joblib.load('{}/{}/{}_lightGBM_out_1.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_out_2 = joblib.load('{}/{}/{}_lightGBM_out_2.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_out_3 = joblib.load('{}/{}/{}_lightGBM_out_3.pkl'.format(base_path, 'solusdt', 'solusdt'))
    model_out_4 = joblib.load('{}/{}/{}_lightGBM_out_4.pkl'.format(base_path, 'solusdt', 'solusdt'))

    closetime = depth['even_time'] // 100 * 100 + 99
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
    if int(closetime) - int(kill_time) > 100:
        # 多仓止盈止损
        if pos > 0:
            # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
            pf = float(bid_price_1 / abs(pos_entry_price)) - 1
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
                um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=abs(pos),
                                         price=bid_price_1, newClientOrderId='sp_long'+create_client_order_id())

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
                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=abs(pos),
                                         price=ask_price_1, newClientOrderId='sp_short' + create_client_order_id())


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
    # ot = cur_time()
    # 计算因子
    factor = add_factor_process(depth=depth_df, trade=trade_df, min=40)
    # logger.info('factor.iloc[-5:, :]{}'.format(factor.iloc[-5:, :]))
    # df = factor.iloc[-5:, :]
    # logger.info("factor['amount'].iloc[-1]{}".format(factor['amount'].iloc[-1]))
    # factor['amount'].iloc[-1]
    # oc = cur_time()
    # logger.info('因子计算start:{},end:{},e-t:{}'.format(ot, oc, oc-ot))

    # factor['datetime'] = pd.to_datetime(factor['closetime'] + 28800000, unit='ms')
    # 计算 5s vwap price
    factor['vwap_5s'] = (factor['price'].fillna(0) * abs(factor['size'].fillna(0))).rolling(5).sum() / abs(
        factor['size'].fillna(0)).rolling(5).sum()
    factor['amount'] = factor['amount'].fillna(method='ffill')
    # logger.info("===debug=== factor['amount']{}".format(factor['amount'].iloc[-1]))
    # if time.time() - self.strategy_time > 30:
    #     print('每十分钟打印一次阈值:',factor['turnover'].iloc[-1] - factor['turnover'].iloc[-2],'时间:',tick.datetime, self.model_symbol)
    #     self.strategy_time = time.time()
    if factor['amount'].iloc[-1] - factor['amount'].iloc[-2] >= threshold:
        logger.info('bar采样触发阈值时间:{},品种:{}'.format(closetime, symbol))
        signal = factor.iloc[-1:, :]
        X_test = np.array(signal.iloc[:, 5:93]).reshape(1, -1)

        y_pred_side_0 = model_side_0.predict(X_test, num_iteration=model_side_0.best_iteration)
        y_pred_side_1 = model_side_1.predict(X_test, num_iteration=model_side_1.best_iteration)
        y_pred_side_2 = model_side_2.predict(X_test, num_iteration=model_side_2.best_iteration)
        y_pred_side_3 = model_side_3.predict(X_test, num_iteration=model_side_3.best_iteration)
        y_pred_side_4 = model_side_4.predict(X_test, num_iteration=model_side_4.best_iteration)
        y_pred_side = (y_pred_side_0[0] * 0.03 + y_pred_side_1[0]*0.05 + y_pred_side_2[0]*0.07 + y_pred_side_3[0]*0.2 +
                       y_pred_side_4[0] * 0.65) / 5
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
            y_pred_out = (y_pred_out_0[0]*0.03 + y_pred_out_1[0]*0.05 + y_pred_out_2[0]*0.07 + y_pred_out_3[0]*0.2 + y_pred_out_4[0]*0.65) / 5
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
                limit_orders_values += float(open_orders[key]['price']) * abs(
                    float(open_orders[key]['cur_qty']))
            # 持仓金额+挂单金额

            final_values = limit_orders_values + pos_entry_price * pos

            # 平多仓
            if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                    y_pred_out_df['out'].iloc[-1]) >= out and pos > 0:
                # print('-------------平仓之前撤销所有订单-------------')
                um_trade_cancel_open_orders(name, symbol)
                # print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                bid_price_3 = depth['bids'][2][0]
                msg_ = f'下空单平多仓---平仓价格:{price * (1 - place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=abs(pos),
                                         price=price * (1 - place_rate), newClientOrderId='cl_long' + create_client_order_id())
            # 平空仓
            if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                    y_pred_out_df['out'].iloc[-1]) >= out and pos < 0:
                # print('-------------平仓之前撤销所有订单-------------')
                um_trade_cancel_open_orders(name, symbol)
                # print('-----------------------------下多单平空仓----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                ask_price_3 = depth['asks'][2][0]
                msg_ = f'下多单平空仓---平仓价格:{price * (1 + place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=abs(pos),
                                         price=price * (1 + place_rate), newClientOrderId='cl_short' + create_client_order_id())
            # 开空仓
            if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                y_pred_out_df['out'].iloc[-1]) >= out \
                    and position_value >= -pos_rate * capital * (1 - 1 / split_count):
                if len(open_orders) > 0:
                    last_order = list(open_orders.keys())[-1]
                    if open_orders[last_order]['dir'] == 'BUY':
                        msg_ = f'此时挂有多弹---同一时刻有反向单---撤单'
                        logger.info(msg_)
                        um_trade_cancel_open_orders(name, symbol)
                # print('--------------开空仓----------------', '品种:',self.model_symbol)
                if max_limited_order_value <= final_values * 1.0001:
                    msg_ = f'超出最大挂单量---撤单'
                    logger.info(msg_)
                    um_trade_cancel_open_orders(name, symbol)
                msg_ = f'开空仓---开仓价格:{price * (1 - place_rate)}---size:{sell_size}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=sell_size,
                                         price=price * (1 - place_rate), newClientOrderId='op_short' + create_client_order_id())
                # self.fill_order_time = tick.closetime

            # 开多仓
            if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                y_pred_out_df['out'].iloc[-1]) >= out and position_value <= pos_rate * capital * (
                1 - 1 / split_count):
                # 如果此时有挂空单，全部撤掉
                if len(open_orders) > 0:
                    last_order = list(open_orders.keys())[-1]
                    if open_orders[last_order]['dir'] == 'SELL':
                        msg_ = f'此时挂有空单---同一时刻有反向单---撤单'
                        logger.info(msg_)
                        um_trade_cancel_open_orders(name, symbol)
                # print('--------------开多仓----------------', '品种:',self.model_symbol)
                if max_limited_order_value <= final_values * 1.0001:
                    msg_ = f'超出最大挂单量---撤单'
                    logger.info(msg_)
                    um_trade_cancel_open_orders(name, symbol)
                msg_ = f'开多仓---开仓价格:{price * (1 + place_rate)}---size:{buy_size}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)

                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=buy_size,
                                         price=price * (1 + place_rate), newClientOrderId='op_long' + create_client_order_id())

                # self.fill_order_time = tick.closetime
                # return
            # else:
            # 保持仓位
            # return

    # 更新因子部分参数
    strategy_factor_arg['y_pred_out_list'] = y_pred_out_list
    strategy_factor_arg['y_pred_side_list'] = y_pred_side_list
    strategy_factor_arg['kill_time'] = kill_time
    return None


def main():
    logger.info('策略mian()开始执行')
    if redis_connect() is False:
        sys.exit(0)
    logger.info('初始化depth:{},trade:{}'.format(depth, trade))
    # 拉起行情推送接听
    trade_chal = 'bin-fur-' + symbol + '-trade'
    depth_chal = 'bin-fur-' + symbol + '-depth'
    trade_thread = threading.Thread(target=trade_broadcast, args=(trade_chal, ))
    depth_thread = threading.Thread(target=depth_broadcast, args=(depth_chal, ))
    trade_thread.start()
    depth_thread.start()
    time.sleep(1)
    if initial() is False:                  # 初始化账户
        sys.exit(0)
    # 拉起账户redis推送接听
    open_orders_channel = 'chal_{n}_op_ord'.format(n=name)
    balance_channel = 'chal_{n}_balance'.format(n=name)
    position_channel = 'chal_{n}_position'.format(n=name)
    op_ord_thread = threading.Thread(target=open_orders_broadcast, args=(open_orders_channel, ))
    balance_thread = threading.Thread(target=balance_broadcast, args=(balance_channel, ))
    position_thread = threading.Thread(target=position_broadcast, args=(position_channel,))
    op_ord_thread.start()
    balance_thread.start()
    position_thread.start()
    logger.info('====循环启动维持策略====')
    global strategy_run_flag
    while True:
        if strategy_run_flag:
            strategy_thread = threading.Thread(target=read_data, args=(name, symbol, depth, trade, open_orders, balance, positions))
            strategy_thread.start()
            strategy_thread.join()
            strategy_run_flag = False

if __name__ == '__main__':
    logger.add("./log/dogeusdt_{time}.log", rotation="1 day", enqueue=True, encoding='utf-8')
    main()



