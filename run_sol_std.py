#!/usr/strategy_main/env python
import pandas as pd
import numpy as np  # linear algebra
import redis, json, time, sys, os, joblib, threading
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from common.HFT_factor_3 import add_factor_process
from common.functions import um_trade_get_orders, um_trade_account, um_trade_new_market_order, cur_time, message_ding, \
    um_trade_change_leverage, get_symbol_daily_from_redis, um_trade_get_position_mode, um_trade_change_position_mode, \
    create_client_order_id, message_deal_part_order_message, message_deal_all_order_message, message_new_order, message_cancel_order
from common.functions import um_trade_cancel_open_orders, um_trade_new_limit_order

import warnings
warnings.filterwarnings("ignore")       # 忽略所有警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.")

redis_client = redis.Redis(host='localhost', port=6379, db=0, password='ht@20230717')

name = 'sub_second'
symbol = 'solusdt'

# 止损止盈开关
stop_profit_flag = True

# 提前储存分钟行情
interval_min = 45 * 2

threshold = 120000
side_long = 0.11676876769002788
side_short = 0.08133578859459789
out = 0.12168416168804841

lever_age = 4
balance = {}
positions = {}
open_orders = {}
symbol_daily = {}
strategy_run_flag = False
risk_flag = False

trade_acc_dicts = []

df_tick1s = pd.DataFrame(columns=['closetime'] + [f'ask_price{i + 1}' for i in range(10)] +[f'ask_size{i + 1}' for i in range(10)] +
                                 [f'bid_price{i + 1}' for i in range(10)] + [f'bid_size{i + 1}' for i in range(10)] +
                                 ['price'] + ['size'] + ['volume'] + ['amount'])
strategy_factor_arg = {
    'y_pred_out_list': [],
    'y_pred_side_list': [],
}   # 初始化策略参数

trade_acc_lock = threading.Lock()
acc_condition = threading.Condition(trade_acc_lock)

def trade_broadcast(channel):
    global symbol_daily, trade_acc_dicts
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
            price = float(trade_record['price'])
            quantity = float(trade_record['amount'])

            # 变更symbol_daily处理
            update_daily_volume = round(symbol_daily['daily_volume'] + quantity, 8)
            update_daily_amount = round(symbol_daily['daily_amount'] + (quantity * price), 8)
            symbol_daily['daily_volume'] = update_daily_volume
            symbol_daily['daily_amount'] = update_daily_amount

            trade_dict = {'closetime': int(trade_record['even_time']) // 100 * 100 + 99,
                        'price': float(trade_record['price']),
                        'size': float(trade_record['amount']) * (-1.0) if str(trade_record['side']) == 'sell' else float(trade_record['amount']),
                        'volume': update_daily_volume,
                        'amount': update_daily_amount}

            cl_time_sec = int(trade_dict.get('closetime') / 1000)
            trade_dict['cl_time_sec'] = cl_time_sec
            with acc_condition:
                trade_acc_dicts.append(trade_dict)
                acc_condition.notify_all()

def depth_broadcast(channel):
    global trade_acc_dicts
    kill_time = 0   # 止盈止损参数
    depth_sub = redis_client.pubsub()
    depth_sub.subscribe(channel)

    prev_closetime = None                 # 上一个closetime
    depth_acc_dicts = []                # 用于累积同一秒钟的depth_dict

    # 循环获取消息
    for depth_msg in depth_sub.listen():
        if depth_msg.get('type') == 'subscribe' and type(depth_msg.get('data')) is int:
            logger.info('depth redis广播连接成功。depth_msg:{}'.format(depth_msg))
        elif depth_msg.get('type') == 'message':
            depth_record = json.loads(depth_msg['data'])
            depth_dict = {'closetime': int(depth_record['even_time']) // 100 * 100 + 99}
            for i in range(10):
                depth_dict[f'ask_price{i + 1}'] = float(depth_record['asks'][i]['price'])
                depth_dict[f'ask_size{i + 1}'] = float(depth_record['asks'][i]['amount'])
                depth_dict[f'bid_price{i + 1}'] = float(depth_record['bids'][i]['price'])
                depth_dict[f'bid_size{i + 1}'] = float(depth_record['bids'][i]['amount'])

            # 走一个止盈止损函数
            if risk_flag and stop_profit_flag:
                kill_time = stop_profit_loss(depth_dict, kill_time)

            cl_time_sec = int(depth_dict.get('closetime') / 1000)
            if prev_closetime is None:              # 如果是第一个depth_dict
                prev_closetime = cl_time_sec
            if cl_time_sec > prev_closetime:        # 如果closetime推送到了新的一秒钟
                if depth_acc_dicts:
                    # 需要用depth_sec_key去trade_acc_dicts里捞数据
                    depth_sec_key = int(depth_acc_dicts[-1]['cl_time_sec'])
                    trade_target_list = []
                    trade_del_list = []
                    for trade_dict in trade_acc_dicts:
                        if int(trade_dict['cl_time_sec']) < depth_sec_key:
                            trade_del_list.append(trade_dict)
                        elif int(trade_dict['cl_time_sec']) == depth_sec_key:
                            trade_target_list.append(trade_dict)
                            trade_del_list.append(trade_dict)
                        else:
                            continue
                    with acc_condition:
                        # 清除trade_acc_dicts的多余元素
                        for trade_dict in trade_del_list:
                            trade_acc_dicts.remove(trade_dict)
                        acc_condition.notify_all()
                    # merge逻辑
                    if trade_target_list:
                        data = [(trade['closetime'], trade['price'], trade['size'], trade['volume'], trade['amount']) for trade in trade_target_list]
                        df_trade = pd.DataFrame(data, columns=['closetime', 'price', 'size', 'volume', 'amount'])

                        df_depth_item = [pd.DataFrame.from_dict(d, orient='index').T.drop(columns='cl_time_sec') for d in depth_acc_dicts]
                        df_depth = pd.concat(df_depth_item, ignore_index=True)

                        data_merge = pd.merge(df_depth, df_trade, on='closetime', how='inner')
                        if data_merge.empty:
                            last_trade_dict = depth_acc_dicts[-1]
                            last_trade_dict['price'] = float('nan')
                            last_trade_dict['size'] = float('nan')
                            last_trade_dict['volume'] = float('nan')
                            last_trade_dict['amount'] = float('nan')
                            last_record = pd.DataFrame.from_records(last_trade_dict, index=[0]).drop(columns='cl_time_sec')
                            df_tick1s_concat(last_record)
                        else:
                            data_merge = data_merge.sort_values(by='closetime', ascending=True)
                            last_record = data_merge.iloc[-1:]
                            df_tick1s_concat(last_record)
                    else:
                        last_trade_dict = depth_acc_dicts[-1]
                        last_trade_dict['price'] = float('nan')
                        last_trade_dict['size'] = float('nan')
                        last_trade_dict['volume'] = float('nan')
                        last_trade_dict['amount'] = float('nan')
                        last_record = pd.DataFrame.from_records(last_trade_dict, index=[0]).drop(columns='cl_time_sec')
                        df_tick1s_concat(last_record)
                    depth_acc_dicts = []
                prev_closetime = cl_time_sec
            depth_dict['cl_time_sec'] = cl_time_sec
            depth_acc_dicts.append(depth_dict)    # 将depth_dict添加到累积列表中

def df_tick1s_concat(last_record):
    global df_tick1s, strategy_run_flag
    last_record['datetime'] = pd.to_datetime(last_record['closetime'] + 28800000, unit='ms')
    last_record_set = last_record.set_index('datetime')
    last_record_set['closetime'] = last_record_set['closetime'].astype('int64')

    df_tick1s = pd.concat([df_tick1s, last_record_set])

    df_tick1s_size = interval_min * 60  # 设置最大允许的条目数量
    if len(df_tick1s) > df_tick1s_size:
        df_tick1s = df_tick1s.iloc[1:]  # 抛弃最前面的一条记录
        strategy_run_flag = True

def open_orders_broadcast(channel):
    op_ord_sub = redis_client.pubsub()
    op_ord_sub.subscribe(channel)
    # 循环获取消息
    for op_ord_msg in op_ord_sub.listen():
        if op_ord_msg.get('type') == 'subscribe' and type(op_ord_msg.get('data')) is int:
            logger.info('open_orders redis广播连接成功。op_ord_msg:{}'.format(op_ord_msg))
        elif op_ord_msg.get('type') == 'message':
            open_orders_record = json.loads(op_ord_msg['data'])
            t = open_orders_record.get('e')
            if t is not None:
                # 过滤事件类型
                if t == 'ORDER_TRADE_UPDATE':
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
                                message_deal_part_order_message(redis_client, name, symbol, order)
                                logger.info("订单部成{}".format(order))
                            elif push_order['X'] == 'FILLED':  # 全成处理
                                trade_ord_id = int(push_order['i'])
                                del_order = open_orders[trade_ord_id]
                                del open_orders[trade_ord_id]
                                message_deal_all_order_message(redis_client, name, symbol, del_order)
                                logger.info("订单全成{}".format(del_order))
                            else:
                                message_ding(redis_client, '捕获到未知成交类型推送,push_order:{}'.format(push_order))
                                logger.error('未知成交类型,push_order:{}'.format(push_order))
                        elif push_order.get('x') == 'NEW':
                            # 强平订单发生(目前只给日志，不给处理)
                            client_ord_id = str(push_order['c'])
                            if 'autoclose' in client_ord_id:
                                message_ding(redis_client, "发生强平订单！push_order:{}".format(push_order))
                                logger.error("发生强平订单！push_order:{}".format(push_order))
                            elif 'adl_autoclose' in client_ord_id:
                                message_ding(redis_client, "订单发生ADL，push_order:{}".format(push_order))
                                logger.error("订单发生ADL，push_order:{}".format(push_order))
                            else:
                                # 新订单发生 (限价和市价情况是一样的，市价也会产生new和trade推送)
                                order_id = int(push_order['i'])
                                new_order = {"order_id": order_id, "client_ord_id": client_ord_id, "dir": push_order['S'],
                                             "org_qty": float(push_order['q']), "cur_qty": float(push_order['q']),
                                             "price": float(push_order['p']), "work_time": int(push_order['T'])}
                                open_orders.setdefault(order_id, new_order)
                                message_new_order(redis_client, name, symbol, new_order)
                                logger.info("新增订单{}".format(new_order))
                        elif push_order.get('x') == 'CANCELED':
                            cancel_ord_id = int(push_order['i'])
                            can_order = open_orders[cancel_ord_id]
                            del open_orders[cancel_ord_id]
                            message_cancel_order(redis_client, name, symbol, can_order)
                            logger.info("订单已撤销{}".format(can_order))
                        elif push_order.get('x') == 'CALCULATED':
                            # 订单ADL或爆仓，参考撤单处理
                            ord_id = int(push_order['i'])
                            if open_orders.get(ord_id) is not None:
                                logger.info("本地open_orders销毁该order{}".format(open_orders[ord_id]))
                                del open_orders[ord_id]
                            message_ding(redis_client, "订单ADL或爆仓,order_id:{}".format(ord_id))
                            logger.info("订单ADL或爆仓,order_id:{}".format(ord_id))
                        elif push_order.get('x') == 'EXPIRED':
                            # 失效，参考撤单处理
                            ord_id = int(push_order['i'])
                            del_order = open_orders[ord_id]
                            del open_orders[ord_id]
                            message_ding(redis_client, "订单失效{}".format(del_order))
                            logger.info("订单失效{}".format(del_order))
                        else:
                            message_ding(redis_client, '捕获到未知的交易类型推送,push_order：{}'.format(push_order))
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
            logger.info('balance redis广播连接成功。op_ord_msg:{}'.format(bal_msg))
        elif bal_msg.get('type') == 'message':
            balance_record = json.loads(bal_msg['data'])
            global balance
            balance = balance_record
            logger.info('收到redis balance推送，更新完成后的balance:{}'.format(balance))

def risk_broadcast(channel):
    risk_sub = redis_client.pubsub()
    risk_sub.subscribe(channel)
    global risk_flag
    for risk_msg in risk_sub.listen():
        if risk_msg.get('type') == 'subscribe' and type(risk_msg.get('data')) is int:
            logger.info('RISK redis广播连接成功。op_ord_msg:{}'.format(risk_msg))
            risk_flag = True                      # 策略只有连上risk监听后才允许开启
        elif risk_msg.get('type') == 'message':
            risk_record = json.loads(risk_msg['data'])
            logger.info('账户RISK收到消息:{}'.format(risk_record))
            if risk_record['type'] == 'STOP':
                if risk_record['account'] == name:
                    if risk_record['symbol'] == 'all' or risk_record['symbol'] == symbol:
                        msg = '账户:{},symbol:{},收到风控指令，立即停止策略并全部市价平仓'.format(name, symbol)
                        message_ding(redis_client, msg)
                        logger.info(msg)
                        risk_flag = False

def position_broadcast(channel):
    pos_sub = redis_client.pubsub()
    pos_sub.subscribe(channel)
    for pos_msg in pos_sub.listen():
        if pos_msg.get('type') == 'subscribe' and type(pos_msg.get('data')) is int:
            logger.info('position redis广播连接成功。op_ord_msg:{}'.format(pos_msg))
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
    # 查看账户所有挂单，撤销所有挂单
    if len(um_trade_get_orders(name, symbol)) != 0:
        if um_trade_cancel_open_orders(name, symbol):
            logger.info("账户已撤下所有挂单")
        else:
            logger.error("挂单未撤成功，请查看日志")
    else:
        logger.info("账户已经无任何挂单")

    # 查看该交易对的持仓情况
    symbol_name = symbol.upper()
    pos_try = positions.get(symbol_name)
    if pos_try is not None:
        # 存在持仓-进行市价平仓(只撤此symbol的仓位)
        pos_amt = positions[symbol_name]['position_amount']
        if pos_amt < 0.0:
            logger.info('交易对:{}存在空仓pos_amt={},进行市价平空仓'.format(symbol_name, pos_amt))
            um_trade_new_market_order(name, symbol, 'BUY', 'BOTH', abs(pos_amt), ('ult' + create_client_order_id()))
        else:
            logger.info('交易对:{}存在多仓pos_amt={},进行市价平多仓'.format(symbol_name, pos_amt))
            um_trade_new_market_order(name, symbol, 'SELL', 'BOTH', abs(pos_amt), ('ult' + create_client_order_id()))

def stop_profit_loss(depth_dict, kill_time):
    closetime = depth_dict['closetime']
    symbol_name = symbol.upper()
    pos_try = positions.get(symbol_name)
    if pos_try is not None:
        pos = positions[symbol_name]['position_amount']
        pos_entry_price = positions[symbol_name]['entry_price']
    else:
        pos = 0.0
        pos_entry_price = 0.0

    ask_price_1 = depth_dict['ask_price1']
    bid_price_1 = depth_dict['bid_price1']

    up_kill_time = kill_time

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
                up_kill_time = closetime
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
                up_kill_time = closetime
                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=abs(pos),
                                         price=ask_price_1, newClientOrderId='sp_short' + create_client_order_id())
    return up_kill_time

def strategy(name, symbol, open_orders, balance, positions, tick1s):
    global strategy_factor_arg
    y_pred_out_list = strategy_factor_arg['y_pred_out_list']
    y_pred_side_list = strategy_factor_arg['y_pred_side_list']

    split_count = 5
    place_rate = 3 / 10000
    capital = balance['USDT']['wallet_balance']
    pos_rate = 1  # 持仓比例

    model_side = []
    model_out = []
    model_path = '/home/ubuntu/binance-market/crypto_saved_model/'
    for i in range(5):
        model_side.append(joblib.load('{}/{}/{}_lightGBM_side_{}.pkl'.format(model_path, symbol, symbol, i)))
        model_out.append(joblib.load('{}/{}/{}_lightGBM_out_{}.pkl'.format(model_path, symbol, symbol, i)))

    tick1s = tick1s.drop_duplicates(subset=['closetime'], keep='last')
    tick1s = tick1s.dropna(subset=['ask_price1'])

    trade_df = tick1s.loc[:, ['closetime', 'price', 'size', 'volume', 'amount']]
    columns = ['closetime']
    for i in range(1, 11):
        columns.extend(['ask_price{}'.format(i), 'ask_size{}'.format(i), 'bid_price{}'.format(i), 'bid_size{}'.format(i)])
    depth_df = tick1s.loc[:, columns]

    # 计算因子
    factor = add_factor_process(depth=depth_df, trade=trade_df, min=40)

    factor['vwap_2s'] = (factor['price'].fillna(method='ffill') * abs(factor['size'].fillna(method='ffill'))).rolling(2).sum() / abs(
        factor['size'].fillna(method='ffill')).rolling(2).sum()
    factor['amount'] = factor['amount'].fillna(method='ffill')

    last_depth_dict = df_tick1s.iloc[-1].to_dict()

    closetime = last_depth_dict['closetime']
    if factor['amount'].iloc[-1] - factor['amount'].iloc[-2] >= threshold:
        logger.info('bar采样触发阈值时间:{},品种:{}'.format(closetime, symbol))
        signal = factor.iloc[-1:, :]
        X_test = np.array(signal.iloc[:, 5:105]).reshape(1, -1)
        y_pred_side_temp = []
        for i in range(5):
            y_pred_side_it = model_side[i].predict(X_test, num_iteration=model_side[i].best_iteration)
            y_pred_side_temp.append(y_pred_side_it[0])
        y_pred_side_avg = sum(y_pred_side_temp[i] * [0.03, 0.05, 0.07, 0.2, 0.65][i] for i in range(5)) / 5
        y_pred_side_list.append([y_pred_side_avg])
        msg_ = f'批式方向信号:{y_pred_side_list[-1]}--time:{closetime}---symbol:{symbol}'
        logger.info(msg_)

        y_pred_side_df = pd.DataFrame(y_pred_side_list, columns=['predict'])

        if y_pred_side_df['predict'].iloc[-1] > side_long or y_pred_side_df['predict'].iloc[-1] < side_short:
            y_pred_out_temp = []
            for i in range(5):
                y_pred_out_it = model_out[i].predict(X_test, num_iteration=model_out[i].best_iteration)
                y_pred_out_temp.append(y_pred_out_it[0])
            y_pred_out_avg = sum(y_pred_out_temp[i] * [0.03, 0.05, 0.07, 0.2, 0.65][i] for i in range(5)) / 5
            y_pred_out_list.append([y_pred_out_avg])
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

            price = factor['vwap_2s'].iloc[-1]  # 挂单价格
            position_value = pos * price  # 持仓金额
            place_value = capital * pos_rate / split_count  # 挂单金额

            buy_size = round(place_value / last_depth_dict['ask_price1'], 8)  # 买单量
            sell_size = round(place_value / last_depth_dict['bid_price1'], 8)  # 卖单量
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
                msg_ = f'下空单平多仓---平仓价格:{price * (1 - place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH', quantity=abs(pos),
                                         price=price * (1 - place_rate), newClientOrderId='cl_long' + create_client_order_id())
            # 平空仓
            if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                    y_pred_out_df['out'].iloc[-1]) >= out and pos < 0:
                # print('-------------平仓之前撤销所有订单-------------')
                um_trade_cancel_open_orders(name, symbol)
                msg_ = f'下多单平空仓---平仓价格:{price * (1 + place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH', quantity=abs(pos),
                                         price=price * (1 + place_rate), newClientOrderId='cl_short' + create_client_order_id())
            # 开空仓
            if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                y_pred_out_df['out'].iloc[-1]) >= out and position_value >= -pos_rate * capital * (1 - 1 / split_count):
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
            # 开多仓
            if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                y_pred_out_df['out'].iloc[-1]) >= out and position_value <= pos_rate * capital * (1 - 1 / split_count):
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
    # 更新因子部分参数
    strategy_factor_arg['y_pred_out_list'] = y_pred_out_list
    strategy_factor_arg['y_pred_side_list'] = y_pred_side_list
    return None

def main():
    logger.info('策略mian()开始执行')
    if redis_connect() is False:
        sys.exit(0)
    # 拉起账户风控推送接听
    risk_channel = 'chal_{n}_risk'.format(n=name)
    risk_thread = threading.Thread(target=risk_broadcast, args=(risk_channel,))
    risk_thread.start()
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
    logger.info('====累计行情数据，时长{}min===='.format(interval_min))
    global strategy_run_flag
    while risk_flag:
        if strategy_run_flag:
            strategy_thread = threading.Thread(target=strategy, args=(name, symbol, open_orders, balance, positions, df_tick1s))
            strategy_thread.start()
            strategy_thread.join()
            strategy_run_flag = False
    # 终止策略流程
    ultimate()
    message_ding(redis_client, '账户:{},symbol:{},策略已关闭，仓位已平，结束进程'.format(name, symbol))
    logger.info('本策略已被终止，结束进程')
    sys.exit(0)

if __name__ == '__main__':
    logger.add("./log/solusdt_sub_{time}.log", rotation="1 day", enqueue=True, encoding='utf-8')
    main()



