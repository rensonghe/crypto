#!/usr/bin/env python
import pandas as pd
import numpy as np  # linear algebra
import redis, json, time, sys, gc, joblib, threading
from loguru import logger
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from HFT_factor_online import add_factor_process

# from dingdingtalk import message_update_position, message_del_position
from functions import um_trade_get_orders, um_trade_account, \
    um_trade_change_leverage, get_symbol_daily_from_redis, um_trade_get_position_mode, um_trade_change_position_mode, \
    um_stream_new_listen_key, um_stream_renew_listen_key, um_trade_new_market_order, um_stream_close_listen_key, cur_time

from functions import um_trade_cancel_open_orders, um_trade_new_limit_order

# 设置展示所有行和列
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

from memory_profiler import profile
import memory_profiler

import warnings         # 消除groupby()函数警告

from send_message import message_del_position, message_update_position, message_deal_part_order_message, \
    message_deal_all_order_message, message_new_order, message_cancel_order, message_currency

warnings.simplefilter(action='ignore', category=FutureWarning)

redis_client = redis.Redis(host='localhost', port=6379, db=0, password='ht@20230717')

name = 'ot-sub-test'
symbol = 'xrpusdt'
listen_key = 'null'
trade = {}
depth = {}
balance = {}
positions = {}
open_orders = {}
symbol_daily = {}
# strategy_allow = True
strategy_run_flag = False
listen_key_flag = False

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

def start_wb_market():
    depth_binance_client = UMFuturesWebsocketClient(on_message=message_depth)
    depth_binance_client.partial_book_depth(symbol=symbol, id=1, level=10, speed=0)
    logger.info('行情depth已启动')

    trade_binance_client = UMFuturesWebsocketClient(on_message=message_trade, is_combined=False)
    trade_binance_client.agg_trade(symbol=symbol)
    logger.info('行情trade已启动')

def message_depth(_, message):
    # cur_org = cur_time()
    push_depth = json.loads(message)
    global depth, strategy_run_flag
    # depth = {'trading_time': 0, 'bids': [], 'asks': []}
    trading_time = push_depth.get('T')
    if trading_time is not None:
        depth['trading_time'] = int(trading_time)
        depth['even_time'] = int(push_depth.get('E'))

        # 确保本条一定有且更新
        bids = push_depth.get('b')
        if bids is not None:
            depth['bids'] = [[float(price), float(quantity)] for price, quantity in push_depth['b']]

        asks = push_depth.get('a')
        if asks is not None:
            depth['asks'] = [[float(price), float(quantity)] for price, quantity in push_depth['a']]

        strategy_run_flag = True

        # logger.info('depth update,cur_org:{},local:{},even:{},trading:{},l-e:{},l-t:{}'
        #             .format(cur_org, cur_time(), int(push_depth.get('E')), int(push_depth.get('T')),
        #                     cur_time()-int(push_depth.get('E')), cur_time()-int(push_depth.get('T'))))


def message_trade(_, message):
    cur_org = cur_time()
    push_trade = json.loads(message)
    global trade, symbol_daily, strategy_run_flag
    # depth = {'trading_time': 0, 'bids': [], 'asks': []}
    # symbol_daily = {'daily_volume': 0.0, 'daily_amount': 0.0}

    # 首条再次读取symbol_daily，确保绝对同步
    fir = push_trade.get('id')
    if fir is not None:
        redis_volume, redis_amount, is_complete = get_symbol_daily_from_redis(redis_client, symbol)
        symbol_daily['daily_volume'] = redis_volume
        symbol_daily['daily_amount'] = redis_amount
        logger.info('再次刷新成功，symbol_daily:{}'.format(symbol_daily))

    # 非首条交易处理
    p = push_trade.get('p')
    if p is not None:
        trade['symbol'] = str(push_trade.get('s'))
        price = float(push_trade.get('p'))
        quantity = float(push_trade.get('q'))
        trade['price'] = price
        trade['quantity'] = quantity
        trade['trading_time'] = int(push_trade.get('T'))
        trade['even_time'] = int(push_trade.get('E'))
        trade['is_maker'] = push_trade.get('m')

        # 变更symbol_daily处理
        update_daily_volume = round(symbol_daily['daily_volume'] + quantity, 8)
        update_daily_amount = round(symbol_daily['daily_amount'] + (quantity * price), 8)
        trade['daily_volume'] = update_daily_volume
        trade['daily_amount'] = update_daily_amount
        symbol_daily['daily_volume'] = update_daily_volume
        symbol_daily['daily_amount'] = update_daily_amount

        strategy_run_flag = True

        # logger.info('trede update,cur_org:{},local:{},even:{},trading:{},l-e:{},l-t:{}'
        #             .format(cur_org, cur_time(), int(push_trade.get('E')), int(push_trade.get('T')),
        #                     cur_time()-int(push_trade.get('E')), cur_time()-int(push_trade.get('T'))))

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
        if um_trade_change_position_mode(name):
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
        logger.error('持仓不为空，判定为初始化错误，策略终止')
        return False

    # 调整杠杆数额
    um_trade_change_leverage(name, symbol, int(1))

    logger.info('初始化策略结束执行')
    return True


def message_account(_, message):
    push_account_message = json.loads(message)
    t = push_account_message.get('e')
    if t is not None:
        # 过滤事件类型
        if t == 'ORDER_TRADE_UPDATE':
            logger.info('收到订单推送，even_time:{t}, local_time:{l}'.format(t=push_account_message.get('E'), l=cur_time()))
            push_order = push_account_message.get('o')
            # 过滤交易对
            if push_order['s'] == symbol.upper():
                global open_orders
                if push_order.get('x') == 'TRADE':
                    if push_order['X'] == 'PARTIALLY_FILLED':      # 部成处理
                        trade_ord_id = int(push_order['i'])
                        order = open_orders.get(trade_ord_id)
                        order['cur_qty'] -= float(push_order['l'])
                        open_orders[trade_ord_id] = order
                        # 钉钉推送
                        send_ord_part_deal = threading.Thread(target=message_deal_part_order_message, args=(name, symbol, order))
                        send_ord_part_deal.start()
                        logger.info("订单部成{}".format(order))
                    elif push_order['X'] == 'FILLED':              # 全成处理
                        trade_ord_id = int(push_order['i'])
                        del_order = open_orders[trade_ord_id]
                        del open_orders[trade_ord_id]
                        # 钉钉推送
                        send_ord_all_deal = threading.Thread(target=message_deal_all_order_message, args=(name, symbol, del_order))
                        send_ord_all_deal.start()
                        logger.info("订单全成{}".format(del_order))
                    else:
                        # 钉钉推送
                        send_unkunow_deal = threading.Thread(target=message_currency, args=('捕获到未知成交类型推送,push_order:{}'.format(push_order)))
                        send_unkunow_deal.start()
                        logger.error('未知成交类型,push_order:{}'.format(push_order))
                elif push_order.get('x') == 'NEW':
                    # 强平订单发生(目前只给日志，不给处理)
                    client_ord_id = str(push_order['c'])
                    if 'autoclose' in client_ord_id:
                        # 钉钉推送
                        send_autoclose = threading.Thread(target=message_currency, args=("发生强平订单！push_order:{}".format(push_order)))
                        send_autoclose.start()
                        logger.error("发生强平订单！push_order:{}".format(push_order))
                    elif 'adl_autoclose' in client_ord_id:
                        # 钉钉推送
                        send_adl_autoclose = threading.Thread(target=message_currency, args=("订单发生ADL，push_order:{}".format(push_order)))
                        send_adl_autoclose.start()
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
                    # 钉钉推送
                    send_calculated = threading.Thread(target=message_currency, args=("订单ADL或爆仓,order_id:{}".format(ord_id)))
                    send_calculated.start()
                    if open_orders.get(ord_id) is not None:
                        logger.info("本地open_orders销毁该order{}".format(open_orders[ord_id]))
                        del open_orders[ord_id]
                    logger.info("订单ADL或爆仓,order_id:{}".format(ord_id))
                elif push_order.get('x') == 'EXPIRED':
                    # 失效，参考撤单处理
                    ord_id = int(push_order['i'])
                    del_order = open_orders[ord_id]
                    del open_orders[ord_id]
                    # 钉钉推送
                    send_expired = threading.Thread(target=message_currency, args=("订单失效{}".format(del_order)))
                    send_expired.start()
                    logger.info("订单失效{}".format(del_order))
                else:
                    # 钉钉推送
                    send_unkunow = threading.Thread(target=message_currency, args=('捕获到未知的交易类型推送,push_order：{}'.format(push_order)))
                    send_unkunow.start()
                    logger.info('未知的交易类型推送,push_order：{}'.format(push_order))
            else:
                logger.info('非本交易对交易推送,symbol：{s}'.format(s=push_order['s']))
        elif t == 'ACCOUNT_UPDATE':
            # 账户持仓更新（未限制交易对）
            logger.info('收到账户变动推送,even_time:{t},local_time:{l}'.format(t=push_account_message.get('E'), l=cur_time()))
            push_account = push_account_message.get('a')
            logger.info('更新balance/position内容，事件原因：{t}'.format(t=push_account['m']))
            global balance, positions
            push_balance = push_account.get('B')
            push_position = push_account.get('P')
            if push_balance is not None and len(push_balance) != 0:
                for push_bal_item in push_balance:
                    push_asset_name = str(push_bal_item['a'])
                    push_wb = float(push_bal_item['wb'])
                    if push_wb == 0.0:                                      # 资产为0，做删除处理
                        del balance[push_asset_name]
                    else:                                                   # 非0做更新处理
                        asset_update = {
                            'asset': push_asset_name,
                            'wallet_balance': float(push_bal_item['wb']),
                            'cross_wallet': float(push_bal_item['cw']),
                            'balance_change': float(push_bal_item['bc']),
                            'trading_time': int(push_account_message.get('T'))}
                        balance[push_asset_name] = asset_update
                # 钉钉推送
                send_balance_update = threading.Thread(target=message_currency, args=('收到balance更新，当前balance:{}'.format(balance)))
                send_balance_update.start()
                logger.info('余额已更新，当前balance:{}'.format(balance))
            if push_position is not None and len(push_position) != 0:
                for push_pos_item in push_position:
                    push_symbol_name = str(push_pos_item['s'])
                    push_pos_amt = float(push_pos_item['pa'])
                    if push_pos_amt == 0.0:                                 # 持仓为0，做删除处理
                        del positions[push_symbol_name]
                        pos_del = {
                            'symbol': push_symbol_name,
                            'position_amount': push_pos_amt,
                            'entry_price': float(push_pos_item['ep']),
                            'cumulative_realized': float(push_pos_item['cr']),
                            'unrealized_profit': float(push_pos_item['up']),
                            'margin_type': str(push_pos_item['mt']),
                            'isolated_wallet': float(push_pos_item['iw']),
                            'position_side': str(push_pos_item['ps']),
                            'break_even_price': float(push_pos_item['bep']),
                            'trading_time': int(push_account_message.get('T'))}
                        # 钉钉推送
                        send_del_pos = threading.Thread(target=message_del_position, args=(name, pos_del))
                        send_del_pos.start()
                    else:                                                   # 非0做更新处理
                        pos_update = {
                            'symbol': push_symbol_name,
                            'position_amount': push_pos_amt,
                            'entry_price': float(push_pos_item['ep']),
                            'cumulative_realized': float(push_pos_item['cr']),
                            'unrealized_profit': float(push_pos_item['up']),
                            'margin_type': str(push_pos_item['mt']),
                            'isolated_wallet': float(push_pos_item['iw']),
                            'position_side': str(push_pos_item['ps']),
                            'break_even_price': float(push_pos_item['bep']),
                            'trading_time': int(push_account_message.get('T'))}
                        positions[push_symbol_name] = pos_update
                        # 钉钉推送
                        send_update_pos = threading.Thread(target=message_update_position, args=(name, pos_update))
                        send_update_pos.start()
                        # message_update_position(name, pos_update)
                logger.info('持仓已更新，当前positions:{}'.format(positions))
        else:
            # 钉钉推送
            send_un = threading.Thread(target=message_currency, args=('未知推送,message:{t}'.format(t=push_account_message)))
            send_un.start()
            logger.info('未知推送,local_time:{l},message:{t}'.format(l=cur_time(), t=push_account_message))

def start_wb_account():
    logger.info('开启账户WebsocketClient监听')
    # 开启 listen_key
    global listen_key, listen_key_flag
    listen_key = um_stream_new_listen_key(name)
    logger.info("Receving listen key : {}".format(listen_key))
    if listen_key != 'null':
        ws_client = UMFuturesWebsocketClient(on_message=message_account)
        ws_client.user_data(listen_key=listen_key, id=1)
        listen_key_flag = True
        logger.info('开启账户WebsocketClient监听成功')
    else:
        logger.error('开启账户WebsocketClient监听失败')
        listen_key_flag = False
        return False

def restart_listen_key():
    logger.info('重置账户WebsocketClient监听')
    global listen_key, listen_key_flag
    # 停止策略循环
    listen_key_flag = False
    logger.info('策略因更换账户监听listen_key而暂停')
    # 关闭账户WebsocketClient
    um_stream_close_listen_key(name, listen_key)
    listen_key = 'null'
    time.sleep(0.01)
    # 账户WebsocketClient监听
    if start_wb_account() is False:
        logger.error('重置账户WebsocketClient监听失败')
    else:
        logger.info('重置账户WebsocketClient监听成功')


def task():
    logger.info('开始启动定时任务')

    job_defaults = {'max_instances': 99}
    sched = BackgroundScheduler(timezone='MST', job_defaults=job_defaults)
    # 添加延长listen_key任务，时间间隔为50分钟
    sched.add_job(renew_listen_key, 'interval', minutes=50, id='renew_listen_key')
    # 添加重建listen_key任务，时间间隔为20小时
    sched.add_job(restart_listen_key, 'interval', hours=20, id='new_listen_key')

    sched.start()

    logger.info('启动定时任务结束')


def renew_listen_key():
    if um_stream_renew_listen_key(name, listen_key) is True:
        logger.info('定时任务try_renew_listen_key,本次延长后的listen_key为：{l}'.format(l=listen_key))
    else:
        logger.error('ERROR!定时任务try_renew_listen_key,延长失败，请查看日志。'.format(l=listen_key))

def ultimate():
    # 查看该交易对的持仓情况
    symbol_name = symbol.upper()
    pos_try = positions.get(symbol_name)
    if pos_try is not None:
        # 存在持仓-进行平仓(只撤此symbol的仓位)
        pos_amt = positions[symbol_name]['position_amount']
        if pos_amt < 0.0:
            logger.info('交易对:{}存在空仓pos_amt={},进行市价平空仓'.format(symbol_name, pos_amt))
            um_trade_new_market_order(name, symbol, 'BUY', 'BOTH', abs(pos_amt))
        else:
            logger.info('交易对:{}存在多仓pos_amt={},进行市价平多仓'.format(symbol_name, pos_amt))
            um_trade_new_market_order(name, symbol, 'SELL', 'BOTH', abs(pos_amt))

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
    interval_time = 60000 * 45  # 提前储存45分钟数据用于计算因子


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
    place_rate = 2 / 10000
    capital = balance['USDT']['wallet_balance']
    pos_rate = 1.6  # 持仓比例

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
    model_out_1 = joblib.load('{}/{}/{}_lightGBM_out_1.pkl'.format(base_path, symbol, symbol))
    model_out_2 = joblib.load('{}/{}/{}_lightGBM_out_2.pkl'.format(base_path, symbol, symbol))
    model_out_3 = joblib.load('{}/{}/{}_lightGBM_out_3.pkl'.format(base_path, symbol, symbol))
    model_out_4 = joblib.load('{}/{}/{}_lightGBM_out_4.pkl'.format(base_path, symbol, symbol))

    closetime = depth['trading_time'] // 100 * 100 + 99
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
    if int(closetime ) - int(kill_time) > 1000:
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
                                         price=bid_price_1)

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
                                         price=ask_price_1)

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

    ot = cur_time()
    # 计算因子
    factor = add_factor_process(depth=depth_df, trade=trade_df, min=20)
    # logger.info('factor.iloc[-5:, :]{}'.format(factor.iloc[-5:, :]))
    # df = factor.iloc[-5:, :]
    # logger.info("factor['amount'].iloc[-1]{}".format(factor['amount'].iloc[-1]))
    # factor['amount'].iloc[-1]
    oc = cur_time()
    logger.info('因子计算start:{},end:{},e-t:{}'.format(ot, oc, oc-ot))

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
                um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH',
                                         quantity=abs(pos), price=price * (1 - place_rate))
            # 平空仓
            if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                    y_pred_out_df['out'].iloc[-1]) >= out and pos < 0:
                # print('-------------平仓之前撤销所有订单-------------')
                um_trade_cancel_open_orders(name, symbol)
                # print('-----------------------------下多单平空仓----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
                ask_price_3 = depth['asks'][2][0]
                msg_ = f'下多单平空仓---平仓价格:{price * (1 + place_rate)}---size:{pos}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH',
                                         quantity=abs(pos), price=price * (1 + place_rate))
            # 开空仓
            if float(y_pred_side_df['predict'].iloc[-1]) <= side_short and float(
                y_pred_out_df['out'].iloc[-1]) >= out \
                    and position_value >= -pos_rate * capital * (1 - 1 / split_count):
                if len(open_orders) > 5:
                    last_order = list(open_orders.keys())[-1]
                    if open_orders[last_order]['dir'] > 'BUY':
                        um_trade_cancel_open_orders(name, symbol)
                # print('--------------开空仓----------------', '品种:',self.model_symbol)
                if max_limited_order_value <= final_values * 1.0001:
                    um_trade_cancel_open_orders(name, symbol)
                msg_ = f'开空仓---开仓价格:{price * (1 - place_rate)}---size:{sell_size}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)
                um_trade_new_limit_order(name, symbol=symbol, side='SELL', positionSide='BOTH',
                                         quantity=sell_size, price=price * (1 - place_rate))
                # self.fill_order_time = tick.closetime

            # 开多仓
            if float(y_pred_side_df['predict'].iloc[-1]) >= side_long and float(
                y_pred_out_df['out'].iloc[-1]) >= out and position_value <= pos_rate * capital * (
                1 - 1 / split_count):
                # 如果此时有挂空单，全部撤掉
                if len(open_orders) > 5:
                    last_order = list(open_orders.keys())[-1]
                    if open_orders[last_order]['dir'] == 'SELL':
                        um_trade_cancel_open_orders(name, symbol)
                # print('--------------开多仓----------------', '品种:',self.model_symbol)
                if max_limited_order_value <= final_values * 1.0001:
                    um_trade_cancel_open_orders(name, symbol)
                msg_ = f'开多仓---开仓价格:{price * (1 + place_rate)}---size:{buy_size}---time:{closetime}---symbol:{symbol}'
                logger.info(msg_)

                um_trade_new_limit_order(name, symbol=symbol, side='BUY', positionSide='BOTH',
                                         quantity=buy_size, price=price * (1 + place_rate))

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
    start_wb_market()                       # 启动行情
    if start_wb_account() is False:         # 启动账户信息监测
        sys.exit(0)
    time.sleep(1)
    if initial() is False:                  # 初始化账户
        sys.exit(0)
    task()                                  # 启动所有定时任务
    logger.info('====循环启动维持策略====')
    global strategy_run_flag, listen_key_flag, strategy_arg
    while True:
        if strategy_run_flag and listen_key_flag:
            strategy_thread = threading.Thread(target=read_data, args=(name, symbol, depth, trade, open_orders, balance, positions))
            # c = cur_time()
            strategy_thread.start()
            strategy_thread.join()
            # e = cur_time()
            # logger.info('行情s:{},行情e:{},s-t:{}'.format(c, e, e-c))
            # gc.collect()
            # logger.info('本次垃圾回收用时:{}'.format(cur_time()-e))
            strategy_run_flag = False

if __name__ == '__main__':
    logger.add("./log/xrpusdt_{time}.log", rotation="1 day", enqueue=True, encoding='utf-8')
    main()



