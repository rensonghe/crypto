import requests, json, random, pymysql, time, sys
from loguru import logger

futures_url = 'http://127.0.0.1:5050/um_futures/'

def cur_time():
    return int(round(time.time() * 1000))

def fsizeof(i):
    return int(sys.getsizeof(i))

# api-name放在本地server的json里
def test_get_key():
    logger.info('开始读取账户api信息')

    with open("key.json", 'r') as load_f:
        load_dict = json.load(load_f)
        load_f.close()
    if type(load_dict) is dict:
        logger.info('读取账户api信息成功:api-name:{n}'.format(n=load_dict['name']))
        return True
    else:
        logger.info('读取账户api信息失败,策略终止，load_dict{l}'.format(l=load_dict))
        return False

def get_key():
    with open("key.json", 'r') as load_f:
        load_dict = json.load(load_f)
        load_f.close()
    if type(load_dict) is dict:
        return load_dict['name']


# u账户和交易-查看当前全部挂单
def um_trade_get_orders(name, symbol):
    u = 'trade/get_orders'
    headers = {'name': name}
    data = {'symbol': symbol}
    logger.info('u账户和交易-查看当前全部挂单:{}'.format(data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info(r.json())
    return r.json()


# u账户和交易-撤销单一交易对的所有挂单
def um_trade_cancel_open_orders(name, symbol):
    u = 'trade/cancel_open_orders'
    headers = {'name': name}
    data = {'symbol': symbol}
    logger.info('u账户和交易-撤销单一交易对的所有挂单:{}'.format(data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info(r.json())
    if r.json()['code'] == 200:
        return True
    else:
        return False

# u账户和交易-账户信息V2
def um_trade_account(name):
    u = 'trade/account'
    headers = {'name': name}
    logger.info('u账户和交易-账户信息V2:')
    r = requests.post(url=futures_url + u, headers=headers)
    # logger.info(r.json())
    return r.json()

# u账户和交易-调整开仓杠杆
def um_trade_change_leverage(name, symbol, leverage):
    u = 'trade/change_leverage'
    headers = {'name': name}
    data = {'symbol': symbol, 'leverage': leverage}
    logger.info('u账户和交易-调整开仓杠杆:{}'.format(data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    if r.json().get('leverage') == leverage:
        logger.info('杠杆已调整至目标:{}'.format(leverage))
    else:
        logger.info('杠杆调整未完成，请查看日志:{}'.format(r.json()))


# u账户和交易-查询持仓模式
def um_trade_get_position_mode(name):
    u = 'trade/get_position_mode'
    headers = {'name': name}
    logger.info('u账户和交易-查询持仓模式')
    r = requests.post(url=futures_url + u, headers=headers)
    logger.info(r.json())
    return bool(r.json().get('dualSidePosition'))

# u账户和交易-更改持仓模式为单向
def um_trade_change_position_mode(name):
    u = 'trade/change_position_mode'
    headers = {'name': name}
    data = {'dualSidePosition': 'false'}    # false单向，true双向
    logger.info('u账户和交易-更改持仓模式为单向:{}'.format(data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info(r.json())
    if r.json().get('code') == 200:
        return True
    else:
        return False


# u账户和交易-现价下单
def um_trade_new_limit_order(name, symbol, side, positionSide, quantity, price):
    u = 'trade/new_order'
    headers = {'name': name}
    data = {'symbol': symbol,
            'side': side,
            'positionSide': positionSide,
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price
            }
    logger.info('u账户和交易-限价下单:local_time:{l},{d}'.format(l=cur_time(), d=data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info('订单回报:{},local_time:{}'.format(r.json(), cur_time()))

# u账户和交易-市价下单
def um_trade_new_market_order(name, symbol, side, positionSide, quantity):
    u = 'trade/new_order'
    headers = {'name': name}
    data = {'symbol': symbol,
            'side': side,
            'positionSide': positionSide,
            'type': 'MARKET',
            'quantity': quantity,
            }
    logger.info('u账户和交易-市价下单:local_time:{l},{d}'.format(l=cur_time(), d=data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info('订单回报:{},local_time:{}'.format(r.json(), cur_time()))


# u账户和交易-撤销订单
def um_trade_cancel_order(name, order_id):
    u = 'trade/cancel_order'
    headers = {'name': name}
    data = {'symbol': 'xrpusdt',
            'orderId': order_id
            # 'origClientOrderId': 'xrpusdt',   # 暂不启用
            }
    logger.info(data)
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info(r.json())


# uWebsocket账户信息推送-生成 Listen Key
def um_stream_new_listen_key(name):
    u = 'stream/new_listen_key'
    headers = {'name': name}
    logger.info('uWebsocket账户信息推送-生成 Listen Key')
    r = requests.post(url=futures_url + u, headers=headers)
    logger.info(r.json())
    if r.json().get('listenKey') is not None:
        return r.json().get('listenKey')
    else:
        logger.error('Websocket账户信息推送-生成 Listen Key 失败，请查看日志')
        return 'null'

# uWebsocket账户信息推送-延长 Listen Key 有效期
def um_stream_renew_listen_key(name, listen_key):
    u = 'stream/renew_listen_key'
    headers = {'name': name}
    data = {'listen_key': listen_key}
    logger.info('uWebsocket账户信息推送-延长 Listen Key 有效期:{}'.format(data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    if r.json() == {}:
        return True
    else:
        logger.info(r.json())

# uWebsocket账户信息推送-关闭 Listen Key
def um_stream_close_listen_key(name, listen_key):
    u = 'stream/close_listen_key'
    headers = {'name': name}
    data = {'listen_key': listen_key}
    logger.info('uWebsocket账户信息推送-关闭 Listen Key:{}'.format(data))
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info(r.json())

# u账户和交易-查看当前全部挂单
def test_um_trade_get_orders(name, symbol):
    u = 'trade/get_orders'
    headers = {'name': name}
    data = {'symbol': symbol}
    logger.info(data)
    r = requests.post(url=futures_url + u, headers=headers, data=json.dumps(data))
    logger.info(type(r), r.json())


def new_client_order_id():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    characters = random.sample(alphabet, 8)
    s = 'whd_'
    for i in characters:
        s += i
    return s

def get_symbol_daily_from_redis(redis_client, symbol):
    total_amount_key = f'daily_total_amount_{symbol.lower()}'
    redis_dict = redis_client.get(total_amount_key)
    if redis_dict is None:
        return None
    redis_dict = json.loads(redis_dict.decode('utf-8'))
    total_volume = float(redis_dict['total_volume'])
    total_amount = float(redis_dict['total_amount'])
    is_complete = bool(redis_dict['complete'])
    return total_volume, total_amount, is_complete

def mysql_method(sql):
    logger.info('sql:{a}'.format(a=sql))
    mysql = pymysql.connect(host='', user='root', password='', port=3306, charset='utf8mb4', database='')
    db = mysql.cursor()

    try:
        db.execute(sql)
        mysql.commit()
        logger.info('写入记录成功')
    except Exception as e:
        logger.error('写入mysql操作失败,回滚操作。e:{e},sql:{s}'.format(e=e, s=sql))
        mysql.rollback()

    db.close()  # 游标关闭
    mysql.close()  # 关闭连接
