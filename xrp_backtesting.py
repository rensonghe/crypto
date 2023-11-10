# -*- coding: utf-8 -*-
# zq
from datetime import datetime, timedelta
import os
import sys
import pandas as pd

# import ai_ctastrategy
BASE_DIR = os.path.abspath('../..')
sys.path.append(BASE_DIR)
from tzquant.trader.optimize import OptimizationSetting
from tz_ctastrategy.backtesting import BacktestingEngine
# from ai_ctastrategy import ai_TickStrategy
from tz_ctastrategy.base import (
    BacktestingMode,
    DataType
)

from tz_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData
)
from tzquant.trader.utility import (
    BarGenerator,
    ArrayManager,
    Interval
)
import time
# from time import time
# from HFT import factor_calculation, cols_list
import numpy as np


class ai_TickStrategy(CtaTemplate):
    """"""
    author = "zq"

    split_count = 5
    place_rate = 3 / 10000

    init_size = 60
    pos_rate = 0.3  # 持仓比例
    record = False  # 是否记录成交记录

    test_trigger = 10

    tick_count = 0
    test_all_done = False
    record = False

    parameters = ["test_trigger"]
    variables = ["tick_count", "test_all_done"]

    trades = []
    strategy_trades = []
    values = []
    fill_order_time = 0

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting, rolling_info=None):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting, rolling_info)
        self.last_tick = None
        self.bg = BarGenerator(self.on_bar)
        self.kill_time = 0
        # self.bg1h = BarGenerator(self.on_bar, 60, self.on_1h_bar)
        # self.bg1d = BarGenerator(self.on_bar, 24, self.on_1d_bar, interval=Interval.HOUR)
        # self.time = time()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """

        self.put_event()

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # 1m kline data to 1d kline data
        # self.bg1d.update_bar(bar)
        # 策略逻辑
        place_value = self.cta_engine.capital * self.pos_rate / self.split_count
        size = round(place_value / bar.close, 8)
        price = bar.feat_dict['vwap_2s']
        position_value = self.pos * price
        max_limited_order_value = self.cta_engine.capital * self.pos_rate

        # if self.cta_engine.ca_balance_now['price'] * abs(self.pos) > self.cta_engine.capital * self.pos_rate:
        #     value = self.cta_engine.ca_balance_now['price'] * abs(self.pos)-self.cta_engine.capital * self.pos_rate
        #     self.strategy_trades.append(value)
        #     if len(self.strategy_trades)>1:
        #         if self.strategy_trades[-1] != self.strategy_trades[-2]:
        #             self.values.append([self.strategy_trades[-1], bar.datetime])
        #             # print('超出金额:',value)
        #             v = pd.DataFrame(self.values,columns=['value','datetime'])
        #             v.to_csv('value.csv')

        # 每60s判断一次
        if int(bar.closetime / 1000) - int(self.kill_time / 1000) > 60:
            # 多仓止盈止损
            if self.pos > 0:
                # print('-------------平仓之前撤销所有订单-------------')
                # self.cancel_all(closetime=bar.closetime / 1000)
                pf = float(bar.close / self.cta_engine.ca_balance_now['price']) - 1
                con1 = 0
                if pf > 0.005:
                    print('-------------多头止盈离场-------------')
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000))
                    self.cancel_all(closetime=bar.closetime / 1000)
                    self.kill_time = bar.closetime / 1000
                    self.sell(price=bar.close, volume=self.pos,  # stop=True,
                              net=True, closetime=bar.closetime / 1000)
                elif pf <= -0.003:
                    print('-------------多头止损离场-------------')
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000))
                    self.cancel_all(closetime=bar.closetime / 1000)
                    self.kill_time = bar.closetime / 1000
                    self.sell(price=bar.close, volume=self.pos,  # stop=True,
                              net=True, closetime=bar.closetime / 1000)

            # 空仓止盈止损
            if self.pos < 0:
                # print('-------------平仓之前撤销所有订单-------------')
                # self.cancel_all(closetime=bar.closetime / 1000)
                pf = 1 - float(bar.close / self.cta_engine.ca_balance_now['price'])
                if pf > 0.005:
                    print('-------------空头止盈离场-------------')
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000))
                    self.cancel_all(closetime=bar.closetime / 1000)
                    self.kill_time = bar.closetime / 1000
                    self.buy(price=bar.close, volume=-self.pos,  # stop=True,
                             net=True, closetime=bar.closetime / 1000)
                elif pf <= -0.003:
                    print('-------------空头止损离场-------------')
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000))
                    self.cancel_all(closetime=bar.closetime / 1000)
                    self.kill_time = bar.closetime / 1000
                    self.buy(price=bar.close, volume=-self.pos,  # stop=True,
                             net=True, closetime=bar.closetime / 1000)

        limit_orders_values = 0  # 挂单价值
        final_values = 0  # 最终总价值 = 持仓价值 + 挂单价值
        for key in self.cta_engine.active_limit_orders.keys():
            # 总挂单价值
            limit_orders_values += float(self.cta_engine.active_limit_orders[key].price) * abs(
                float(self.cta_engine.active_limit_orders[key].volume))
            # if bar.closetime > 1673170199999 and bar.closetime<1673170199999 + 1000*60*60:
            # print('当前挂单:',self.cta_engine.active_limit_orders)
            # self.values.append([limit_orders_values,bar.datetime])
            # v = pd.DataFrame(self.values, columns=['limited_orders_values','time'])
            # v.to_csv('value.csv')
        final_values = limit_orders_values + self.cta_engine.ca_balance_now['price'] * abs(self.pos)

        # 平多仓
        if bar.feat_dict['side'] == 'sell' and self.pos > 0:
            # print('-------------平仓之前撤销所有订单-------------')
            self.cancel_all(closetime=bar.closetime / 1000)
            print('-----------------下空单平多仓-------------',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000)))
            self.sell(price=bar.close * (1 + self.place_rate), volume=self.pos,  # stop=True,
                      net=True, closetime=bar.closetime / 1000)
        # 平空仓
        if bar.feat_dict['side'] == 'buy' and self.pos < 0:
            # print('-------------平仓之前撤销所有订单-------------')
            self.cancel_all(closetime=bar.closetime / 1000)
            print('---------------下多单平空仓---------------',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000)))
            self.buy(price=bar.close * (1 - self.place_rate), volume=-self.pos,  # stop=True,
                     net=True, closetime=bar.closetime / 1000)

        # 开空仓
        if bar.feat_dict['side'] == 'sell' and position_value >= -self.pos_rate * self.cta_engine.capital * (
                1 - 1 / self.split_count):
            # 如果此时有挂多单，全部撤掉
            if len(self.cta_engine.active_limit_orders) > 0:
                last_order = list(self.cta_engine.active_limit_orders.keys())[-1]
                if self.cta_engine.active_limit_orders[last_order].volume > 0:
                    self.cancel_all(closetime=bar.closetime / 1000)

            if max_limited_order_value <= final_values * 1.00001:
                self.cancel_all(closetime=bar.closetime / 1000)
            print('---------------下空单---------------',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000)))
            self.sell(price=price * (1 - self.place_rate), volume=size,  # stop=True,
                      net=True, closetime=bar.closetime / 1000)
            # self.fill_order_time = bar.closetime
        # 开多仓
        if bar.feat_dict['side'] == 'buy' and position_value <= self.pos_rate * self.cta_engine.capital * (
                1 - 1 / self.split_count):
            # 如果此时有挂空单，全部撤掉
            if len(self.cta_engine.active_limit_orders) > 0:
                last_order = list(self.cta_engine.active_limit_orders.keys())[-1]
                if self.cta_engine.active_limit_orders[last_order].volume < 0:
                    self.cancel_all(closetime=bar.closetime / 1000)

            if max_limited_order_value <= final_values * 1.00001:
                self.cancel_all(closetime=bar.closetime / 1000)
            print('---------------下多单--------------',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bar.closetime / 1000)))
            self.buy(price=price * (1 + self.place_rate), volume=size,  # stop=True,
                     net=True, closetime=bar.closetime / 1000)
            # self.fill_order_time = bar.closetime

        # 保持仓位
        else:
            return

        pass

    def on_1h_bar(self, bar: BarData):
        self.bg1d.update_bar(bar)
        # print(bar)

    def on_1d_bar(self, bar: BarData):
        # print('1d', bar)
        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        print(order)
        self.put_event()

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        # 交易所成交记录
        self.cta_engine.output(
            'trades:{}'.format({"price": trade.price, "size": trade.volume, "direction": trade.direction,
                                "o_id": trade.orderid, "t_id": trade.tradeid, "datetime": trade.datetime,
                                "pos": self.pos, 'closetime': trade.datetime.timestamp() * 1000 - 1,
                                '持仓均价': self.cta_engine.ca_balance_now['price'],
                                'pnl': self.cta_engine.total_balance + self.cta_engine.total_unrealized_pnl}))

        # if self.record:
        #     self.trades.append({"price": trade.price, "size": trade.volume, "direction": trade.direction,
        #                                 "o_id": trade.orderid, "t_id": trade.tradeid, "datetime": trade.datetime,
        #                                 "pos": self.pos, 'closetime': trade.datetime.timestamp() * 1000 - 1})
        # if trade:
        #     self.strategy_trades.append({'price':trade.price,'size':trade.volume})

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        self.put_event()

    def test_market_order(self):
        """"""
        self.buy(self.last_tick.limit_up, 1)
        self.write_log("执行市价单测试")

    def test_limit_order(self):
        """"""
        self.buy(self.last_tick.limit_down, 1)
        self.write_log("执行限价单测试")

    def test_stop_order(self):
        """"""
        self.buy(self.last_tick.ask_price_1, 1, True)
        self.write_log("执行停止单测试")

    def test_cancel_all(self):
        """"""
        self.cancel_all()
        self.write_log("执行全部撤单测试")


if __name__ == '__main__':
    engine = BacktestingEngine()
    # 设置时间从今天的零点开始
    now_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    engine.set_parameters(
        vt_symbol=f"{'xrpusdt'}.{'binance_swap_u'}",
        # start=now_datetime - timedelta(days=13),  # 形如：datetime(2022,1,1)
        # end=now_datetime - timedelta(days=12),
        start=datetime(2023, 6, 1),  # 形如：datetime(2022,1,1)
        end=datetime(2023, 6, 30),
        maker_fee=0 / 10000,  # 挂单手续费
        taker_fee=3 / 10000,  # 吃单手续费
        slippage=3 / 10000,  # 滑点
        size=1,  # 杠杆倍数 默认为1
        pricetick=0.00000001,  # 价格精度
        capital=200,  # 本金
        annual_days=365,  # 一年的连续交易天数
        # label=DataType.DCT,  # tick级别的市场选择
        # mode=BacktestingMode.TICK,  # tick级别回测
        feat_path='datafile/eval/songhe/cta_binance_20bar_vwap_20230601_0608_20230609_xrpusdt_70.cur'
    )
    engine.add_strategy(ai_TickStrategy, {})

    engine.load_data()
    engine.run_backtesting()

    # ----------- 回测并画图 --------------
    df = engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()
