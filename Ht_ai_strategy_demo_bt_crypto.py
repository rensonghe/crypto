# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:15:37 2023

@author: linyiSky
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:30:53 2023

@author: linyiSky
"""

import os
import csv 


from datetime import datetime,timedelta
from wtpy.WtDataDefs import WtBarRecords, WtTickRecords, WtOrdDtlRecords, WtOrdQueRecords, WtTransRecords
from wtpy import CtaContext,HftContext
#from LogUtils import LogUtils

from HtDataDefs import TickData,BarData,OrderData,TradeData

from CtaTemplate import CtaTemplate
# import datetime
from datetime import datetime, timedelta, timezone
import pandas as pd
import time

#日志类使用
#logger = LogUtils().get_log()
#%%

def makeTime(date:int, time:int, secs:int):
    '''
    将系统时间转成datetime\n
    @date   日期，格式如20200723\n
    @time   时间，精确到分，格式如0935\n
    @secs   秒数，精确到毫秒，格式如37500
    '''
    return datetime(year=int(date/10000), month=int(date%10000/100), day=date%100, 
        hour=int(time/100), minute=time%100, second=int(secs/1000), microsecond=secs%1000*1000)


class Ht_ai_strategy_demo_bt(CtaTemplate):



    split_count = 5
    place_rate = 4 / 10000

    init_size = 60
    pos_rate = 0.3  # 持仓比例
    # futuresize = 15
    capital = 1000
    # deposit_rate = 0.1


    def __init__(self, name:str, code:str, expsecs:int, offset:int,signal_file:str="", freq:int=30):
        super().__init__(name, code, expsecs, offset,signal_file,freq)

        self.kill_time = 0
        self.last_time=int(time.time())


    def on_init_new(self):

        return 
    
    def on_tick_new(self, newTick:TickData):


        # ordict = self.get_active_limit_orders()
        # msg_ = f'-----------当前挂单 :{list(ordict.keys())}'
        # self.log_text(msg_)
        current_time = pd.to_datetime(newTick.closetime, format='%Y%m%d%H%M%S%f')
        # print('current_time',current_time)
        # print('closetime', newTick.closetime)
        beijing_tz = timezone(timedelta(hours=8))
        closetime_ = int(current_time.replace(tzinfo=beijing_tz).timestamp() * 1000)
        # closetime_ = int(current_time.timestamp() * 1000)
        # if int(closetime_ / 1000) - int(self.kill_time / 1000) > 60 * 5:
        #     # 多仓止盈止损
        #     if self.get_position(self.__code__) > 0:
        #         # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
        #         # self.cancel_all(closetime=bar.closetime / 1000)
        #         pf = float(newTick.price / abs(self.get_position_avgpx(self.__code__))) - 1
        #         # print('close:', newTick.price)
        #         # print('pos_avg_price:', self.get_position_avgpx(self.__code__))
        #         con1 = 0
        #         if pf > 0.05:
        #             con1 = 1
        #             self.cancel_all(self.__code__)
        #             self.kill_time = closetime_ / 1000
        #             self.sell(price=newTick.bid_price_1, volume=self.get_position(self.__code__),
        #                       closetime=closetime_)
        #             msg_ = f'多头止盈离场---平仓价格:{newTick.bid_price_1}---size:{self.get_position(self.__code__)}---time:{current_time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         elif pf <= -0.005:
        #             con1 = 1
        #             self.cancel_all(self.__code__)
        #             self.kill_time = closetime_ / 1000
        #             self.sell(price=newTick.bid_price_1, volume=self.get_position(self.__code__),
        #                       closetime=closetime_)
        #             msg_ = f'多头止损离场---平仓价格:{newTick.bid_price_1}---size:{self.get_position(self.__code__)}---time:{current_time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         # if con1 == 1:
        #         #     # print('-------------离场时间-----------------',
        #         #     # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
        #         #     self.cancel_all(self.__code__)
        #         #     self.kill_time = newBar.closetime / 1000
        #         #     self.sell(price=newBar.close * (1 - self.place_rate), volume=self.get_position(self.__code__),
        #         #               closetime=newBar.time / 1000)
        #
        #     # 空仓止盈止损
        #     if self.get_position(self.__code__) < 0:
        #         # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
        #         # self.cancel_all(closetime=bar.closetime / 1000)
        #         pf = 1 - float(newTick.price / abs(self.get_position_avgpx(self.__code__)))
        #         # print('close:', newTick.price)
        #         # print('pos_avg_price:', self.get_position_avgpx(self.__code__))
        #         con1 = 0
        #         if pf > 0.05:
        #             con1 = 1
        #             self.cancel_all(self.__code__)
        #             self.kill_time = closetime_ / 1000
        #             self.buy(price=newTick.ask_price_1, volume=-self.get_position(self.__code__),
        #                      closetime=closetime_)
        #             msg_ = f'空头止盈离场---平仓价格:{newTick.ask_price_1}---size:{self.get_position(self.__code__)}---time:{current_time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         elif pf <= -0.005:
        #             con1 = 1
        #             # print('-------------空头止损离场-------------', '品种:',self.model_symbol)
        #
        #             self.cancel_all(self.__code__)
        #             self.kill_time = closetime_ / 1000
        #             self.buy(price=newTick.ask_price_1, volume=-self.get_position(self.__code__),
        #                      closetime=closetime_)
        #             msg_ = f'空头止损离场---平仓价格:{newTick.ask_price_1}---size:{self.get_position(self.__code__)}---time:{current_time}---symbol:{self.__code__}'
        #             self.log_text(msg_)


        match_signal = self.signal_data.get(newTick.closetime)
        if match_signal is not None:
            date_time = match_signal[0]
            closetime = match_signal[1]
            vwapv = float(match_signal[2])
            close = float(match_signal[3])
            predict = match_signal[3]
            target = match_signal[4]
            side = match_signal[6]
            # signal_flag = match_signal[7]
            # print(datetime,side)
            # self.log_text("match_signal:{}  datetime:{}".format(match_signal, date_time))
            # print('signal_file flag:{}, tick flag : {}'.format(signal_flag, newTick.exchg))

            # =========策略内容=========

            price = vwapv
            # print('vwapv:',vwapv)
            # print(price)
            # price = factor['vwapv_30s'].iloc[-1]  # 挂单价格
            # 读取当前持仓
            curPos = self.get_position(self.__code__)
            position_value = curPos * price  # 持仓金额
            place_value = self.capital * self.pos_rate / self.split_count  # 挂单金额
            buy_size = round(place_value / (newTick.ask_price_1 ), 8)  # 买单量
            sell_size = round(place_value / (newTick.bid_price_1), 8)  # 卖单量
            max_limited_order_value = self.capital * self.pos_rate  # 最大挂单金额

            # 计算挂单金额
            limit_orders_values = 0
            final_values = 0
            ordict = self.get_active_limit_orders()
            for key in ordict.keys():
                limit_orders_values += abs(float(ordict[key].price)) * abs(float(ordict[key].leftQty))
            # 持仓金额+挂单金额
            final_values = limit_orders_values + (abs(self.get_position_avgpx(self.__code__)) * abs(curPos))

            # time_1 = datetime.strptime('09:00:00', '%H:%M:%S').time()
            # time_2 = datetime.strptime('09:05:00', '%H:%M:%S').time()
            # time_3 = datetime.strptime('15:30:00', '%H:%M:%S').time()
            # time_4 = datetime.strptime('21:05:00', '%H:%M:%S').time()
            # time_5 = datetime.strptime('23:59:59', '%H:%M:%S').time()
            # print(datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f").time(),'-------------')
            # if (datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f").time() >= time_2 and datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f").time() <= time_3) or \
            #         (datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f").time() >= time_4 and datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f").time()<= time_5):
            # 平多仓
            if side == 'sell' and curPos > 0:
                self.cancel_all(self.__code__)
                msg_ = f'下空单平多仓撤单'
                self.log_text(msg_)
                # print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(newTick.closetime)), '品种:',self.__code__)
                self.sell(price=price * (1 -self.place_rate), volume=curPos, closetime=newTick.closetime)
                msg_ = f'下空单平多仓---平仓价格:{price * (1 -self.place_rate)}---size:{curPos}---time:{current_time}---symbol:{self.__code__}'
                self.log_text(msg_)

            # 平空仓
            if side == 'buy' and curPos < 0:
                self.cancel_all(self.__code__)
                msg_ = f'下多单平多空撤单'
                self.log_text(msg_)
                # print('---------------------------下空单平多仓-----------------------',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(newTick.closetime)), '品种:',self.__code__)
                self.buy(price=price * (1 +self.place_rate), volume=-curPos, closetime=newTick.closetime)
                msg_ = f'下多单平空仓---平仓价格:{price * (1 + self.place_rate)}---size:{curPos}---time:{current_time}---symbol:{self.__code__}'
                self.log_text(msg_)

            # 开空仓
            if side == 'sell' and position_value >= -self.pos_rate * self.capital * (
                    1 - 1 / self.split_count):

                # print('--------------开空仓----------------', '品种:',self.__code__)
                if max_limited_order_value <= final_values * 1.0001 and curPos == 0:
                    self.cancel_all(self.__code__)
                    msg_ = f'空单超出挂单金额--撤单'
                    self.log_text(msg_)
                self.sell(price=price * (1 -self.place_rate), volume=sell_size, closetime=newTick.closetime )
                msg_ = f'开空仓---开仓价格:{price * (1 -self.place_rate)}---size:{-sell_size}---time:{current_time}---symbol:{self.__code__}'
                self.log_text(msg_)


            # 开多仓
            if  side == 'buy' and position_value <= self.pos_rate * self.capital * (
                    1 - 1 / self.split_count):

                # print('--------------开多仓----------------', '品种:',self.__code__)
                if max_limited_order_value <= final_values * 1.0001 and curPos == 0 :
                    self.cancel_all(self.__code__)
                    msg_ = f'多单超出挂单金额--撤单'
                    self.log_text(msg_)
                self.buy(price=price * (1 +self.place_rate), volume=buy_size, closetime=newTick.closetime )
                msg_ = f'开多仓---开仓价格:{price * (1 +self.place_rate)}---size:{buy_size}---time:{current_time}---symbol:{self.__code__}'
                self.log_text(msg_)

            else:

                pass




        '''
        
        for key,value in ordict.items():
            print("----------------------------",self.last_time,key,value.isBuy,value.isCanceled,value.code,value.leftQty)
            pass
        
        print("on_tick_new--------------",newTick.price)
        '''
        
        #print("on_tick_new,last_time,curtime--------------",newTick.price,self.last_time,newTick.closetime)
        
        #除了__init__外能使用self.log_text
        # self.log_text(f"on_tick_new:{newTick.price},last_time:{self.last_time},curtime:{newTick.closetime},type:{type(newTick.closetime)}")

        return
    
    def on_bar_new(self, newBar:BarData):


        # if int(newBar.time / 1000) - int(self.kill_time / 1000) > 60 *5:
        #     # 多仓止盈止损
        #     if self.get_position(self.__code__) > 0:
        #         # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
        #         # self.cancel_all(closetime=bar.closetime / 1000)
        #         pf = float(newBar.close / self.get_position_avgpx(self.__code__)) - 1
        #         con1 = 0
        #         if pf > 0.05:
        #             con1 = 1
        #             # print('-------------多头止盈离场-------------', '品种:',self.model_symbol)
        #             msg_ = f'多头止盈离场---平仓价格:{newBar.close * (1 + self.place_rate)}---size:{self.get_position(self.__code__)}---time:{newBar.time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         elif pf <= -0.009:
        #             con1 = 1
        #             # print('-------------多头止损离场-------------', '品种:',self.model_symbol)
        #             msg_ = f'多头止损离场---平仓价格:{newBar.close * (1 + self.place_rate)}---size:{self.get_position(self.__code__)}---time:{newBar.time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         if con1 == 1:
        #             # print('-------------离场时间-----------------',
        #             # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
        #             self.cancel_all(self.__code__)
        #             self.kill_time = newBar.closetime / 1000
        #             self.sell(price=newBar.close * (1 - self.place_rate), volume=self.get_position(self.__code__),
        #                       closetime=newBar.time / 1000)
        #
        #     # 空仓止盈止损
        #     if self.get_position(self.__code__) < 0:
        #         # print('-------------平仓之前撤销所有订单-------------', '品种:',self.model_symbol)
        #         # self.cancel_all(closetime=bar.closetime / 1000)
        #         pf = 1 - float(newBar.close / self.get_position_avgpx(self.__code__))
        #         con1 = 0
        #         if pf > 0.05:
        #             con1 = 1
        #             # print('-------------空头止盈离场-------------', '品种:',self.model_symbol)
        #             msg_ = f'空头止盈离场---平仓价格:{newBar.close * (1 - self.place_rate)}---size:{self.get_position(self.__code__)}---time:{newBar.time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         elif pf <= -0.009:
        #             con1 = 1
        #             # print('-------------空头止损离场-------------', '品种:',self.model_symbol)
        #             msg_ = f'空头止损离场---平仓价格:{newBar.close * (1 - self.place_rate)}---size:{self.get_position(self.__code__)}---time:{newBar.time}---symbol:{self.__code__}'
        #             self.log_text(msg_)
        #         if con1 == 1:
        #             # print('-------------离场时间-----------------',
        #             # time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(tick.closetime/1000)), '品种:',self.model_symbol)
        #             self.cancel_all(self.__code__)
        #             self.kill_time = newBar.closetime / 1000
        #             self.buy(price=newBar.close * (1 + self.place_rate), volume=-self.get_position(self.__code__),
        #                      closetime=newBar.time / 1000)
        # else:
        #     return
        #
        #     pass
        #
        #
        return
    
    def on_order_new(self, order: OrderData):

        self.log_text(f"[order info]  localid:{order.localid},code:{order.code},isBuy:{order.isBuy},totalQty:{order.totalQty},price:{order.price},isCanceled:{order.isCanceled}" )
        return
    
    def on_trade_new(self, trade: TradeData):

        self.log_text(
            f"[trade info]  localid:{trade.localid},code:{trade.code},isBuy:{trade.isBuy},volume:{trade.volume},price:{trade.price}")

        return
    
    
    def on_backtest_end(self, context:CtaContext):
        '''
        回测结束时回调，只在回测框架下会触发

        @context    策略上下文
        '''
        return
    
    
    
    
        
       
        
        
        
        
        
        
        
        
        
        