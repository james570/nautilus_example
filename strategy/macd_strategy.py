from decimal import Decimal
from typing import Dict
import pandas as pd
import numpy as np
from collections import deque

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.data import BarSpecification, BarType
from nautilus_trader.model.instruments import Instrument

from config.macd_strategy_config import MACDStrategyConfig


class MACDStrategy(Strategy):
    """
    不使用TA-Lib的MACD策略实现
    """
    
    def __init__(self, config: MACDStrategyConfig):
        super().__init__(config)
        
        # MACD参数
        self.fast_period = config.fast_period
        self.slow_period = config.slow_period
        self.signal_period = config.signal_period
        
        # 交易参数
        self.trade_size = Decimal(str(config.trade_size)) if config.trade_size else Decimal('0.01')
        self.commission_rate = config.commission_rate
        self.max_position_ratio = config.max_position_ratio
        
        # 数据存储
        self.price_data: Dict[InstrumentId, deque] = {}
        self.ema_fast: Dict[InstrumentId, float] = {}
        self.ema_slow: Dict[InstrumentId, float] = {}
        self.ema_signal: Dict[InstrumentId, float] = {}
        self.macd_line: Dict[InstrumentId, float] = {}
        self.positions: Dict[InstrumentId, bool] = {}
        
        # 3分钟Bar规格
        self.bar_spec = BarSpecification(3, BarAggregation.MINUTE, PriceType.LAST)

    def on_start(self):
        """策略启动时初始化"""
        self.log.info("策略启动中...")
        try:
            for instrument_id in self.config.instrument_ids:
                self.price_data[instrument_id] = deque(maxlen=self.slow_period + self.signal_period)
                self.ema_fast[instrument_id] = None
                self.ema_slow[instrument_id] = None
                self.ema_signal[instrument_id] = None
                self.macd_line[instrument_id] = None
                self.positions[instrument_id] = None
                
                bar_type = BarType(
                    instrument_id=instrument_id,
                    bar_spec=self.bar_spec
                )
                self.subscribe_bars(bar_type)
                self.log.info(f"订阅 {instrument_id} 的3分钟K线数据")
        except Exception as e:
            self.log.error(f"策略启动失败: {str(e)}")
            raise

    def _calculate_ema(self, current_value: float, previous_ema: float, period: int) -> float:
        """计算指数移动平均(EMA)"""
        try:
            if previous_ema is None:
                return current_value
            multiplier = 2 / (period + 1)
            return (current_value - previous_ema) * multiplier + previous_ema
        except Exception as e:
            self.log.error(f"计算EMA失败: {str(e)}")
            return current_value

    def on_bar(self, bar: Bar):
        """处理3分钟K线数据"""
        try:
            instrument_id = bar.bar_type.instrument_id
            close_price = bar.close.as_double()
            
            # 存储价格数据
            self.price_data[instrument_id].append(close_price)
            
            # 确保有足够数据计算MACD
            if len(self.price_data[instrument_id]) < self.slow_period:
                return
            
            # 计算EMA
            self.ema_fast[instrument_id] = self._calculate_ema(
                close_price, 
                self.ema_fast.get(instrument_id), 
                self.fast_period
            )
            self.ema_slow[instrument_id] = self._calculate_ema(
                close_price, 
                self.ema_slow.get(instrument_id), 
                self.slow_period
            )
            
            # 计算MACD线
            if None not in (self.ema_fast[instrument_id], self.ema_slow[instrument_id]):
                self.macd_line[instrument_id] = self.ema_fast[instrument_id] - self.ema_slow[instrument_id]
            
            # 计算信号线
            if (len(self.price_data[instrument_id]) >= self.slow_period + self.signal_period and 
                self.macd_line.get(instrument_id) is not None):
                self.ema_signal[instrument_id] = self._calculate_ema(
                    self.macd_line[instrument_id],
                    self.ema_signal.get(instrument_id),
                    self.signal_period
                )
            
            # 检查是否有有效的MACD和信号线
            if None in (self.macd_line.get(instrument_id), self.ema_signal.get(instrument_id)):
                return
            
            # 获取当前仓位状态
            current_position = self.positions.get(instrument_id)
            
            # 生成交易信号
            self._generate_trading_signal(instrument_id, current_position)
            
        except Exception as e:
            self.log.error(f"处理K线时出错: {str(e)}")

    def _generate_trading_signal(self, instrument_id: InstrumentId, current_position: bool):
        """生成交易信号"""
        try:
            # MACD线上穿信号线 - 买入信号
            if (self.macd_line[instrument_id] > self.ema_signal[instrument_id] and 
                (current_position is None or not current_position)):
                self._submit_order(instrument_id, OrderSide.BUY)
                self.positions[instrument_id] = True
                self.log.info(f"BUY信号 {instrument_id} - MACD: {self.macd_line[instrument_id]:.4f}, Signal: {self.ema_signal[instrument_id]:.4f}")
            
            # MACD线下穿信号线 - 卖出信号
            elif (self.macd_line[instrument_id] < self.ema_signal[instrument_id] and 
                  (current_position is None or current_position)):
                self._submit_order(instrument_id, OrderSide.SELL)
                self.positions[instrument_id] = False
                self.log.info(f"SELL信号 {instrument_id} - MACD: {self.macd_line[instrument_id]:.4f}, Signal: {self.ema_signal[instrument_id]:.4f}")
        except Exception as e:
            self.log.error(f"生成交易信号时出错: {str(e)}")

    def _validate_quantity(self, quantity: Decimal, instrument: Instrument) -> Quantity:
        """验证并调整订单数量精度"""
        try:
            if quantity is None:
                raise ValueError("数量不能为None")
            
            # 获取合约定义的精度
            size_increment = instrument.size_increment
            if size_increment is None:
                raise ValueError(f"合约 {instrument.id} 未设置size_increment")
            
            size_precision = abs(int(Decimal(str(size_increment)).as_tuple().exponent))
            validated_qty = round(quantity, size_precision)
            
            # 确保不小于最小订单量
            if hasattr(instrument, 'min_quantity') and instrument.min_quantity is not None:
                validated_qty = max(validated_qty, instrument.min_quantity)
            
            self.log.debug(f"数量调整: {quantity} -> {validated_qty} ({instrument.id})")
            return Quantity(validated_qty, precision=size_precision)
            
        except Exception as e:
            self.log.error(f"验证数量时出错: {str(e)}")
            return Quantity.zero()

    def _calculate_position_size(self, instrument_id: InstrumentId, price: Price) -> Decimal:
        """计算仓位大小"""
        try:
            account = self.cache.account_for_venue(instrument_id.venue)
            if not account:
                raise ValueError(f"未找到账户 {instrument_id.venue}")
            
            free_balance = account.balance().free.as_decimal()
            if free_balance <= 0:
                raise ValueError(f"账户可用资金不足: {free_balance}")
            
            max_position_value = free_balance * Decimal(str(self.max_position_ratio))
            size = max_position_value / price.as_decimal()
            
            # 使用固定交易量或计算值
            return self.trade_size if self.trade_size > 0 else size
            
        except Exception as e:
            self.log.error(f"计算仓位大小时出错: {str(e)}")
            return Decimal(0)

    def _submit_order(self, instrument_id: InstrumentId, side: OrderSide):
        """提交订单"""
        try:
            instrument = self.cache.instrument(instrument_id)
            if instrument is None:
                raise ValueError(f"无法找到合约: {instrument_id}")
            
            # 修正：使用正确的BarType获取最新K线
            bar_type = BarType(
                instrument_id=instrument_id,
                bar_spec=self.bar_spec
            )
            last_bar = self.cache.bar(bar_type)  # 传入BarType而不是InstrumentId
            
            if last_bar is None:
                raise ValueError("没有可用的最新K线数据")
            
            quantity = self._calculate_position_size(instrument_id, last_bar.close)
            if quantity <= 0:
                raise ValueError(f"无效的交易量: {quantity}")
            
            validated_quantity = self._validate_quantity(quantity, instrument)
            if validated_quantity == 0:
                raise ValueError("验证后的交易量为0")
            
            # 先平掉现有仓位
            positions = self.cache.positions(instrument_id=instrument_id)
            if positions:
                self.close_position(positions[0])
            
            # 开新仓位
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=side,
                quantity=validated_quantity,
            )
            self.submit_order(order)
            
        except Exception as e:
            self.log.error(f"提交订单时出错: {str(e)}")

    def on_stop(self):
        """策略停止时清理所有仓位"""
        try:
            open_positions = list(self.cache.positions_open())
            if not open_positions:
                self.log.info("策略停止：无持仓需要清理")
                return

            self.log.info(f"开始清理 {len(open_positions)} 个持仓...")
            for position in open_positions:
                self.close_position(position)
                self.positions[position.instrument_id] = None
                self.log.info(f"已发送平仓指令: {position}")
                
        except Exception as e:
            self.log.error(f"策略停止时出错: {str(e)}")
            raise