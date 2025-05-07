from nautilus_trader.trading.strategy import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId
from typing import Sequence

class MACDStrategyConfig(StrategyConfig):
    instrument_ids: Sequence[InstrumentId] = []
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    trade_size: float = 0.01
    commission_rate: float = 0.0002
    max_position_ratio: float = 0.1
    starting_balance: float = 10000.0