import os
import pandas as pd
from pathlib import Path
from decimal import Decimal
from typing import Dict, List

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model import TraderId, Venue, Money
from nautilus_trader.model.objects import Currency, Price, Quantity
from nautilus_trader.model.enums import AccountType, OmsType, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.config import LoggingConfig

from strategy.macd_strategy import MACDStrategy
from config.macd_strategy_config import MACDStrategyConfig

def load_and_convert_data(file_path: Path, instrument: CryptoPerpetual, bar_spec: BarSpecification) -> List[Bar]:
    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            print(f"⚠️ 空数据文件: {file_path}")
            return []

        # 确保包含所有必要列
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"❌ 缺少必要列 {missing} 在文件 {file_path}")
            return []

        df = df[required_cols]
        
        # 转换数据类型
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df['volume'] = df['volume'].fillna(0).astype(float)
        
        # 统一时间戳处理
        if isinstance(df['timestamp'].iloc[0], (int, float)):
            # 如果时间戳是数字，转换为纳秒
            if df['timestamp'].max() < 1e18:  # 假设是毫秒时间戳
                df['timestamp'] = pd.to_datetime(df['timestamp'] * 1_000_000)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif isinstance(df['timestamp'].iloc[0], str):
            # 如果是字符串格式的时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 确保时间戳是datetime64[ns]类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        bar_type = BarType(
            instrument_id=instrument.id,
            bar_spec=bar_spec
        )

        wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
        bars = list(wrangler.process(df.set_index('timestamp')))
        
        print(f"✅ 成功加载 {len(bars)} 条K线数据: {file_path.name}")
        return bars

    except Exception as e:
        print(f"❌ 加载失败 {file_path}: {type(e).__name__} - {str(e)}")
        return []

def initialize_instruments(df: pd.DataFrame, venue: Venue) -> Dict[str, CryptoPerpetual]:
    instruments = {}
    
    # 设置默认值
    defaults = {
        'price_increment': 0.1,
        'size_increment': 0.0001,  # 默认精度调整为0.0001 (4位小数)
        'margin_init': 0.1,
        'margin_maint': 0.05,
        'maker_fee': -0.0002,
        'taker_fee': 0.0005,
        'min_order_size': 0.001,
        'max_order_size': 1000
    }
    
    # 检查必要列
    required_columns = ['symbol']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要列: {missing_columns}")
    
    for _, row in df.iterrows():
        try:
            symbol = row["symbol"]
            base = symbol.split('-')[0]
            quote = symbol.split('-')[1] if '-' in symbol else "USDT"
            
            # 自动计算精度
            def get_precision(x):
                s = str(float(x))
                if '.' in s:
                    return len(s.split('.')[1].rstrip('0'))
                return 0
            
            price_increment = float(row.get("price_increment", defaults['price_increment']))
            size_increment = float(row.get("size_increment", defaults['size_increment']))
            
            price_precision = get_precision(price_increment)
            size_precision = get_precision(size_increment)
            
            # 确保size_increment至少有4位小数
            size_increment = max(size_increment, 0.0001)
            
            inst = CryptoPerpetual(
                instrument_id=InstrumentId(
                    symbol=Symbol(symbol.replace('-', '/')),
                    venue=venue
                ),
                raw_symbol=Symbol(symbol),
                base_currency=Currency.from_str(base),
                quote_currency=Currency.from_str(quote),
                settlement_currency=Currency.from_str(quote),
                is_inverse=False,
                price_precision=price_precision,
                size_precision=size_precision,
                price_increment=Price.from_str(str(price_increment)),
                size_increment=Quantity.from_str(str(size_increment)),
                margin_init=Decimal(str(row.get("margin_init", defaults['margin_init']))),
                margin_maint=Decimal(str(row.get("margin_maint", defaults['margin_maint']))),
                maker_fee=Decimal(str(row.get("maker_fee", defaults['maker_fee']))),
                taker_fee=Decimal(str(row.get("taker_fee", defaults['taker_fee']))),
                ts_event=pd.Timestamp.now().value,
                ts_init=pd.Timestamp.now().value,
            )
            instruments[symbol] = inst
            print(f"✅ 成功初始化 {symbol} (价格精度: {price_precision}, 数量精度: {size_precision})")
            
        except Exception as e:
            print(f"❌ 初始化失败 {symbol}: {str(e)}")
            raise
    
    return instruments

def main():
    base_dir = Path(__file__).parent
    catalog_path = base_dir / "crypto_data"
    output_dir = base_dir / "backtest_results"
    
    # 确保目录存在
    catalog_path.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # 加载合约信息
        instruments_file = catalog_path / "_instruments.parquet"
        if not instruments_file.exists():
            # 如果文件不存在，创建默认文件
            create_default_instruments_file(instruments_file)
            
        instruments_df = pd.read_parquet(instruments_file)
        print(f"✅ 加载 {len(instruments_df)} 个合约信息")
        print("可用列:", instruments_df.columns.tolist())  # 打印所有列名用于调试
        
        # 初始化回测引擎
        bar_spec = BarSpecification(3, BarAggregation.MINUTE, PriceType.LAST)
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id=TraderId("OKX-MACD"),
                logging=LoggingConfig(log_level="INFO")
            )
        )

        # 添加模拟交易所
        sim_venue = Venue("OKX")
        engine.add_venue(
            venue=sim_venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=Currency.from_str("USDT"),
            starting_balances=[Money(10000.0, Currency.from_str("USDT"))],
        )

        # 初始化所有交易对
        instrument_map = initialize_instruments(instruments_df, sim_venue)
        for inst in instrument_map.values():
            engine.add_instrument(inst)
            
        # 加载K线数据
        total_bars = 0
        for file in catalog_path.glob("*.parquet"):
            if file.name.startswith("_"):
                continue

            symbol = file.stem
            if symbol not in instrument_map:
                print(f"⚠️ 跳过无匹配标的: {file.name}")
                continue

            bars = load_and_convert_data(file, instrument_map[symbol], bar_spec)
            if bars:
                engine.add_data(bars)
                total_bars += len(bars)
                print(f"✅ 加载 {len(bars)} 条3分钟K线: {symbol}")

        if total_bars == 0:
            raise ValueError("没有加载到任何K线数据")

        # 添加策略
        strategy_cfg = MACDStrategyConfig(
            instrument_ids=[inst.id for inst in instrument_map.values()],
            fast_period=12,
            slow_period=26,
            signal_period=9,
            trade_size=0.01,  # 确保trade_size符合精度要求
            commission_rate=0.0002,
            max_position_ratio=0.1
        )
        engine.add_strategy(MACDStrategy(config=strategy_cfg))

        # 运行回测
        print("🚀 开始回测...")
        engine.run()
        print("✅ 回测完成!")

        # 保存结果
        engine.trader.generate_order_fills_report().to_csv(output_dir / "order_fills.csv")
        engine.trader.generate_positions_report().to_csv(output_dir / "positions.csv")
        engine.trader.generate_account_report(sim_venue).to_csv(output_dir / "account.csv")
        print(f"📊 回测结果已保存到 {output_dir}")

    except Exception as e:
        print(f"❌ 回测失败: {e}")
        raise

def create_default_instruments_file(file_path: Path):
    """创建默认的合约信息文件"""
    data = {
        "symbol": ["BTC-USDT", "ETH-USDT"],
        "base_currency": ["BTC", "ETH"],
        "quote_currency": ["USDT", "USDT"],
        "price_increment": [0.1, 0.01],
        "size_increment": [0.0001, 0.0001],  # 确保默认精度为4位小数
        "margin_init": [0.1, 0.1],
        "margin_maint": [0.05, 0.05],
        "maker_fee": [-0.0002, -0.0002],
        "taker_fee": [0.0005, 0.0005],
        "min_order_size": [0.001, 0.01],
        "max_order_size": [1000, 1000],
        "last_updated": [pd.Timestamp.now(), pd.Timestamp.now()]
    }
    df = pd.DataFrame(data)
    df.to_parquet(file_path)
    print(f"✅ 已创建默认合约信息文件: {file_path}")

if __name__ == "__main__":
    main()