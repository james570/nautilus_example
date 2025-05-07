# generate_instruments.py
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_instruments_file(output_dir: str = "crypto_data"):
    """
    生成包含OKX永续合约标准参数的_instruments.parquet文件
    
    参数:
        output_dir: 输出目录路径 (默认: "crypto_data")
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # OKX永续合约标准参数
    instruments_data = [
        {
            "symbol": "BTC-USDT",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "price_increment": 0.1,        # 价格最小变动单位
            "size_increment": 0.0001,      # 数量最小变动单位
            "price_precision": 1,          # 等于price_increment的小数位数
            "size_precision": 8,           # 等于size_increment的小数位数
            "margin_init": 0.1,            # 初始保证金率(10%)
            "margin_maint": 0.05,          # 维持保证金率(5%)
            "maker_fee": -0.0002,          # Maker费率(负表示返佣)
            "taker_fee": 0.0005,           # Taker费率
            "min_order_size": 0.001,       # 最小下单量
            "max_order_size": 1000,        # 最大下单量
            "last_updated": datetime.now().isoformat()
        },
        {
            "symbol": "ETH-USDT",
            "base_currency": "ETH",
            "quote_currency": "USDT",
            "price_increment": 0.01,
            "size_increment": 0.0001,
            "price_precision": 2,
            "size_precision": 8,
            "margin_init": 0.1,
            "margin_maint": 0.05,
            "maker_fee": -0.0002,
            "taker_fee": 0.0005,
            "min_order_size": 0.01,
            "max_order_size": 1000,
            "last_updated": datetime.now().isoformat()
        }
    ]

    # 创建DataFrame并保存
    df = pd.DataFrame(instruments_data)
    file_path = output_path / "_instruments.parquet"
    df.to_parquet(file_path)
    
    print(f"✅ 成功生成合约信息文件: {file_path}")
    print("包含的交易对:")
    print(df[['symbol', 'base_currency', 'quote_currency']].to_string(index=False))

if __name__ == "__main__":
    generate_instruments_file()