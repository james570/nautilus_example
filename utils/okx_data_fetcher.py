import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from decimal import Decimal
import humanize
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('okx_data_fetcher.log')
    ]
)
logger = logging.getLogger(__name__)

class OKXDataFetcher:
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None):
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'},
            'verbose': False  # 設為True可查看詳細HTTP請求
        })
        try:
            self.exchange.load_markets()
            logger.info("OKX市場數據加載成功，共%d個交易對", len(self.exchange.symbols))
        except Exception as e:
            logger.error("加載市場數據失敗: %s", str(e))
            raise

    def _normalize_symbol(self, symbol: str) -> str:
        """將各種格式的交易對轉換為OKX標準格式 BASE/QUOTE:QUOTE"""
        try:
            if '/' in symbol:
                base, quote = symbol.split('/')
            elif '-' in symbol:
                base, quote = symbol.split('-')
            else:
                raise ValueError(f"無法解析的交易對格式: {symbol}")
            
            okx_symbol = f"{base}/{quote}:{quote}"
            if okx_symbol not in self.exchange.symbols:
                raise ValueError(f"交易對 {okx_symbol} 不存在於OKX市場")
            return okx_symbol
        except Exception as e:
            logger.error("交易對轉換錯誤: %s", str(e))
            raise

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '3m',
        days: int = 180,
        limit: int = 100,
        output_dir: str = 'crypto_data',
        save_every: int = 100
    ) -> Tuple[Optional[Path], int]:
        """
        獲取OHLCV數據並保存
        返回: (文件路徑, 總條數) 或 (None, 0) 如果失敗
        """
        try:
            okx_symbol = self._normalize_symbol(symbol)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            since_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            output_path = Path(output_dir) / f"{symbol.replace('/', '-')}.parquet"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # 處理現有數據
            if output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    if not existing_df.empty:
                        last_timestamp = existing_df['timestamp'].max().timestamp() * 1000
                        since_timestamp = max(since_timestamp, int(last_timestamp) + 1)
                        logger.info("從現有數據恢復，最後時間戳: %s", pd.to_datetime(last_timestamp, unit='ms'))
                except Exception as e:
                    logger.warning("讀取現有文件失敗: %s，將重新開始", str(e))

            all_data = []
            total_bars = 0
            retry_count = 0
            max_retries = 3
            
            logger.info("開始獲取 %s 數據，時間範圍: %s 至 %s", 
                      okx_symbol, 
                      pd.to_datetime(since_timestamp, unit='ms'),
                      pd.to_datetime(end_timestamp, unit='ms'))

            while since_timestamp < end_timestamp and retry_count < max_retries:
                try:
                    # 獲取數據
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=okx_symbol,
                        timeframe=timeframe,
                        since=since_timestamp,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        logger.info("沒有更多數據可用")
                        break
                    
                    # 轉換數據
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # 檢查數據有效性
                    if df.isnull().values.any():
                        logger.warning("發現空值數據，跳過此批次")
                        continue
                    
                    all_data.append(df)
                    total_bars += len(df)
                    
                    # 更新時間戳
                    since_timestamp = ohlcv[-1][0] + 1
                    retry_count = 0  # 重置重試計數
                    
                    # 進度報告
                    progress = (since_timestamp - int(start_time.timestamp() * 1000)) / \
                               (end_timestamp - int(start_time.timestamp() * 1000))
                    logger.info("進度: %.1f%%, 已獲取 %d 條數據", min(100, progress * 100), total_bars)
                    
                    # 定期保存
                    if len(all_data) >= save_every or since_timestamp >= end_timestamp:
                        self._save_data(all_data, output_path)
                        all_data = []
                    
                    # 速率限制
                    time.sleep(max(0.1, self.exchange.rateLimit / 1000))
                    
                except ccxt.NetworkError as e:
                    retry_count += 1
                    logger.warning("網絡錯誤 (嘗試 %d/%d): %s", retry_count, max_retries, str(e))
                    time.sleep(5 ** retry_count)  # 指數退避
                except Exception as e:
                    logger.error("獲取數據時發生錯誤: %s", str(e))
                    break
            
            # 最終保存
            if all_data:
                self._save_data(all_data, output_path)
            
            if total_bars > 0:
                logger.info("成功獲取 %d 條 %s 數據", total_bars, okx_symbol)
                return output_path, total_bars
            else:
                logger.warning("未獲取到任何數據")
                return None, 0
                
        except Exception as e:
            logger.error("獲取OHLCV數據失敗: %s", str(e))
            return None, 0

    def _save_data(self, data: List[pd.DataFrame], output_path: Path):
        """保存數據到文件"""
        try:
            combined_df = pd.concat(data).drop_duplicates('timestamp').sort_values('timestamp')
            
            if output_path.exists():
                existing_df = pd.read_parquet(output_path)
                combined_df = pd.concat([existing_df, combined_df]) \
                               .drop_duplicates('timestamp') \
                               .sort_values('timestamp')
            
            combined_df.to_parquet(output_path)
            logger.info("成功保存 %d 條數據到 %s", len(combined_df), output_path)
        except Exception as e:
            logger.error("保存數據失敗: %s", str(e))

    def generate_instruments_file(
        self,
        symbols: List[str],
        output_dir: str = 'crypto_data'
    ) -> Optional[Path]:
        """生成交易對信息文件"""
        instruments = []
        
        for symbol in symbols:
            try:
                okx_symbol = self._normalize_symbol(symbol)
                market = self.exchange.market(okx_symbol)
                info = market.get('info', {})
                
                # 獲取精度信息
                tick_size = float(info.get('tickSz', '0.1'))
                lot_size = float(info.get('lotSz', '0.0001'))
                
                instrument = {
                    'symbol': symbol.replace('/', '-'),
                    'okx_symbol': okx_symbol,
                    'base': market['base'],
                    'quote': market['quote'],
                    'price_precision': self._calculate_precision(tick_size),
                    'size_precision': self._calculate_precision(lot_size),
                    'tick_size': tick_size,
                    'lot_size': lot_size,
                    'min_order_size': float(info.get('minSz', '0.01')),
                    'contract_type': market['type'],
                    'last_updated': datetime.utcnow().isoformat()
                }
                instruments.append(instrument)
                logger.info("處理交易對成功: %s", symbol)
                
            except Exception as e:
                logger.warning("處理交易對 %s 失敗: %s", symbol, str(e))
                continue
        
        if not instruments:
            logger.error("未成功處理任何交易對")
            return None
            
        try:
            df = pd.DataFrame(instruments)
            output_path = Path(output_dir) / "_instruments.parquet"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_parquet(output_path)
            logger.info("成功保存 %d 個交易對信息到 %s", len(df), output_path)
            return output_path
        except Exception as e:
            logger.error("保存交易對信息失敗: %s", str(e))
            return None

    def _calculate_precision(self, value: float) -> int:
        """計算數值的精度位數"""
        s = format(value, '.10f').rstrip('0')
        if '.' in s:
            return len(s.split('.')[1])
        return 0

def fetch_sample_data():
    try:
        fetcher = OKXDataFetcher()
        
        # 使用標準化格式的交易對 (支持 BTC-USDT 或 BTC/USDT)
        symbols = ["BTC-USDT", "ETH-USDT"]
        
        # 1. 生成交易對信息文件
        logger.info("\n" + "="*50 + "\n開始生成交易對信息文件")
        instruments_file = fetcher.generate_instruments_file(symbols)
        
        if not instruments_file:
            logger.error("無法生成交易對信息文件，退出")
            return
        
        # 2. 獲取每個交易對的OHLCV數據
        for symbol in symbols:
            logger.info("\n" + "="*50 + f"\n開始獲取 {symbol} 數據")
            file_path, total_bars = fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe="3m",
                days=7,  # 測試用7天數據
                save_every=50
            )
            
            if total_bars > 0:
                logger.info("成功獲取 %s 數據，共 %d 條", symbol, total_bars)
            else:
                logger.warning("未能獲取 %s 數據", symbol)
                
    except Exception as e:
        logger.error("主程序運行失敗: %s", str(e), exc_info=True)

if __name__ == "__main__":
    fetch_sample_data()