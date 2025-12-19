import yfinance as yf
import pandas as pd

def get_intraday_stock_data(symbol: str, period: str = "1d", interval: str = "1m"):
    data = yf.download(
        tickers=symbol, 
        period=period, 
        interval=interval, 
        progress=False, 
        auto_adjust=False,
        multi_level_index=False  
    )
    
    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    
    df.columns = ["open", "high", "low", "close", "volume"]
    return df

if __name__ == "__main__":
    stock_symbol = "TSLA"
    stock_df = get_intraday_stock_data(stock_symbol)
    
    if not stock_df.empty:
        print(stock_df.tail())
    else:
        print("No data found!")