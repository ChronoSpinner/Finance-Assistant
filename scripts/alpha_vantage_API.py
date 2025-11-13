import requests
import pandas as pd

def get_api_key(file_path="API_KEYS/Alpha_Vantage"):
    with open(file_path, "r") as file:
        key = file.read().strip()
    return key

API_KEY = get_api_key()
BASE_URL = "https://www.alphavantage.co/query"

def get_intraday_stock_data(symbol, interval="5min", outputsize='compact'):
    parameters = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY
    }

    response = requests.get(BASE_URL, params=parameters)
    data = response.json()

    key_name=f'Time Series ({interval})'

    if key_name not in data:
        raise ValueError(f"Error fetching data: {data.get('Note', 'Unknown error')}")
        return None
    
    time_series = data[key_name]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return df

if __name__ == "__main__":
    stock_symbol = "IBM"
    stock_df = get_intraday_stock_data(stock_symbol)

    if stock_df is not None:
        print(stock_df.head())