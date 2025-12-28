import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
import gc
import random

# --- SYSTEM CONFIG ---
warnings.filterwarnings("ignore")
matplotlib.use('Agg') 
random.seed(42)
np.random.seed(42)

# --- TRADING CONFIG ---
TICKERS = [
    # TECHNOLOGY (Mega Cap & Growth)
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "CRM", "ADBE", "INTC", "CSCO", "ORCL", "QCOM", "TXN", "AVGO", "MU", "LRCX",
    "AMAT", "IBM", "NOW", "UBER", "ABNB", "PLTR", "SNOW", "PANW", "CRWD", "FTNT",
    "ZS", "NET", "TEAM", "SHOP", "ZM", "DOCU", "ROKU", "TWLO", "DDOG",

    # FINANCIALS (Banks, Payments, Asset Managers)
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    "COIN", "HOOD", "KKR", "BX", "USB", "PNC", "TFC", "BK", "STT", "CME", "ICE",

    # CONSUMER DISCRETIONARY (Retail, Auto, Services)
    "HD", "LOW", "MCD", "SBUX", "NKE", "TGT", "TJX", "COST", "WMT", "DG", "DLTR",
    "LULU", "CMG", "MAR", "HLT", "BKNG", "EXPE", "RCL", "CCL", "NCLH", "F", "GM",
    "TM", "HMC", "TSCO", "ORLY", "AZO", "ULTA",

    # HEALTHCARE (Pharma, Biotech, Devices)
    "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "AMGN", "GILD", "BIIB", "VRTX",
    "REGN", "ISRG", "SYK", "EW", "BSX", "MDT", "ABT", "TMO", "DHR", "BMY", "CVS",
    "CI", "HUM", "ELV", "MRNA", "BNTX",

    # ENERGY (Oil, Gas, Solar)
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX", "VLO", "HAL",
    "BKR", "KMI", "WMB", "OKE", "ENPH", "SEDG", "FSLR", "NEE", "DUK", "SO",

    # INDUSTRIALS (Defense, Aerospace, Machinery)
    "CAT", "DE", "HON", "GE", "MMM", "ETN", "ITW", "EMR", "PH", "CMI", "PCAR",
    "LMT", "RTX", "BA", "GD", "NOC", "LHX", "HII", "TDG", "TXT",
    "UPS", "FDX", "UNP", "CSX", "NSC", "DAL", "UAL", "AAL", "LUV",

    # MATERIALS & COMMODITIES (Miners, Chemicals, Steel)
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "SCCO", "AA", "NUE", "STLD",
    "CLF", "MOS", "CF", "CTRA", "DOW", "DD",

    # COMMUNICATION & ENTERTAINMENT
    "DIS", "CMCSA", "CHTR", "TMUS", "VZ", "T", "WBD", "LYV",

    # REAL ESTATE (REITs)
    "PLD", "AMT", "CCI", "EQIX", "DLR", "PSA", "O", "SPG", "VICI", "WELL"
]

BENCHMARK_TICKER = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
TRAIN_CUTOFF = "2023-01-01" 
VAL_CUTOFF = "2024-01-01"       
LOOKBACK_WINDOW = 64        
PREDICTION_HORIZON = 10     
RISK_REWARD_RATIO = 2.0     
ATR_MULTIPLIER = 1.5        
IMG_SIZE = 224              
DPI = 96
IMG_DIM_INCH = IMG_SIZE / DPI
DATA_DIR = "swing_dataset_3class_exact224" 

# --- HELPER FUNCTIONS ---

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9) 
    return 100 - (100 / (1 + rs))

def calculate_indicators(df):
    df = df.copy()
    prev_close = df['Close'].shift()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - prev_close)
    low_close = np.abs(df['Low'] - prev_close)
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA_20'] + (2 * df['STD_20'])
    df['Lower_BB'] = df['SMA_20'] - (2 * df['STD_20'])
    df['RSI'] = calculate_rsi(df['Close'])
    return df.dropna()

def apply_triple_barrier_3class(df):
    """
    Applies 3-Class Labeling:
    0 = SELL (Hit Stop Loss first)
    1 = HOLD (Hit neither / Time limit)
    2 = BUY  (Hit Take Profit first)
    """
    df = df.copy()
    future_labels = []
    closes, highs, lows, atrs, dates = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values, df.index
    
    for i in range(len(df) - PREDICTION_HORIZON):
        entry_price = closes[i]
        current_atr = atrs[i]
        
        if np.isnan(current_atr) or current_atr <= 0: continue
        
        # Define Barriers
        stop_loss = entry_price - (current_atr * ATR_MULTIPLIER)
        take_profit = entry_price + (current_atr * ATR_MULTIPLIER * RISK_REWARD_RATIO)
        
        outcome = 1 # Default to HOLD
        
        for f in range(1, PREDICTION_HORIZON + 1):
            idx = i + f
            
            # Check Stop Loss (SELL Signal - 0)
            if lows[idx] <= stop_loss:
                outcome = 0
                break
            
            # Check Take Profit (BUY Signal - 2)
            if highs[idx] >= take_profit:
                outcome = 2
                break
        
        future_labels.append({'date': dates[i], 'idx': i, 'label': outcome})
        
    return pd.DataFrame(future_labels)

# --- REFINED NORMALIZATION & CHARTING ---

def normalize_window(df, spy_slice=None):
    df = df.copy()
    ref_price = df['Close'].iloc[0]
    if ref_price == 0: return df, None
    
    # Normalize Stock Prices (First Close = 1.0)
    cols = ['Open', 'High', 'Low', 'Close', 'EMA_50', 'EMA_200', 'Upper_BB', 'Lower_BB']
    for col in [c for c in cols if c in df.columns]:
        df[col] = df[col] / ref_price

    # Normalize SPY
    normalized_spy = None
    if spy_slice is not None:
        spy_ref = spy_slice.iloc[0]
        if spy_ref > 0:
            normalized_spy = spy_slice / spy_ref

    # Volume Normalization
    avg_vol = df['Volume'].mean()
    df['Volume'] = (df['Volume'] / avg_vol).clip(upper=5.0) if avg_vol > 0 else 0
    return df, normalized_spy

def generate_chart(args):
    window_df, spy_slice, ticker, date_str, label, split, save_dir = args
    filepath = os.path.join(save_dir, split, str(label), f"{ticker}_{date_str}_{label}.png")
    if os.path.exists(filepath): return 0

    try:
        window_df, normalized_spy = normalize_window(window_df, spy_slice)
        mc = mpf.make_marketcolors(up='#00FF00', down='#FF0000', edge='inherit', wick='inherit', volume='inherit')
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle='', facecolor='black', edgecolor='black', figcolor='black')

        addplots = []
        
        # LAYER 1: SPY
        if normalized_spy is not None:
            addplots.append(mpf.make_addplot(normalized_spy, color='#9b59b6', width=2.8, alpha=0.6))

        # LAYER 2: INDICATORS
        if 'EMA_50' in window_df.columns:
            addplots.append(mpf.make_addplot(window_df['EMA_50'], color='orange', width=2.2))
        if 'EMA_200' in window_df.columns:
            addplots.append(mpf.make_addplot(window_df['EMA_200'], color='white', width=2.5, linestyle='--'))
        if 'Upper_BB' in window_df.columns:
            addplots.append(mpf.make_addplot(window_df[['Upper_BB', 'Lower_BB']], color='cyan', width=1.7, alpha=0.8))
        
        # LAYER 3: PANELS
        addplots.append(mpf.make_addplot(window_df['Volume'], panel=1, type='bar', color='yellow', width=0.8, alpha=0.7))
        if 'RSI' in window_df.columns:
             addplots.append(mpf.make_addplot(window_df['RSI'], panel=2, color='magenta', width=2.0, ylim=(0, 100)))

        # PLOTTING
        # Note: We use tight_layout=False here because we will manually adjust subplots to fill 100%
        fig, _ = mpf.plot(window_df, type='candle', style=s, addplot=addplots, volume=False, 
                          figsize=(IMG_DIM_INCH, IMG_DIM_INCH), panel_ratios=(4, 1, 1), 
                          axisoff=True, returnfig=True, scale_padding=0.02, tight_layout=False)
        
        # FORCE EXACT DIMENSIONS
        # This removes all margins and forces the plot to fill the exact figure size
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Save without bbox_inches='tight' and with 0 padding to keep strict pixels
        fig.savefig(filepath, dpi=DPI, bbox_inches=None, pad_inches=0, facecolor='black')
        
        plt.close(fig)
        return 1
    except Exception:
        plt.close('all')
        return 0

# --- PIPELINE CLASS ---

class Pipeline:
    def __init__(self):
        self.raw_dir = os.path.join(DATA_DIR, "raw_pickle")
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def get_data(self):
        print("--- 1. Data Ingestion ---")
        data_store = {}
        all_tickers = list(set(TICKERS + [BENCHMARK_TICKER]))
        
        raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=True)
        
        for t in all_tickers:
            try:
                df = raw[t].dropna() if len(all_tickers) > 1 else raw.dropna()
                if len(df) < 200: continue
                if t != BENCHMARK_TICKER: df = calculate_indicators(df)
                df.to_pickle(os.path.join(self.raw_dir, f"{t}.pkl"))
                data_store[t] = df
            except: continue
        return data_store

    def process_metadata(self, data_store):
        print("\n--- 2. Labeling & 3-Class Balancing ---")
        all_tasks, spy_close = [], data_store.get(BENCHMARK_TICKER, pd.DataFrame())['Close']
        train_cut, val_cut = pd.Timestamp(TRAIN_CUTOFF), pd.Timestamp(VAL_CUTOFF)
        
        for ticker, df in tqdm(data_store.items()):
            if ticker == BENCHMARK_TICKER: continue
            
            # Apply 3-Class Labeling
            labeled_df = apply_triple_barrier_3class(df)
            
            splits = [
                ('train', labeled_df[labeled_df['date'] < train_cut]),
                ('val', labeled_df[(labeled_df['date'] >= train_cut + pd.Timedelta(days=LOOKBACK_WINDOW)) & (labeled_df['date'] < val_cut)]),
                ('test', labeled_df[labeled_df['date'] >= val_cut + pd.Timedelta(days=LOOKBACK_WINDOW)])
            ]
            
            for split_name, subset in splits:
                if subset.empty: continue
                
                # SEPARATE CLASSES
                sells = subset[subset['label'] == 0] # Sell
                holds = subset[subset['label'] == 1] # Hold (Majority)
                buys  = subset[subset['label'] == 2] # Buy
                
                # --- BALANCING STRATEGY (1:1:1 APPROX) ---
                target_count = max(len(sells), len(buys))
                if target_count < 10: target_count = 50 
                
                if len(holds) > target_count:
                    holds_sampled = holds.sample(n=target_count, random_state=42)
                else:
                    holds_sampled = holds

                balanced_subset = pd.concat([sells, holds_sampled, buys])
                
                # Create Tasks
                for _, row in balanced_subset.iterrows():
                    end_idx = row['idx']
                    window = df.iloc[end_idx - LOOKBACK_WINDOW + 1 : end_idx + 1]
                    if len(window) < LOOKBACK_WINDOW: continue
                    
                    spy_slice = spy_close.reindex(window.index).ffill().bfill() if not spy_close.empty else None
                    all_tasks.append((window.copy(), spy_slice, ticker, str(row['date'].date()), int(row['label']), split_name, DATA_DIR))
                    
        return all_tasks

    def run(self):
        # Create directories for 3 classes: 0 (Sell), 1 (Hold), 2 (Buy)
        for s in ['train', 'val', 'test']:
            for l in ['0', '1', '2']: 
                os.makedirs(os.path.join(DATA_DIR, s, l), exist_ok=True)
                
        tasks = self.process_metadata(self.get_data())
        print(f"Generating {len(tasks)} images (3-Class Balanced, Exact 224x224)...")
        random.shuffle(tasks)
        
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-1)) as ex:
            list(tqdm(ex.map(generate_chart, tasks), total=len(tasks)))

if __name__ == "__main__":
    Pipeline().run()