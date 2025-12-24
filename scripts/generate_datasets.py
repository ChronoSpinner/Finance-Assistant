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
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "PLTR", "COIN", "SNOW", "UBER", "ABNB", "CRWD", "PANW", "ROKU", "SHOP", 
    "PYPL", "ZM", "DOCU", "PTON", "INTC", "MMM", 
    "JPM", "GS", "BAC", "MS", "V", "MA", "AXP", "BLK", "C",
    "XOM", "CVX", "CAT", "DE", "LMT", "BA", "GE", "UNP",
    "COST", "WMT", "TGT", "HD", "MCD", "PEP", "KO", "PG", "LLY", "UNH"
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
NEG_POS_RATIO = 1.2         
IMG_SIZE = 224              
DPI = 96
IMG_DIM_INCH = IMG_SIZE / DPI
DATA_DIR = "swing_dataset_v8_market_context"

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

def apply_triple_barrier_atr(df):
    df = df.copy()
    future_labels = []
    closes, highs, lows, atrs, dates = df['Close'].values, df['High'].values, df['Low'].values, df['ATR'].values, df.index
    for i in range(len(df) - PREDICTION_HORIZON):
        entry_price, current_atr = closes[i], atrs[i]
        if np.isnan(current_atr) or current_atr <= 0: continue
        stop_price = entry_price - (current_atr * ATR_MULTIPLIER)
        take_profit = entry_price + (current_atr * ATR_MULTIPLIER * RISK_REWARD_RATIO)
        outcome, max_p = 0, 0.0
        for f in range(1, PREDICTION_HORIZON + 1):
            idx = i + f
            if lows[idx] <= stop_price: break
            if highs[idx] >= take_profit:
                outcome = 1
                break
            max_p = max(max_p, highs[idx] - entry_price)
        future_labels.append({'date': dates[i], 'idx': i, 'label': outcome, 'max_profit': max_p})
    return pd.DataFrame(future_labels)

# --- REFINED NORMALIZATION & CHARTING ---

def normalize_window(df, spy_slice=None):
    """
    Fixed Scaling: Anchors both Stock and SPY to 1.0 at start of window.
    This enables visual Relative Strength analysis.
    """
    df = df.copy()
    ref_price = df['Close'].iloc[0]
    if ref_price == 0: return df, None
    
    # Normalize Stock Prices (First Close = 1.0)
    cols = ['Open', 'High', 'Low', 'Close', 'EMA_50', 'EMA_200', 'Upper_BB', 'Lower_BB']
    for col in [c for c in cols if c in df.columns]:
        df[col] = df[col] / ref_price

    # Normalize SPY to the SAME STARTING ANCHOR (1.0)
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
        # Thicker lines for 224x224 visibility
        if 'EMA_50' in window_df.columns:
            addplots.append(mpf.make_addplot(window_df['EMA_50'], color='orange', width=2.2))
        if 'EMA_200' in window_df.columns:
            addplots.append(mpf.make_addplot(window_df['EMA_200'], color='white', width=2.5, linestyle='--'))
        if 'Upper_BB' in window_df.columns:
            addplots.append(mpf.make_addplot(window_df[['Upper_BB', 'Lower_BB']], color='cyan', width=1.2, alpha=0.3))
        
        # Volume Panel
        addplots.append(mpf.make_addplot(window_df['Volume'], panel=1, type='bar', color='yellow', width=0.8, alpha=0.7))
        
        # RSI Panel
        if 'RSI' in window_df.columns:
             addplots.append(mpf.make_addplot(window_df['RSI'], panel=2, color='magenta', width=2.0, ylim=(0, 100)))

        # SPY OVERLAY - Fixed alpha and scaling
        if normalized_spy is not None:
            addplots.append(mpf.make_addplot(normalized_spy, color='#9b59b6', width=2.8, alpha=0.6))

        fig, _ = mpf.plot(window_df, type='candle', style=s, addplot=addplots, volume=False, 
                          figsize=(IMG_DIM_INCH, IMG_DIM_INCH), panel_ratios=(4, 1, 1), 
                          axisoff=True, returnfig=True, scale_padding=0, tight_layout=True)
        
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight', pad_inches=0, facecolor='black')
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
        
        # Download in one batch for speed
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
        print("\n--- 2. Labeling & Splitting ---")
        all_tasks, spy_close = [], data_store.get(BENCHMARK_TICKER, pd.DataFrame())['Close']
        train_cut, val_cut = pd.Timestamp(TRAIN_CUTOFF), pd.Timestamp(VAL_CUTOFF)
        
        for ticker, df in tqdm(data_store.items()):
            if ticker == BENCHMARK_TICKER: continue
            labeled_df = apply_triple_barrier_atr(df)
            
            # Simple Time Split
            splits = [
                ('train', labeled_df[labeled_df['date'] < train_cut]),
                ('val', labeled_df[(labeled_df['date'] >= train_cut + pd.Timedelta(days=LOOKBACK_WINDOW)) & (labeled_df['date'] < val_cut)]),
                ('test', labeled_df[labeled_df['date'] >= val_cut + pd.Timedelta(days=LOOKBACK_WINDOW)])
            ]
            
            for split_name, subset in splits:
                if subset.empty: continue
                pos, neg = subset[subset['label'] == 1], subset[subset['label'] == 0]
                # Hard Negative Mining
                n_neg = int(len(pos) * NEG_POS_RATIO)
                neg_sampled = pd.concat([neg.sort_values('max_profit', ascending=False).iloc[:n_neg//2], 
                                         neg.sample(min(len(neg), n_neg//2))]) if not pos.empty else neg.iloc[:0]
                
                for _, row in pd.concat([pos, neg_sampled]).iterrows():
                    end_idx = row['idx']
                    window = df.iloc[end_idx - LOOKBACK_WINDOW + 1 : end_idx + 1]
                    if len(window) < LOOKBACK_WINDOW: continue
                    
                    spy_slice = spy_close.reindex(window.index).ffill().bfill() if not spy_close.empty else None
                    all_tasks.append((window.copy(), spy_slice, ticker, str(row['date'].date()), row['label'], split_name, DATA_DIR))
        return all_tasks

    def run(self):
        for s in ['train', 'val', 'test']:
            for l in ['0', '1']: os.makedirs(os.path.join(DATA_DIR, s, l), exist_ok=True)
        tasks = self.process_metadata(self.get_data())
        print(f"Generating {len(tasks)} images...")
        random.shuffle(tasks)
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-1)) as ex:
            list(tqdm(ex.map(generate_chart, tasks), total=len(tasks)))

if __name__ == "__main__":
    Pipeline().run()