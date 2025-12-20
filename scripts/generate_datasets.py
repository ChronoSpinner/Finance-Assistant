import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
TICKERS = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC"]
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
LOOKBACK_WINDOW = 30    
FUTURE_TARGET = 5       
RETURN_THRESHOLD = 0.015 
IMG_SIZE = (64, 64)

# Folders
DATASET_DIR = "dataset_v1"
IMG_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

def fetch_data_safe(ticker):
    """Downloads data for a SINGLE ticker."""
    print(f"Downloading {ticker}...")
    try:
        # Auto_adjust=True fixes the 'FutureWarning' you saw
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False, auto_adjust=True)
        
        if df.empty:
            return None
            
        df.columns = [c.capitalize() for c in df.columns]
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            return None
            
        return df.dropna()
    except Exception as e:
        print(f"ERROR downloading {ticker}: {e}")
        return None

def create_labels(df):
    """Creates target labels."""
    df = df.copy()
    
    # Target: Return from Tomorrow's Open (t+1) to (t+1+5)
    buy_price = df['Open'].shift(-1) 
    sell_price = df['Open'].shift(-(1 + FUTURE_TARGET))
    
    df['return'] = (sell_price - buy_price) / buy_price
    
    conditions = [
        (df['return'] > RETURN_THRESHOLD), 
        (df['return'] < -RETURN_THRESHOLD)
    ]
    choices = [1, 0]
    
    df['label'] = np.select(conditions, choices, default=np.nan)
    df_clean = df.dropna(subset=['label'])
    return df_clean

def generate_chart(window_df, filename):
    """Generates the minimalist Computer Vision chart."""
    mc = mpf.make_marketcolors(up='g', down='r', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc)
    
    params = dict(
        type='candle',
        style=s,
        volume=True,
        axisoff=True,
        figscale=1.0,
        savefig=dict(fname=filename, bbox_inches='tight', pad_inches=0)
    )
    mpf.plot(window_df, **params, returnfig=True, closefig=True)

def main():
    labeled_data = []
    
    # --- 1. Fetch Data ---
    cache_dfs = {} # Save DFs here so we don't download twice
    
    for ticker in TICKERS:
        df = fetch_data_safe(ticker)
        if df is None: continue
        
        # Save to cache for later image generation
        cache_dfs[ticker] = df
        
        if len(df) < LOOKBACK_WINDOW + FUTURE_TARGET:
            continue
            
        df_labeled = create_labels(df)
        df_labeled['ticker'] = ticker
        
        # IMPORTANT FIX: Save the index (Date) as a column before it gets lost!
        df_labeled['Date'] = df_labeled.index
        
        print(f"  > {ticker}: Found {len(df_labeled)} signals.")
        labeled_data.append(df_labeled)

    if not labeled_data:
        print("No valid data found.")
        return

    full_df = pd.concat(labeled_data)
    
    # --- 2. Class Balancing ---
    buy_df = full_df[full_df['label'] == 1.0]
    sell_df = full_df[full_df['label'] == 0.0]
    
    print(f"\nTotal: {len(buy_df)} Buys, {len(sell_df)} Sells")
    
    if len(buy_df) == 0 or len(sell_df) == 0:
        print("Error: One class has 0 samples.")
        return

    min_count = min(len(buy_df), len(sell_df))
    print(f"Balancing to {min_count} samples each...")
    
    buy_balanced = buy_df.sample(n=min_count, random_state=42)
    sell_balanced = sell_df.sample(n=min_count, random_state=42)
    
    # We reset index, BUT we now have 'Date' as a column, so it's safe.
    final_df = pd.concat([buy_balanced, sell_balanced]).sample(frac=1).reset_index(drop=True)
    
    # --- 3. Generate Images ---
    print(f"\nGenerating {len(final_df)} images...")
    
    final_metadata = []
    
    for i, row in tqdm(final_df.iterrows(), total=len(final_df)):
        ticker = row['ticker']
        target_date = row['Date'] # FIX: Access the preserved Date column
        
        # Use cached dataframe instead of downloading again
        if ticker not in cache_dfs: continue
        original_df = cache_dfs[ticker]
        
        try:
            # Find the location of the date
            idx = original_df.index.get_loc(target_date)
        except KeyError:
            continue
            
        # Slice window
        start_idx = idx - LOOKBACK_WINDOW + 1
        if start_idx < 0: continue
        
        window = original_df.iloc[start_idx : idx + 1]
        
        # Filename
        label_str = "buy" if row['label'] == 1.0 else "sell"
        safe_date = str(target_date).split(" ")[0]
        fname = f"{label_str}_{ticker}_{safe_date}.png"
        fpath = os.path.join(IMG_DIR, fname)
        
        try:
            generate_chart(window, fpath)
            final_metadata.append([fname, int(row['label'])])
        except Exception as e:
            print(f"Err: {e}")

    # 4. Save Metadata
    if len(final_metadata) > 0:
        meta_df = pd.DataFrame(final_metadata, columns=['filename', 'label'])
        meta_df.to_csv(os.path.join(DATASET_DIR, "labels.csv"), index=False)
        print(f"\nSuccess! Generated {len(meta_df)} images.")
    else:
        print("\nFailed to generate images.")

if __name__ == "__main__":
    main()