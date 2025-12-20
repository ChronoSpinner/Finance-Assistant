import yfinance as yf
import mplfinance as mpf
import pandas as pd
import os

# Setup
SYMBOL = "AAPL"
LOOKBACK_WINDOW = 30  # How many candles the AI sees at once
OUTPUT_FOLDER = "datasets/images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#    Fetch Data
data = yf.download(SYMBOL, period="2y", interval="1d", progress=False, multi_level_index=False,auto_adjust=False)

#    The Generator Loop
for i in range(len(data) - LOOKBACK_WINDOW):
    
    # Slice the 30-day window
    window = data.iloc[i : i + LOOKBACK_WINDOW]
    
    # Generate Filename ("AAPL_2023-11-01.png")
    date_str = window.index[-1].strftime("%Y-%m-%d")
    filename = f"{OUTPUT_FOLDER}/{SYMBOL}_{date_str}.png"
    

    mpf.plot(
        window, 
        type='candle', 
        style='charles', 
        volume=False, 
        axisoff=True, 
        savefig=dict(fname=filename, bbox_inches='tight', pad_inches=0)
    )
    
    if i % 50 == 0: print(f"Generated {i} images...")

print("Charts Generated!")