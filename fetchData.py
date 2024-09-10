import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Initialize the Binance exchange object
exchange = ccxt.binance()

# Define the trading pair and timeframe
symbol = 'BTC/USDT'
timeframe = '1h'

# Initialize an empty list to store historical data chunks
historical_data_chunks = []

# Set the end date to the current date and time
end_date = datetime.now() - timedelta(days=(90)) 

# Calculate the start date for fetching last 3 months of data
start_date = end_date - timedelta(days=(90+3*365)) 

# Fetch historical data in chunks
while end_date > start_date:
    # Fetch historical OHLCV (Open, High, Low, Close, Volume) data for the chunk
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(start_date.timestamp()) * 1000,
                                  limit=1000)  # Limit per request is 1000 candles
    
    # If no data is returned, break the loop
    if len(ohlcv) == 0:
        break
    
    # Convert the data to a pandas DataFrame
    chunk_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Append the chunk data to the list of chunks
    historical_data_chunks.append(chunk_df)
    
    # Update the start date for the next chunk
    start_date = pd.to_datetime(chunk_df['timestamp'].iloc[-1], unit='ms') + timedelta(hours=1)

# Concatenate all chunks into a single DataFrame
historical_data = pd.concat(historical_data_chunks, ignore_index=True)

# Convert the timestamp to a datetime format
historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms')

# Set the timestamp as the DataFrame's index
historical_data.set_index('timestamp', inplace=True)

historical_data = historical_data.sort_values('timestamp')

# Print the first few rows of the data
print(historical_data.head())

# Optionally, you can save the data to a CSV file
historical_data.to_csv('new.csv')