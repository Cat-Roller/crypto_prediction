from keras.models import load_model
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt

# Initialize the Binance exchange object
exchange = ccxt.binance()

#Load trained model
model = load_model('direction_0.1544.h5')

# Define the trading pair and timeframe
symbol = 'BTC/USDT'
timeframe = '1h'
sequence_length = 13

# Initialize an empty list to store historical data chunks
historical_data_chunks = []

# Set the end date to the current date and time
end_date = datetime.now() 

# Calculate the start date for fetching last 3 months of data
start_date = end_date - timedelta(days=(90)) 

historical_data_chunks = []

while end_date > start_date:
    # Fetch historical OHLCV (Open, High, Low, Close, Volume) data for the chunk
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(start_date.timestamp()) * 1000, limit=2000)
    
    # If no data is returned, break the loop
    if len(ohlcv) == 0:
        break
    
    # Convert the data to a pandas DataFrame
    chunk_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Append the chunk data to the list of chunks
    historical_data_chunks.append(chunk_df)
    
    # Update the start date for the next chunk
    start_date = pd.to_datetime(chunk_df['timestamp'].iloc[-1], unit='ms') + timedelta(hours=1)

#Prepare and standartize data
data = pd.concat(historical_data_chunks, ignore_index=True)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data['timestamp'] = data['timestamp'] + pd.Timedelta(hours=4)
data['fast_ma'] = data['close'].rolling(window=12).mean()
data['slow_ma'] = data['close'].rolling(window=25).mean()
data['change'] = data['close'].shift(-1) - data['close']
pd.options.display.float_format = '{:.2f}'.format
print(data.tail(4))
data = data.iloc[26:-1, 1:]
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = data.values
mean = mean.values
std = std.values
standardized_data = (data - mean) / std

#Separate the needed sequence
sequence = standardized_data[-sequence_length : , :-1]
sequence = np.array(sequence)
sequence = sequence.reshape((-1, 13, 7))


#Make prediction
prediction = model.predict(sequence)

#Destandartize prediction
mean_prediction = np.mean(data[:,-1])
std_prediction = np.std(data[:,-1])
destandartized_prediction = (prediction * std_prediction) + mean_prediction

print(prediction)
print(destandartized_prediction)