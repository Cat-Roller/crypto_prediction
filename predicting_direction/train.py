from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import numpy as np

#Load model
model = load_model('direction_v2.h5')

# Load and preprocess the data
data = pd.read_csv('xrp_usdt_price_3_months_1h_candles.csv')
data = data.iloc[:, 1:]
data['fast_ma'] = data['close'].rolling(window=12).mean()
data['slow_ma'] = data['close'].rolling(window=25).mean()
data['change'] = data['close'].shift(-1) - data['close']
data = data.iloc[26:-1, :]
data = data.values

#Standartize data
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# Define sequence length
sequence_length = 13  # Change this to your desired sequence length
tries = 100

# Prepare sequences and predictions once for all models
sequences = []
next_values = []

for i in range(len(standardized_data) - sequence_length - 2):
    sequences.append(standardized_data[i:i + sequence_length,:-1])
    next_values.append([standardized_data[i,-1],standardized_data[i+1, -1]]) 

sequences = np.array(sequences)
next_values = np.array(next_values)
print(sequences[:1])
print(next_values[:1])

# Train the model on the training sequences and predictions
model.fit(sequences, next_values, epochs=3, batch_size=20, verbose=1)

mae_model_evaluation = model.evaluate(sequences, next_values)

print(f"Model Evaluation MAE: {mae_model_evaluation}")

# Save the model with sequence length in the file name
model.save(f'direction_v2.h5')