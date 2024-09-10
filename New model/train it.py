from keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('CNN_BiLSTM_AM.h5')

# Load and preprocess the data
# Load your data (replace this with your data loading logic)
data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')

# Drop the first column
data = data.iloc[:, 1:]

# Reverse row order
data = data.values

# Z-score standardization
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# Define the sequence length (number of rows as input)
sequence_length = 5

# Prepare sequences and predictions
sequences = []
next_values = []
for i in range(len(standardized_data) - sequence_length):
    sequences.append(standardized_data[i:i + sequence_length])
    next_values.append(standardized_data[i + sequence_length - 1][-2])

# Convert sequences and predictions to numpy arrays
sequences = np.array(sequences)
next_values = np.array(next_values)

nan_indices = np.isnan(sequences).any(axis=(1, 2)) | np.isnan(next_values)

sequences = sequences[~nan_indices]
next_values = next_values[~nan_indices]

# Split the sequences and predictions into training and testing sets (80/20 split)
split_index = int(len(sequences) * 0.8)

train_sequences = sequences[:split_index]
train_next_values = next_values[:split_index]
test_sequences = sequences[split_index:]
test_next_values = next_values[split_index:]

# Train the model on the training sequences and predictions
model.fit(train_sequences, train_next_values, epochs=15, batch_size=32)

# Evaluate model accuracy on the test set
accuracy = model.evaluate(test_sequences, test_next_values)
print(f"Accuracy on the test set: {accuracy}")

model.save('CNN_BiLSTM_AM_trained4.h5')