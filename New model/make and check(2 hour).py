from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Activation, Attention
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.metrics import MeanSquaredError
import pandas as pd
import numpy as np

# Load and preprocess the data
data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')
data = data.iloc[:, 1:]
data = data.values
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# Define sequence length
sequence_length = 20 
tries = 100

# Prepare sequences and predictions once for all models
sequences = []
next_values = []

for i in range(len(standardized_data) - sequence_length-1):
    sequences.append(standardized_data[i:i + sequence_length])
    next_values.append(standardized_data[i+sequence_length:i+sequence_length+2,-2])

sequences = np.array(sequences)
next_values = np.array(next_values)

print(sequences.shape, next_values.shape)

split_index = int(len(sequences) * 0.8)
    
train_sequences=sequences[:split_index]
train_next_values=next_values[:split_index]
test_sequences = sequences[split_index:]
test_next_values = next_values[split_index:]

# Train and evaluate models
for i in range(tries):

    # Define the input layer
    inputs = Input(shape=(sequence_length, 5))

    # Add Convolutional layer
    conv_layer = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(inputs)

    # Add MaxPooling layer
    maxpool_layer = MaxPooling1D(pool_size=1, padding='same')(conv_layer)
    activation_layer = Activation('relu')(maxpool_layer)

    # Add BiLSTM layer
    bilstm_layer = Bidirectional(LSTM(units=64, return_sequences=False))(activation_layer)

    # Attention Mechanism
    attention = Attention()([bilstm_layer, bilstm_layer])

    # Add output layer with 2 units for regression
    output = Dense(units=2)(attention)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    # Compile the model with specified parameters and metrics
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[MeanSquaredError()])

    # Train the model on the training sequences and predictions
    model.fit(train_sequences, train_next_values, epochs=50, batch_size=32, verbose=0)

    # Evaluate model accuracy on the test set
    metrics = model.evaluate(test_sequences, test_next_values)
    print(f"Metrics on the test set (Sequence Length {sequence_length}): {metrics}")

    loss = metrics[0]
    print(f"Loss on the test set (Sequence Length {sequence_length}): {loss}")

    # Save the model with sequence length in the file name
    if(metrics[0]<0.01 ):
        model.save(f'BTC_{loss:.4f}.h5')
        print(f"Model0 saved, loss: {loss}")