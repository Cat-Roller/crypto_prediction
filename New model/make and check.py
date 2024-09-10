from tensorflow import keras 
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Activation, Attention
from keras.optimizers import Adam
from keras.metrics import MeanSquaredError
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load and preprocess the data
data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')
data = data.iloc[:, 1:]
data = data.values
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# Define sequence length and number of tries
sequence_length = 13 
tries = 10 

# Prepare sequences and predictions
sequences = []
next_values = []
for i in range(len(standardized_data) - sequence_length):
    sequences.append(standardized_data[i:i + sequence_length])
    next_values.append(standardized_data[i + sequence_length][-2])
    
sequences = np.array(sequences)
next_values = np.array(next_values)

nan_indices = np.isnan(sequences).any(axis=(1, 2)) | np.isnan(next_values)
sequences = sequences[~nan_indices]
next_values = next_values[~nan_indices]

# Split the data into train and test sets using sklearn
train_sequences, test_sequences, train_next_values, test_next_values = train_test_split(sequences, next_values, test_size=0.2, random_state=42, shuffle=True)

print(train_sequences.shape, train_next_values.shape)

# Train and evaluate models for the specified number of tries
for i in range(tries):
    print(f"Training model for try {i+1}")

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

    # Add output layer
    output = Dense(units=2)(attention)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=output) # Use keras.Model instead of Model

    # Compile the model with specified parameters and metrics
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[MeanSquaredError()])

    # Train the model on the training sequences and predictions
    model.fit(train_sequences, train_next_values, epochs=15, batch_size=32, verbose=0)

    # Evaluate model accuracy on the test set
    accuracy = model.evaluate(test_sequences, test_next_values)
    print(f"Accuracy on the test set (Try {i+1}): {accuracy}")

    loss = accuracy[0]
    print(f"Loss on the test set (Try {i+1}): {loss}")

    # Check if loss is less than 0.006
    if loss < 0.006:
        # Save the model with try number in the file name
        model.save(f'CBA_try_{i+1}_loss_{loss:.4f}.h5')
        print(f"Model for try {i+1} saved due to low loss: {loss}")
