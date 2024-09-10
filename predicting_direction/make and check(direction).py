from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Activation, Attention
from keras.optimizers import Adam
import pandas as pd
import numpy as np

# Load and preprocess the data
data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')
data = data.iloc[:, 1:]
data['fast_ma'] = data['close'].rolling(window=12).mean()
data['slow_ma'] = data['close'].rolling(window=25).mean()
data['change'] = data['close'].shift(-1) - data['close']
data = data.iloc[26:-1, :]
data = data.values
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
print(sequences.shape, next_values.shape)

split_index = int(len(sequences) * 0.8)
    
train_sequences=sequences[:split_index]
train_next_values=next_values[:split_index]
test_sequences = sequences[split_index:]
test_next_values = next_values[split_index:]

# Train and evaluate models
for i in range(tries):

    # Define the input layer
    inputs = Input(shape=(sequence_length, 7))

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
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[])

    # Train the model on the training sequences and predictions
    model.fit(train_sequences, train_next_values, epochs=30, batch_size=20, verbose=1)

    mae_model_evaluation = model.evaluate(train_sequences, train_next_values)

    # Evaluate model accuracy on the test set
    mae = model.evaluate(test_sequences, test_next_values)

    print(f"Metrics on the test set (Sequence Length {sequence_length}): {mae}")

    print(f"Model Evaluation MAE: {mae_model_evaluation}")

    # Save the model with sequence length in the file name
    model.save(f'direction_{mae:.4f}.h5')
    print(f"Model with loss saved: {mae}")
