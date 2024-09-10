from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Activation, Attention
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError, MeanSquaredError

# Define the input layer
inputs = Input(shape=(5, 5))

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
output = Dense(units=1)(attention)

# Create the model
model = Model(inputs=inputs, outputs=output)

# Compile the model with specified parameters and metrics
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[MeanAbsoluteError(),MeanSquaredError()])

# Save the model as an H5 file
model.save('CNN_BiLSTM_AM.h5')

model.summary()