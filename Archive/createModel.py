from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import LeakyReLU
from keras.metrics import MeanSquaredError
from keras.optimizers import Adam
from keras.losses import mean_absolute_error

# Define the model
model = Sequential()
model.add(LSTM(50, input_shape=(120, 5), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25, activation=LeakyReLU(alpha=0.2)))  
model.add(Dense(1, activation='linear'))

# Compile the model (you can choose an appropriate optimizer and loss function)
model.compile(optimizer=Adam(learning_rate=0.02), loss=mean_absolute_error,metrics=MeanSquaredError())

# Save the model to an HDF5 file without fitting it
model.save('model.h5')