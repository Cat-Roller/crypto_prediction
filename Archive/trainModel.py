import numpy as np
import pandas as pd
import pandas_ta as ta
from keras.models import load_model

# Load the pre-trained model from an HDF5 file
model = load_model('model.h5')  # Replace with the path to your model file

# Load your data for processing (assuming it's in a CSV file)
data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')  # Replace with the path to your data file

# Data Preprocessing
# 1. Drop first column
data = data.iloc[:, 1:]

# 2. Reverse columns order
data = data.iloc[:, ::-1]

# 4. Add target value column (price change in percents for the next five hours)
data['target'] = data['close'].pct_change() * 100

# 5. Drop rows with empty values
data.dropna(inplace=True)

# Determine the split point for training (75% of rows)
split_point = int(0.75 * len(data))

# Create arrays to hold input features and target values
X_train = []
y_train = []

# Populate the arrays with training data
for i in range(split_point - 120):
    # Extract a window of 120 rows as input features
    input_features = data.iloc[i:i + 120, :5].values
    # Extract the target values for the next 5 hours
    target_values = data.iloc[i + 120, -1]

    # Reshape the input features to match the model's input shape
    input_features = input_features.reshape(120, 5)

    X_train.append(input_features)
    y_train.append(target_values)

# Convert the arrays to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
print(y_train.shape)
# Train the model in 15 batches and 2 epochs
model.fit(X_train, y_train, batch_size=20)

# After training, you can save the model
model.save('trained_model_4.h5')
