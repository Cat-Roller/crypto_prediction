import numpy as np
import pandas as pd
import pandas_ta as ta
from keras.models import load_model

# Load the pre-trained model
model = load_model('trained_model_4.h5')

# Load your data for testing (assuming it's in a CSV file)
data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')

# Data Preprocessing 
data = data.iloc[:, 1:]
data = data.iloc[:, ::-1]
data['target'] = data['close'].pct_change(periods=1).shift(-1) * 100
data.dropna(inplace=True)
split = int(0.75 * len(data))
testing_data = data.iloc[split:, :]

X_test = []
y_test = []

# Metrics
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# Iterate through the testing data
for i in range(len(testing_data) - 125):
    # Extract a window of 120 rows as input features
    input_features = testing_data.iloc[i:i + 120, :5].values
    # Extract the target values from the 120th row
    target_values = testing_data.iloc[i + 120, -1]
    # Reshape the input features to match the model's input shape
    input_features = input_features.reshape(1, 120, 5)

    # Predict using the model
    predictions = model.predict(input_features)[0]  # Get the array of predictions

    # Define actual and predicted values
    actual = target_values
    predicted = predictions

    threshold = 0.0
    # Apply the threshold to determine the predicted class (positive or negative)
    predicted_class = 1 if predicted > threshold else 0
    actual_class = 1 if actual > 0 else 0

    if actual_class == 1 and predicted_class == 1:
        true_positives += 1
    elif actual_class == 0 and predicted_class == 0:
        true_negatives += 1
    elif actual_class == 0 and predicted_class == 1:
        false_positives += 1
    elif actual_class == 1 and predicted_class == 0:
        false_negatives += 1

# Calculate and print the metrics
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print(f"True Positives: {true_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
