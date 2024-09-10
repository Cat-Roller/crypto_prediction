from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the trained model
model = load_model('CBA_seq_length_13_loss_0.0046.h5')

# Load and preprocess the data
data = pd.read_csv('btc_usdt_price_3_months_1h_candles.csv')
data = data.iloc[:, 1:].values

reference_data = pd.read_csv('btc_usdt_price_3_years_1h_candles.csv')
reference_data = reference_data.iloc[:, 1:].values

concatenated_data = np.concatenate((data, reference_data), axis=0)
print(concatenated_data.shape)

# Perform standardization using mean and std from the concatenated data
mean = np.mean(concatenated_data, axis=0)
std = np.std(concatenated_data, axis=0)
standardized_data_concat = (concatenated_data - mean) / std

# Separate the datasets back
standardized_data_main = standardized_data_concat[:len(data)]
standardized_data_reference = standardized_data_concat[len(data):]

# toggle between different data
# standardized_data=standardized_data_main

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# data=reference_data
# standardized_data=standardized_data_reference

# mean=np.mean(reference_data, axis=0)
# std=np.std(reference_data, axis=0)
# data=reference_data
# standardized_data=(reference_data - mean) / std

# data=concatenated_data
# standardized_data=standardized_data_concat

sequence_length = 13

# Prepare sequences and predictions for the first predicted hour
sequences = []
next_values = []
for i in range(len(standardized_data) - sequence_length - 1):
    sequences.append(standardized_data[i:i + sequence_length])
    next_values.append(standardized_data[i + sequence_length, -2])  # Predict next hour

# Convert sequences and predictions to numpy arrays
sequences = np.array(sequences)
next_values = np.array(next_values)

# Make predictions on the standardized data
predictions = model.predict(sequences)

# Calculate Mean Absolute Error (MAE) on predictions
mae_model_evaluation = model.evaluate(sequences, next_values, verbose=0)
print(f"Model Evaluation MAE: {mae_model_evaluation[0]}")

# Calculate MAE manually
mae_manual = np.mean(np.abs(predictions.squeeze() - next_values))
print(f"Manual Calculation MAE: {mae_manual}")

# De-standardize predictions
destandardized_predictions = predictions * std[:][-2] + mean[:][-2]

# Calculate Mean Absolute Error (MAE) on destandardized predictions and original data
mae_after_destd = np.mean(np.abs((destandardized_predictions.squeeze()) - data[sequence_length : -1, -2]))
print(f"MAE on De-standardized Predictions and Original Data: {mae_after_destd}\n")

# Calculate accuracy for price movement direction prediction
actual_price_changes = np.sign(data[sequence_length + 1:, -2] - data[sequence_length:-1, -2])
predicted_price_changes = np.sign(np.squeeze(destandardized_predictions) - data[sequence_length:-1, -2])

# Calculate true positive (TP) and true negative (TN)
tp = np.sum((actual_price_changes == 1) & (predicted_price_changes == 1))
tn = np.sum((actual_price_changes == -1) & (predicted_price_changes == -1))
fp = np.sum((actual_price_changes == -1) & (predicted_price_changes == 1))
fn = np.sum((actual_price_changes == 1) & (predicted_price_changes == -1))

# Calculate total instances
total_instances = len(actual_price_changes)

# Calculate accuracy
accuracy = np.mean(actual_price_changes == predicted_price_changes)
print(f"Accuracy of Price Movement Direction Prediction: {accuracy}")

# Calculate accuracy for top 5% most volatile price changes
absolute_price_changes = np.abs(data[sequence_length + 1:, -2] - data[sequence_length:-1, -2])

top_5_percent_threshold = np.percentile(absolute_price_changes, 80)
top_5_percent_indices = np.where(absolute_price_changes >= top_5_percent_threshold)[0]

# Calculate accuracy for middle 60% of volatility
middle_60_percent_threshold_low = np.percentile(absolute_price_changes, 40)
middle_60_percent_threshold_high = np.percentile(absolute_price_changes, 80)
middle_60_percent_indices = np.where(
    (absolute_price_changes >= middle_60_percent_threshold_low) &
    (absolute_price_changes <= middle_60_percent_threshold_high)
)[0]

bottom_40_percent_threshold = np.percentile(absolute_price_changes, 40)
bottom_40_percent_indices = np.where(absolute_price_changes <= bottom_40_percent_threshold)[0]

accuracy_bottom_40_percent = np.mean(
    actual_price_changes[bottom_40_percent_indices] == predicted_price_changes[bottom_40_percent_indices]
)

accuracy_middle_60_percent = np.mean(
    actual_price_changes[middle_60_percent_indices] == predicted_price_changes[middle_60_percent_indices]
)

accuracy_top_5_percent = np.mean(actual_price_changes[top_5_percent_indices] == predicted_price_changes[top_5_percent_indices])

print(f"Accuracy for Bottom 40% of Volatility: {accuracy_bottom_40_percent}")
print(f"Accuracy for Middle 60% of Volatility: {accuracy_middle_60_percent}")
print(f"Accuracy for Top 5% Most Volatile Price Changes: {accuracy_top_5_percent}\n",)

print(f"True Positive (TP): {tp} ({(tp / total_instances) * 100:.2f}%)")
print(f"True Negative (TN): {tn} ({(tn / total_instances) * 100:.2f}%)")
print(f"False Positive (FP): {fp} ({(fp / total_instances) * 100:.2f}%)")
print(f"False Negative (FN): {fn} ({(fn / total_instances) * 100:.2f}%)\n")

print(f"Positive winrate: {tp / (fp + tp) * 100:.2f}%")
print(f"Negative winrate: {tn / (fn + tn) * 100:.2f}%\n")

# Calculate Mean Absolute Price Change in the original data
df = pd.DataFrame(data)
absolute_price_change_last_column = np.mean(np.abs(df.iloc[:, -2].diff().fillna(0)))
print(f"Mean Absolute Price Change in the Last Column of Data: {absolute_price_change_last_column}")

# Plotting de-standardized predictions and -2nd column of the data
plt.figure(figsize=(10, 6))

# Plotting de-standardized predictions
plt.plot(destandardized_predictions, label='Predictions for next hour', alpha=0.7)

# Plotting -2nd column of the data
plt.plot(data[sequence_length + 1:, -2], label='actual close price', alpha=0.7)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Predictions vs actual close price')
plt.legend()
plt.show()

# Create an array representing percentages from 1 to 100
percentages = np.arange(1, 101)

# Initialize an array to store accuracies for each percentage
accuracies = []

# Calculate accuracy for each percentage
for percent in percentages:
    threshold = np.percentile(absolute_price_changes, percent)
    indices = np.where(absolute_price_changes >= threshold)[0]

    # Calculate accuracy for the current percentage
    accuracy = np.mean(actual_price_changes[indices] == predicted_price_changes[indices])

    # Append the accuracy to the accuracies array
    accuracies.append(accuracy)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(percentages, accuracies, marker='o')
plt.axhline(y=0, color='black', linestyle='--', label='y=0')
plt.axhline(y=0.5, color='red', linestyle='--', label='y=0.5')
plt.xlabel('Percentage of Volatility')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Percentage of Volatility')
plt.grid(True)
plt.show()
