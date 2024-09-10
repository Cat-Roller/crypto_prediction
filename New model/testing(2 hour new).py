from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the trained model

model = load_model('XPR_0.0080.h5')

# Load and preprocess the data
data = pd.read_csv('xrp_usdt_price_3_months_1h_candles.csv')
data['fast_ma'] = data['close'].rolling(window=12).mean()
data['slow_ma'] = data['close'].rolling(window=25).mean()
data['macro-trend'] = np.where(data['fast_ma'] > data['slow_ma'], 1, np.where(data['fast_ma'] < data['slow_ma'], -1, 0))
data['upper_band'] = data['close'].rolling(20).mean() + (data['close'].rolling(20).std() * 2) # Calculate the upper band
data['lower_band'] = data['close'].rolling(20).mean() - (data['close'].rolling(20).std() * 2) 
data = data.iloc[26:, :]
data = data.iloc[:, 1:].values

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

sequence_length = 13

# Prepare sequences and predictions for the first predicted hour
sequences = []
next_values = []
for i in range(len(standardized_data) - sequence_length - 1):
    sequences.append(standardized_data[i:i + sequence_length])
    next_values.append(standardized_data[i + sequence_length:i+sequence_length+2, 3]) 

# Convert sequences and predictions to numpy arrays
sequences = np.array(sequences)
next_values = np.array(next_values)

# Make predictions on the standardized data for both hours
predictions_2_hours = model.predict(sequences)

# Separate predictions for the first and second hour
predictions_1st_hour = predictions_2_hours[:, 0]
predictions_2nd_hour = predictions_2_hours[:, 1]

next_values_1st_hour = next_values[:, 0] 
next_values_2nd_hour = next_values[:, 1] 

# Calculate Mean Absolute Error (MAE) on predictions for both hours
mae_model_evaluation = model.evaluate(sequences, next_values, verbose=0)

print(f"Model Evaluation MAE: {mae_model_evaluation[0]}")

# De-standardize predictions for both hours
destandardized_predictions_1st_hour = predictions_1st_hour * std[:][-2] + mean[:][-2]
destandardized_predictions_2nd_hour = predictions_2nd_hour * std[:][-2] + mean[:][-2]

# Calculate MAE manually for the first hour
mae_manual_1st_hour = np.mean(np.abs(predictions_1st_hour.squeeze() - next_values_1st_hour))
print(f"Manual Calculation MAE (1st hour): {mae_manual_1st_hour}")

# Calculate Mean Absolute Error (MAE) on destandardized predictions and original data for the first hour
mae_after_destd_1st_hour = np.mean(np.abs((destandardized_predictions_1st_hour.squeeze()) - data[sequence_length :-1, 3]))
print(f"MAE on De-standardized Predictions and Original Data (1st hour): {mae_after_destd_1st_hour}\n")

# Calculate MAE manually for the second hour
mae_manual_2nd_hour = np.mean(np.abs(predictions_2nd_hour.squeeze() - next_values_2nd_hour))
print(f"Manual Calculation MAE (2nd hour): {mae_manual_2nd_hour}")

# Calculate Mean Absolute Error (MAE) on destandardized predictions and original data for the second hour
mae_after_destd_2nd_hour = np.mean(np.abs((destandardized_predictions_2nd_hour.squeeze()) - data[sequence_length + 1:, 3]))
print(f"MAE on De-standardized Predictions and Original Data (2nd hour): {mae_after_destd_2nd_hour}\n")

# Calculate accuracy for price movement direction prediction for the first hour
actual_price_changes_1st_hour = np.sign(data[sequence_length + 1:, 3] - data[sequence_length :-1, 3])
predicted_price_changes_1st_hour = np.sign(
    np.squeeze(destandardized_predictions_1st_hour) - data[sequence_length :-1, 3]
)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Calculate accuracy for price movement direction prediction for the second hour
actual_price_changes_2nd_hour = np.sign(data[sequence_length + 2:, 3] - data[sequence_length + 1:-1, 3])
predicted_price_changes_2nd_hour = np.sign(
    np.squeeze(destandardized_predictions_2nd_hour) - data[sequence_length + 1:, 3]
)

# Calculate true positive (TP) and true negative (TN) for the first hour
tp_1st_hour = np.sum((actual_price_changes_1st_hour == 1) & (predicted_price_changes_1st_hour == 1))
tn_1st_hour = np.sum((actual_price_changes_1st_hour == -1) & (predicted_price_changes_1st_hour == -1))
fp_1st_hour = np.sum((actual_price_changes_1st_hour == -1) & (predicted_price_changes_1st_hour == 1))
fn_1st_hour = np.sum((actual_price_changes_1st_hour == 1) & (predicted_price_changes_1st_hour == -1))

# Calculate total instances for the first hour
total_instances_1st_hour = len(actual_price_changes_1st_hour)

# Calculate accuracy for the first hour
accuracy_1st_hour = np.mean(actual_price_changes_1st_hour == predicted_price_changes_1st_hour)
print(f"Accuracy of Price Movement Direction Prediction (1st hour): {accuracy_1st_hour}")

# Calculate accuracy for top 5% most volatile price changes for the first hour
absolute_price_changes_1st_hour = np.abs(data[sequence_length + 1:, 3] - data[sequence_length :-1, 3])

top_5_percent_threshold_1st_hour = np.percentile(absolute_price_changes_1st_hour, 95)
top_5_percent_indices_1st_hour = np.where(absolute_price_changes_1st_hour >= top_5_percent_threshold_1st_hour)[0]

# Calculate accuracy for middle 60% of volatility for the first hour
middle_60_percent_threshold_low_1st_hour = np.percentile(absolute_price_changes_1st_hour, 30)
middle_60_percent_threshold_high_1st_hour = np.percentile(absolute_price_changes_1st_hour, 90)
middle_60_percent_indices_1st_hour = np.where(
    (absolute_price_changes_1st_hour >= middle_60_percent_threshold_low_1st_hour) &
    (absolute_price_changes_1st_hour <= middle_60_percent_threshold_high_1st_hour)
)[0]

bottom_40_percent_threshold_1st_hour = np.percentile(absolute_price_changes_1st_hour, 40)
bottom_40_percent_indices_1st_hour = np.where(absolute_price_changes_1st_hour <= bottom_40_percent_threshold_1st_hour)[0]

accuracy_bottom_40_percent_1st_hour = np.mean(
    actual_price_changes_1st_hour[bottom_40_percent_indices_1st_hour] ==
    predicted_price_changes_1st_hour[bottom_40_percent_indices_1st_hour]
)

accuracy_middle_60_percent_1st_hour = np.mean(
    actual_price_changes_1st_hour[middle_60_percent_indices_1st_hour] ==
    predicted_price_changes_1st_hour[middle_60_percent_indices_1st_hour]
)

accuracy_top_5_percent_1st_hour = np.mean(
    actual_price_changes_1st_hour[top_5_percent_indices_1st_hour] ==
    predicted_price_changes_1st_hour[top_5_percent_indices_1st_hour]
)

print(f"Accuracy for Bottom 40% of Volatility (1st hour): {accuracy_bottom_40_percent_1st_hour}")
print(f"Accuracy for Middle 60% of Volatility (1st hour): {accuracy_middle_60_percent_1st_hour}")
print(f"Accuracy for Top 5% Most Volatile Price Changes (1st hour): {accuracy_top_5_percent_1st_hour}\n")

print(f"True Positive (TP) for the 1st hour: {tp_1st_hour} ({(tp_1st_hour / total_instances_1st_hour) * 100:.2f}%)")
print(f"True Negative (TN) for the 1st hour: {tn_1st_hour} ({(tn_1st_hour / total_instances_1st_hour) * 100:.2f}%)")
print(f"False Positive (FP) for the 1st hour: {fp_1st_hour} ({(fp_1st_hour / total_instances_1st_hour) * 100:.2f}%)")
print(f"False Negative (FN) for the 1st hour: {fn_1st_hour} ({(fn_1st_hour / total_instances_1st_hour) * 100:.2f}%)\n")

print(f"Positive winrate for the 1st hour: {tp_1st_hour / (fp_1st_hour + tp_1st_hour) * 100:.2f}%")
print(f"Negative winrate for the 1st hour: {tn_1st_hour / (fn_1st_hour + tn_1st_hour) * 100:.2f}%\n")

# Calculate accuracy for price movement direction prediction for the second hour
actual_price_changes_2nd_hour = np.sign(data[sequence_length + 2 :, 3] - data[sequence_length + 1 : -1, 3])
predicted_price_changes_2nd_hour = np.sign(
    np.squeeze(destandardized_predictions_2nd_hour[:-1]) - data[sequence_length + 1  :-1, 3]
)

# Calculate true positive (TP) and true negative (TN) for the second hour
tp_2nd_hour = np.sum((actual_price_changes_2nd_hour == 1) & (predicted_price_changes_2nd_hour == 1))
tn_2nd_hour = np.sum((actual_price_changes_2nd_hour == -1) & (predicted_price_changes_2nd_hour == -1))
fp_2nd_hour = np.sum((actual_price_changes_2nd_hour == -1) & (predicted_price_changes_2nd_hour == 1))
fn_2nd_hour = np.sum((actual_price_changes_2nd_hour == 1) & (predicted_price_changes_2nd_hour == -1))

# Calculate total instances for the second hour
total_instances_2nd_hour = len(actual_price_changes_2nd_hour)

# Calculate accuracy for the second hour
accuracy_2nd_hour = np.mean(actual_price_changes_2nd_hour == predicted_price_changes_2nd_hour)
print(f"Accuracy of Price Movement Direction Prediction (2nd hour): {accuracy_2nd_hour}")

# Calculate accuracy for top 5% most volatile price changes for the second hour
absolute_price_changes_2nd_hour = np.abs(data[sequence_length + 2 : , 3] - data[sequence_length + 1 : -1, 3])
top_5_percent_threshold_2nd_hour = np.percentile(absolute_price_changes_2nd_hour, 95)
top_5_percent_indices_2nd_hour = np.where(absolute_price_changes_2nd_hour >= top_5_percent_threshold_2nd_hour)[0]

# Calculate accuracy for middle 60% of volatility for the second hour
middle_60_percent_threshold_low_2nd_hour = np.percentile(absolute_price_changes_2nd_hour, 30)
middle_60_percent_threshold_high_2nd_hour = np.percentile(absolute_price_changes_2nd_hour, 90)
middle_60_percent_indices_2nd_hour = np.where(
    (absolute_price_changes_2nd_hour >= middle_60_percent_threshold_low_2nd_hour) &
    (absolute_price_changes_2nd_hour <= middle_60_percent_threshold_high_2nd_hour)
)[0]

bottom_40_percent_threshold_2nd_hour = np.percentile(absolute_price_changes_2nd_hour, 40)
bottom_40_percent_indices_2nd_hour = np.where(absolute_price_changes_2nd_hour <= bottom_40_percent_threshold_2nd_hour)[0]

accuracy_bottom_40_percent_2nd_hour = np.mean(
    actual_price_changes_2nd_hour[bottom_40_percent_indices_2nd_hour] ==
    predicted_price_changes_2nd_hour[bottom_40_percent_indices_2nd_hour]
)

accuracy_middle_60_percent_2nd_hour = np.mean(
    actual_price_changes_2nd_hour[middle_60_percent_indices_2nd_hour] ==
    predicted_price_changes_2nd_hour[middle_60_percent_indices_2nd_hour]
)

accuracy_top_5_percent_2nd_hour = np.mean(
    actual_price_changes_2nd_hour[top_5_percent_indices_2nd_hour] ==
    predicted_price_changes_2nd_hour[top_5_percent_indices_2nd_hour]
)

print(f"Accuracy for Bottom 40% of Volatility (2nd hour): {accuracy_bottom_40_percent_2nd_hour}")
print(f"Accuracy for Middle 60% of Volatility (2nd hour): {accuracy_middle_60_percent_2nd_hour}")
print(f"Accuracy for Top 5% Most Volatile Price Changes (2nd hour): {accuracy_top_5_percent_2nd_hour}\n")

print(f"True Positive (TP) for the 2nd hour: {tp_2nd_hour} ({(tp_2nd_hour / total_instances_2nd_hour) * 100:.2f}%)")
print(f"True Negative (TN) for the 2nd hour: {tn_2nd_hour} ({(tn_2nd_hour / total_instances_2nd_hour) * 100:.2f}%)")
print(f"False Positive (FP) for the 2nd hour: {fp_2nd_hour} ({(fp_2nd_hour / total_instances_2nd_hour) * 100:.2f}%)")
print(f"False Negative (FN) for the 2nd hour: {fn_2nd_hour} ({(fn_2nd_hour / total_instances_2nd_hour) * 100:.2f}%)\n")

print(f"Positive winrate for the 2nd hour: {tp_2nd_hour / (fp_2nd_hour + tp_2nd_hour) * 100:.2f}%")
print(f"Negative winrate for the 2nd hour: {tn_2nd_hour / (fn_2nd_hour + tn_2nd_hour) * 100:.2f}%")

# Create an array representing percentages from 1 to 100
percentages = np.arange(1, 101)

# Initialize an array to store accuracies for each percentage for the first hour
accuracies_1st_hour = []

# Calculate accuracy for each percentage for the first hour
for percent in percentages:
    threshold = np.percentile(absolute_price_changes_1st_hour, percent)
    indices = np.where(absolute_price_changes_1st_hour >= threshold)[0]
    
    # Calculate accuracy for the current percentage for the first hour
    accuracy = np.mean(actual_price_changes_1st_hour[indices] == predicted_price_changes_1st_hour[indices])
    
    # Append the accuracy to the accuracies array
    accuracies_1st_hour.append(accuracy)

# Initialize an array to store accuracies for each percentage for the second hour
accuracies_2nd_hour = []

# Calculate accuracy for each percentage for the second hour
for percent in percentages:
    threshold = np.percentile(absolute_price_changes_2nd_hour, percent)
    indices = np.where(absolute_price_changes_2nd_hour >= threshold)[0]
    
    # Calculate accuracy for the current percentage for the second hour
    accuracy = np.mean(actual_price_changes_2nd_hour[indices] == predicted_price_changes_2nd_hour[indices])
    
    # Append the accuracy to the accuracies array
    accuracies_2nd_hour.append(accuracy)

# Plotting de-standardized predictions, actual close prices, and predicted close prices for both hours
plt.figure(figsize=(10, 6))

# Plotting actual close prices
plt.plot(data[sequence_length + 2:-1, 3], label='Actual close price', alpha=0.7)

# Plotting de-standardized predictions for the first hour
plt.plot(destandardized_predictions_1st_hour, label='Predictions for 1st hour', alpha=0.7)

# Plotting de-standardized predictions for the second hour
plt.plot(destandardized_predictions_2nd_hour, label='Predictions for 2nd hour', alpha=0.7)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual Close Price vs Predictions for Both Hours')
plt.legend()
plt.show()

# Plot the graph for both hours
plt.figure(figsize=(10, 6))
plt.plot(percentages, accuracies_1st_hour, marker='o', label='1st hour')
plt.plot(percentages, accuracies_2nd_hour, marker='o', label='2nd hour')
plt.axhline(y=0, color='black', linestyle='--', label='y=0')
plt.axhline(y=0.5, color='red', linestyle='--', label='y=0.5')
plt.xlabel('Percentage of Volatility')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Percentage of Volatility for Both Hours')
plt.grid(True)
plt.legend()
plt.show()