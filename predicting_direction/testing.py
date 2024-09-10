from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the trained model
model = load_model('direction_v2.h5')

# Load and preprocess the data
data = pd.read_csv('new.csv')
data = data.iloc[:, 1:]
data['fast_ma'] = data['close'].rolling(window=12).mean()
data['slow_ma'] = data['close'].rolling(window=25).mean()
data['change'] = data['close'].shift(-1) - data['close']
data = data.iloc[26:-1, :]
print(data.head())
data = data.values

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

# Add the new column to the standardized_array
sequence_length = 13

# Prepare sequences and predictions for the first predicted hour
sequences = []
next_values = []
actual_next_values = []

for i in range(len(standardized_data) - sequence_length - 2):
    sequences.append(standardized_data[i:i + sequence_length, :-1])
    next_values.append([standardized_data[i,-1],standardized_data[i+1, -1]]) 
    actual_next_values.append([data[i,-1],data[i+1,-1]])

# Convert sequences and predictions to numpy arrays
sequences = np.array(sequences)
next_values = np.array(next_values)
actual_next_values = np.array(actual_next_values)


# Make predictions on the standardized data for both hours
predictions_2_hours = model.predict(sequences)

print(predictions_2_hours[-20:,:])
print(actual_next_values[-20:,:])
# Separate predictions for the first and second hour
predictions_1st_hour = predictions_2_hours[:, 0]
predictions_2nd_hour = predictions_2_hours[:, 1]

next_values_1st_hour = next_values[:, 0] 
next_values_2nd_hour = next_values[:, 1] 

# Calculate Mean Absolute Error (MAE) on predictions for both hours
mae_model_evaluation = model.evaluate(sequences, next_values, verbose=0)

print(f"Model Evaluation MAE: {mae_model_evaluation}")

# De-standardize predictions for both hours
destandardized_predictions_1st_hour = predictions_1st_hour * std[:][-1] + mean[:][-1]
destandardized_predictions_2nd_hour = predictions_2nd_hour * std[:][-1] + mean[:][-1]

# Calculate MAE manually for the first hour
mae_manual_1st_hour = np.mean(np.abs(predictions_1st_hour.squeeze() - next_values_1st_hour))
print(f"Manual Calculation MAE (1st hour): {mae_manual_1st_hour}")
actual_next_values=np.array(actual_next_values)
# Calculate Mean Absolute Error (MAE) on destandardized predictions and original data for the first hour
mae_after_destd_1st_hour = np.mean(np.abs(destandardized_predictions_1st_hour.squeeze() - actual_next_values[:,0]))
print(f"MAE on De-standardized Predictions and Original Data (1st hour): {mae_after_destd_1st_hour}\n")

# Calculate MAE manually for the second hour
mae_manual_2nd_hour = np.mean(np.abs(predictions_2nd_hour.squeeze() - next_values_2nd_hour))
print(f"Manual Calculation MAE (2nd hour): {mae_manual_2nd_hour}")

# Calculate Mean Absolute Error (MAE) on destandardized predictions and original data for the second hour
mae_after_destd_2nd_hour = np.mean(np.abs((destandardized_predictions_2nd_hour.squeeze()) - actual_next_values[:,1]))
print(f"MAE on De-standardized Predictions and Original Data (2nd hour): {mae_after_destd_2nd_hour}\n")

# Calculate accuracy for price movement direction prediction for the first hour
actual_price_changes_1st_hour = np.sign(actual_next_values[:,0])
predicted_price_changes_1st_hour = np.sign(np.squeeze(destandardized_predictions_1st_hour))

# Calculate accuracy for price movement direction prediction for the second hour
actual_price_changes_2nd_hour = np.sign(actual_next_values[:,1])
predicted_price_changes_2nd_hour = np.sign(np.squeeze(destandardized_predictions_2nd_hour))

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
absolute_price_changes_1st_hour = np.abs(actual_next_values[:,0])


print(f"True Positive (TP) for the 1st hour: {tp_1st_hour} ({(tp_1st_hour / total_instances_1st_hour) * 100:.2f}%)")
print(f"True Negative (TN) for the 1st hour: {tn_1st_hour} ({(tn_1st_hour / total_instances_1st_hour) * 100:.2f}%)")
print(f"False Positive (FP) for the 1st hour: {fp_1st_hour} ({(fp_1st_hour / total_instances_1st_hour) * 100:.2f}%)")
print(f"False Negative (FN) for the 1st hour: {fn_1st_hour} ({(fn_1st_hour / total_instances_1st_hour) * 100:.2f}%)\n")

print(f"Positive winrate for the 1st hour: {tp_1st_hour / (fp_1st_hour + tp_1st_hour) * 100:.2f}%")
print(f"Negative winrate for the 1st hour: {tn_1st_hour / (fn_1st_hour + tn_1st_hour) * 100:.2f}%\n")

# Calculate accuracy for price movement direction prediction for the second hour
actual_price_changes_2nd_hour = np.sign(actual_next_values[:,1])
predicted_price_changes_2nd_hour = np.sign(np.squeeze(destandardized_predictions_2nd_hour))

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
absolute_price_changes_2nd_hour = np.abs(actual_next_values[:,1])

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
plt.plot(data[sequence_length :, 3], label='Actual close price', alpha=0.7)

# Plotting de-standardized predictions for the first hour
predicted_close_1st_hour=data[sequence_length:-2,3] + destandardized_predictions_1st_hour
plt.plot(predicted_close_1st_hour, label='Predictions for 1st hour', alpha=0.7)

# Plotting de-standardized predictions for the second hour
predicted_close_2nd_hour=data[sequence_length+1:-1,3] + destandardized_predictions_2nd_hour
plt.plot(predicted_close_2nd_hour, label='Predictions for 2nd hour', alpha=0.7)

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