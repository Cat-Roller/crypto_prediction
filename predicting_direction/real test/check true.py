import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file
df = pd.read_csv('variables.csv', header=None)

# Name the columns
df.columns = ['date and time', 'prediction1', 'prediction2', 'true change']

#true and wrong for both hours
h1_true=0
h1_false=0
h2_true=0
h2_false=0

# Function to calculate the time difference in hours, returning only natural numbers
def time_diff_in_hours(t1, t2):
    return int(abs((t1 - t2).total_seconds() / 3600))

# Convert the 'date and time' column to datetime objects
df['date and time'] = pd.to_datetime(df['date and time'])

# Loop through the DataFrame starting from the 4th row
for i in range(3, len(df)):
    # Check if the time difference between the last 4 rows is 1 hour each
    if (time_diff_in_hours(df['date and time'].iloc[i], df['date and time'].iloc[i-1]) == 1 and
        time_diff_in_hours(df['date and time'].iloc[i-1], df['date and time'].iloc[i-2]) == 1 and
        time_diff_in_hours(df['date and time'].iloc[i-2], df['date and time'].iloc[i-3]) == 1):
        
        # Check if the sign of true change matches the sign of prediction1 in -2 row
        if (df['true change'].iloc[i] > 0) == (df['prediction1'].iloc[i-2] > 0):
            result1 = 1
            h1_true += 1
        else:
            result1 = -1
            h1_false += 1
        
        # Check if the sign of true change matches the sign of prediction2 in -3 row
        if (df['true change'].iloc[i] > 0) == (df['prediction2'].iloc[i-3] > 0):
            result2 = 1
            h2_true += 1
        else:
            result2 = -1
            h2_false += 1
            
        # Add the results to the row
        df.loc[i, 'Result1'] = result1
        df.loc[i, 'Result2'] = result2
    else:
        # Add 0's to the row if the time difference condition is not met
        df.loc[i, 'Result1'] = 0
        df.loc[i, 'Result2'] = 0

# Save the modified DataFrame back to a CSV file
print("first hour winrate: ", h1_true/(h1_true+h1_false))
print("second hour winrate: ", h2_true/(h2_true+h2_false))