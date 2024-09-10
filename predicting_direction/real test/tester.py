# scheduler.py
import schedule
import time
import csv
import datetime
from predict_next_2_hours import get_vars

def job():
    # Call the function from code.py
    prediction1, prediction2, true_change = get_vars()
    
    # Get the current timestamp
    timestamp = datetime.datetime.now()
    
    # Specify the CSV file path
    csv_file_path = 'variables.csv'
    
    # Write the variables to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, prediction1, prediction2, true_change])

# Schedule the job every hour
schedule.every().hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
