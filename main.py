#Mark Montenieri
#MS548 - Spring 2024
import matplotlib
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

stock = "GOOG"
google_data = yf.download(stock, start, end)

# print(google_data.head())# Print first 5 rows
# print(google_data.shape)# Print number of rows and columns
# print(google_data.describe())# Prints info like mean, min, max, etc
# print(google_data.isna().sum())# Finds null rows


def plot_graph(figsize, values, column_name):
    plt.figure()
    values.plot(figsize=figsize)
    plt.xlabel("years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of Google data")

    plt.show()# Shows the Google Stock data in graph form

# print(google_data.columns)# Prints column names

# Loops through the data in each column and prints out a graph for each
#for column in google_data.columns:
    #plot_graph((15, 5), google_data[column], column)

# Prints the Moving Average for the number of days specified
#temp_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#print(sum(temp_data[1:6])/5)

# Another movine average example
#data = pd.DataFrame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#print(data.head())
#print(data.rolling(5).mean())
#data['MA'] = data.rolling(3).mean()
#print(data)

# Prints the number of trading days per year (i.e. not weekends) for the range indicated
#for i in range(2004, 2025):
    #print(list(google_data.index.year).count(i))

# Displays mean data for Adj. Close price
#google_data['MA_for_250_days'] = google_data['Adj Close'].rolling(window=250).mean()
#google_data['MA_for_100_days'] = google_data['Adj Close'].rolling(window=100).mean()
#print(google_data['MA_for_250_days'])
#print(google_data['MA_for_250_days'][0:250].tail())# Last row only

# graph for moving average of 250 days of data
#plot_graph((15, 5), google_data['MA_for_250_days'], 'MA_for_250_days')

# graph showing both Adj. Close and Moving Avg for 100 and 250 days
#plot_graph((15, 5), google_data[['Adj Close', 'MA_for_100_days', 'MA_for_250_days']], 'MA')

# Print the day-to-day percentage change for the first 5 rows
#google_data['percentage_change_cp'] = google_data['Adj Close'].pct_change()
#print(google_data[['Adj Close', 'percentage_change_cp']].head())

# Plot the daily change in percentage of the Adjusted close price
#plot_graph((15, 5), google_data['percentage_change_cp'], 'percentage_change')

# Print the min and max Adj Close price
Adj_close_price = google_data[['Adj Close']]
#print("Max and Min values: ", max(Adj_close_price.values), min(Adj_close_price.values))

# Scales the min and max values from original values to a value between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(Adj_close_price)
#print(scaled_data)
num_rows = len(scaled_data)
print("Length of Scaled Data: ", num_rows)  # Number of rows

x_data = []
y_data = []

for i in range(100, len(scaled_data)):  # Start at row 100
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)
#print(x_data[0], "\n Predicted value result based on 100 x values: ", y_data[0])

print("70% :", int(len(x_data) * 0.7))  # Get 70% of the data (num rows)
print("30% :", num_rows - 100 - int(len(x_data) * 0.7))  # Get the remaining 30% of the data (num rows - Also take into account the first 100 rows we ignored earlier)

splitting_len = int(len(x_data) * 0.7)
# Training Data
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

# Testing Data
x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # 128 neurons
model.add(LSTM(64, return_sequences=False))  # 64 neurons
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1)
print(model.summary())






