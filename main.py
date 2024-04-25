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
import tkinter as tk
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

stock = "BTC-USD"
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
    plt.title(f"{column_name} of Bitcoin data")
    plt.show()  # Shows the Google Stock data in graph form


def get_price_prediction():
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
    print("30% :", num_rows - 100 - int(
        len(x_data) * 0.7))  # Get the remaining 30% of the data (num rows - Also take into account the first 100 rows we ignored earlier)

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
    print(model.summary())  # Print the summary of the training

    predictions = model.predict(x_test)
    print("0-1 scaled predictions", predictions)  # Print the price prediction (0-1 scaled value)
    inv_predictions = scaler.inverse_transform(predictions)
    print("Dollar value predictions", inv_predictions)

    inv_y_test = scaler.inverse_transform(y_test)
    print("Dollar value predictions - Y Test: ", inv_y_test)

    rmse = np.sqrt(np.mean((inv_predictions - inv_y_test) ** 2))
    print("RMSE: ", rmse)  # Root-mean-square deviation (lower = better)

    # DataFrame, for plotting multiple sets of graph data
    plotting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_predictions.reshape(-1)
        },
        index=google_data.index[splitting_len + 100:]
    )
    #  print(plotting_data.head())  # Get the first 5 rows, showing both the original values and the predictions
    print(plotting_data.tail())  # Get the last 5 rows, showing both the original values and the predictions
    plot_graph((15, 6), plotting_data, 'test data')

    # Plot a (concatenated) graph using both the plotting_data DataFrame and the Adjusted close value plot
    plot_graph((15, 6), pd.concat([Adj_close_price[:splitting_len + 100], plotting_data], axis=0), 'whole data')

    model.save("Latest_stock_price_model.keras")
    prediction_str.set(plotting_data.tail(0))


# plot function is created for
# plotting the graph in
# tkinter window
def plot():
    # the figure that will contain the plot
    fig = Figure(figsize=(5, 5),
                 dpi=100)

    # list of squares
    y = [i ** 2 for i in range(101)]

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(y)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master=window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().grid(row=5, column=1, columnspan=3)()


window = tk.Tk()
window.title("Bitcoin Price Prediction")
window.geometry('700x700')

"""
centers a tkinter window
:param window: the main window or Toplevel window to center
"""
window.update_idletasks()
width = window.winfo_width()
frm_width = window.winfo_rootx() - window.winfo_x()
win_width = width + 2 * frm_width
height = window.winfo_height()
titlebar_height = window.winfo_rooty() - window.winfo_y()
win_height = height + titlebar_height + frm_width
x = window.winfo_screenwidth() // 2 - win_width // 2
y = window.winfo_screenheight() // 2 - win_height // 2
window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
window.deiconify()

calculate_button = tk.Button(window, text="Get Prediction", command=get_price_prediction)
calculate_button.grid(columnspan=1, row=1, padx=5, pady=5)

prediction_str = tk.StringVar()
prediction_str.set('')
prediction_label = tk.Label(window, text="Predicted price")  # Accuracy label
prediction_label.grid(column=0, row=2, padx=5, pady=5)
prediction_box = tk.Entry(window)  # Accuracy textbox
# entry.grid(column=1, row=4, padx=5, pady=5)
prediction_box = Entry(textvariable=prediction_str, state=DISABLED).grid(column=1, row=2, padx=5, pady=5)

# button that displays the plot
plot_button = tk.Button(master=window, command=plot, width=10, text="Plot")
plot_button.grid(column=2, columnspan=1, row=1, padx=5, pady=5)

window.mainloop()
