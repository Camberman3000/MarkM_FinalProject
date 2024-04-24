#Mark Montenieri
#MS548 - Spring 2024
import matplotlib
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime

end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

stock = "GOOG"
google_data = yf.download(stock, start, end)

# print(google_data.head())# Print first 5 rows
# print(google_data.shape)# Print number of rows and columns
# print(google_data.describe())# Prints info like mean, min, max, etc
# print(google_data.isna().sum())# Finds null rows

plt.figure(figsize=(10, 6))
google_data['Adj Close'].plot()
plt.xlabel('years')
plt.ylabel('Adj Close')
plt.title("Closing Price of Google Stock")

plt.show()# Shows the Google Stock data in graph form