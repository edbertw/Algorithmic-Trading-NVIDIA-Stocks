import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as p
from autots import AutoTS

current = date.today() #Date Today is June 18 2024
end = current.strftime("%Y-%m-%d")
start = (date.today() - timedelta(days = 365)).strftime("%Y-%m-%d")
stock = yf.download('NVDA', start = start, end = end, progress = False) #NVIDIA Stocks
print(stock.tail())
stock['Date'] = stock.index
select_cols = ["Date","Open","High","Low","Close","Adj Close","Volume"]
stock = stock[select_cols]
stock.reset_index(drop = True, inplace = True)
print(stock.head())
print(stock.tail())

stock = stock[["Date","Close"]]
plt.figure(figsize = (16,10))
sns.lineplot(x = stock.Date, y = stock.Close)
sns.set_style("whitegrid")
sns.set_palette("deep")
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.xticks(rotation = 45)
plt.title("Time Series Plot of Closing Prices")

model = AutoTS(forecast_length = 35, #Forecast until end of July 2024
               model_list = ['Prophet', 'ARIMA'],
               frequency = 'infer',
               ensemble = 'simple')
model = model.fit(stock, date_col = 'Date', value_col = 'Close', id_col = None)
predictions = model.predict().forecast
print(predictions)

predictions = pd.DataFrame(predictions)
predictions = predictions.reset_index()
predictions = predictions.rename(columns = {'index': 'Date'})
predictions.head()

preds = predictions.copy()
preds.head()
stock['momentum'] = stock['Close'].pct_change()
preds['momentum'] = preds['Close'].pct_change()
full_vals = pd.concat([stock,preds], ignore_index = True)

full_vals['MA10'] = full_vals['Close'].rolling(window = 10).mean()
full_vals['MA20'] = full_vals['Close'].rolling(window = 20).mean()
full_vals['Signal'] = 0
full_vals['Signal'][10:] = full_vals['MA10'][10:] > full_vals['MA20'][10:]
full_vals['Pos'] = full_vals['Signal'].diff()
full_vals.head()

# Momentum Strategy
fig = make_subplots(rows = 2, cols = 1)
fig.add_trace(go.Scatter(x = full_vals.Date,
                        y = full_vals.Close,
                        name = 'Closing Price'))
fig.add_trace(go.Scatter(x = full_vals.Date,
                        y = full_vals.momentum,
                        name = "Momentum",
                        yaxis = 'y2'))
fig.add_trace(go.Scatter(x = full_vals.loc[full_vals['momentum'] > 0].Date,
                        y = full_vals.loc[full_vals['momentum'] > 0].Close,
                        mode = 'markers',
                        name = 'BUY',
                        marker = dict(color = 'green', symbol = 'triangle-up')))
fig.add_trace(go.Scatter(x = full_vals.loc[full_vals['momentum'] < 0].Date,
                        y = full_vals.loc[full_vals['momentum'] < 0].Close,
                        mode = 'markers',
                        name = 'SELL',
                        marker = dict(color = 'red', symbol = 'triangle-down')))
fig.update_layout(title = 'NVIDIA Predicted Stocks',
                 xaxis_title = 'Date',
                 yaxis_title = 'Close',
                 yaxis2_title = "Momentum")
fig.update_yaxes(secondary_y = True)
fig.show()

# Moving Average Strategy
plt.figure(figsize = (12,8))
plt.plot(full_vals.Date, full_vals.Close, label = "Closing Price")
plt.plot(full_vals.Date, full_vals.MA10, label = "10 day MA")
plt.plot(full_vals.Date, full_vals.MA20, label = "20 day MA")

# Add BUY/SELL SIGNALS
plt.plot(full_vals[full_vals['Pos'] == 1].Date, full_vals['MA10'][full_vals['Pos'] == 1], '^', markersize = 10, color ='g', label = 'BUY')
plt.plot(full_vals[full_vals['Pos'] == -1].Date, full_vals['MA10'][full_vals['Pos'] == -1], 'v', markersize = 10, color = 'r', label = "SELL")

plt.xlabel("Date")
plt.ylabel("Close")
plt.title("SMA Strategy for NVIDIA Stock Prices")
plt.legend()
plt.show()

#SMA STRATEGY ACCURACY
full_vals["Returns"] = full_vals["momentum"]
full_vals["StratReturns"] = full_vals["Returns"] * full_vals["Pos"].shift(1)
SMA_accuracy = (full_vals["Pos"].diff() == full_vals['Returns'].shift(-1)).mean()
print(SMA_accuracy)
