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
               model_list = ['Prophet'],
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
