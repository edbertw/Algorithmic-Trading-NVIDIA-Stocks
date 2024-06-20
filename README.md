# Predicting Future NVIDIA Stock Prices and Making Trading Decisions
## Overview
This time series project aims to utilize machine learning algorithms to predict real-time NVIDIA stocks 35 days from now (Until Early August 2024). I implemented the powerful autoTS time series library limited to the Prophet by Facebook ML model to accurately predict future stock prices. Furthermore, I conducted an analysis of past, current and future stock closing prices to construct a "Momentum" and "Moving Average" Strategy for Algorithmic Trading Decisions.
## Dataset
I utilized the yfinance API to gather real time data of Date, Opening Price, Closing Price, Volume of NVIDIA stocks traded in the past 1 year. Since there are no missing values in all columns, there is no need to do validation for missing data
## Results
Predictions (Current Date is June 18 2024)
```
                 Close
2024-06-18  124.979766
2024-06-19  129.935653
2024-06-20  131.108219
2024-06-21  130.395055
2024-06-24  130.256770
2024-06-25  133.580000
2024-06-26  133.732131
2024-06-27  132.920354
2024-06-28  134.324483
2024-07-01  135.421297
2024-07-02  136.417627
2024-07-03  136.773155
2024-07-04  137.339627
2024-07-05  139.191142
2024-07-08  139.511921
2024-07-09  139.296480
2024-07-10  139.887856
2024-07-11  142.529925
2024-07-12  142.191649
2024-07-15  143.099060
2024-07-16  144.411869
2024-07-17  145.023600
2024-07-18  145.385406
2024-07-19  146.050281
2024-07-22  147.346593
2024-07-23  148.621164
2024-07-24  148.818604
2024-07-25  149.940687
2024-07-26  150.831691
2024-07-29  151.045197
2024-07-30  152.421400
2024-07-31  152.706006
2024-08-01  154.508260
2024-08-02  154.917464
2024-08-05  155.962418
```
Simple Moving Average Strategy Accuracy = ``` 0.0034965034965034965 / 0.35% ``` (Extremely Low!)
## Conclusion
While the autoTS function was able to find the most accurate model based on a set of fixed metrics, theoretically, the predicted prices should be quite accurate. However, taking into account the highly volatile nature of the stock market, which movements are determined by many external dynamics and factors, one model can never be too precise. Hence, there are still lots of room for improvements in predictions involving the stock market. As indicated by the low performance of the SMA strategy, we would require more complex and precise strategies for larger accuracies.
