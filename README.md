# Predicting Future NVIDIA Stock Prices and Making Trading Decisions
## Overview
This time series project aims to utilize machine learning algorithms to predict real-time NVIDIA stocks 30 days from now (Until End of July 2024). I implemented the powerful autoTS time series library limited to 2 models to optimize (ARIMA, Prophet) to accurately predict future stock prices. Furthermore, I conducted an analysis of past, current and future stock closing prices to construct a "Momentum" Strategy for Algorithmic Trading. This momentum investing is a system of buying stocks or other securities that have had high returns over the past three to twelve months, and selling those that have had poor returns over the same period. Hence, if the momentum value is positive, we buy. If it's negative, we sell.
## Dataset
I utilized the yfinance API to gather real time data of Date, Opening Price, Closing Price, Volume of NVIDIA stocks traded in the past 1 year. Since there are no missing values in all columns, there is no need to do validation for missing data
# Results
Predictions (Current Date is June 18 2024)
```
                Close
2024-06-18  126.850574
2024-06-19  127.751707
2024-06-20  128.707031
2024-06-21  129.666701
2024-06-24  129.869194
2024-06-25  130.739477
2024-06-26  131.946853
2024-06-27  132.948685
2024-06-28  133.843719
2024-07-01  134.792952
2024-07-02  135.746535
2024-07-03  135.942949
2024-07-04  136.807158
2024-07-05  138.008466
2024-07-08  139.004236
2024-07-09  139.893215
2024-07-10  141.296070
2024-07-11  141.783938
2024-07-12  141.974314
2024-07-15  142.832492
2024-07-16  144.027775
2024-07-17  145.017526
2024-07-18  145.900492
2024-07-19  146.837667
2024-07-22  147.779206
2024-07-23  147.963587
2024-07-24  148.815775
2024-07-25  150.005075
2024-07-26  150.988848
2024-07-29  151.865843
```
![Algo Trading Results](https://drive.google.com/file/d/1em0aqXWLskhD6ApXJcrpgDpvkPBDgzim/view?usp=sharing)
## Conclusion
While the autoTS function was able to find the most accurate model based on a set of fixed metrics, theoretically, the predicted prices should be quite accurate. However, taking into account the highly volatile nature of the stock market, which movements are determined by many external dynamics and factors, one model can never be too precise. Hence, there are still lots of room for improvements in predictions involving the stock market.
