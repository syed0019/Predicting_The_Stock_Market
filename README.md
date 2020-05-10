# Predicting_The_Stock_Market

### Note: You shouldn't make trades with any models developed in this project. Trading stocks has risks, and nothing in this project constitutes stock trading advice.

In this project, we worked with data from the S&P500 Index which is a stock market index. The S&P500 Index aggregates the stock prices of 500 large companies. Moreover, we used historical data on the price of the S&P500 Index to make predictions about future prices. Predicting whether an index will go up or down will help us forecast how the stock market as a whole will perform. Since stocks tend to correlate with how well the economy as a whole is performing, it can also help us make economic forecasts.

The file that we worked upon is a csv file containing index prices. Each row in the file contains a daily record of the price of the S&P500 Index from 1950 to 2015. The dataset is stored in [sphist.csv](https://github.com/syed0019/Predicting_The_Stock_Market/blob/master/sphist.csv).

The columns of the dataset are:
- Date -- The date of the record.
- Open -- The opening price of the day (when trading starts).
- High -- The highest trade price during the day.
- Low -- The lowest trade price during the day.
- Close -- The closing price for the day (when trading is finished).
- Volume -- The number of shares traded.
- Adj Close -- The daily closing price, adjusted retroactively to include any corporate actions. Read more [here](http://www.investopedia.com/terms/a/adjusted_closing_price.asp).
