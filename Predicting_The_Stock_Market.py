# importing libraries
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# reading file into dataframe
sphist = pd.read_csv('sphist.csv')

# converting Date column to pandas datetime
sphist['Date'] = pd.to_datetime(sphist['Date'])

# sorting values by Date column and in ascending order
sphist.sort_values('Date', inplace=True)

# creating new column for 5 days mean against stock closing price
sphist['5_days_mean'] = sphist.Close.rolling(5, win_type='triang', on='Date').mean()

# rolling mean will use the current day's price, therefore we need to reindex the resulting series
# to shift all the values "forward" one day, i.e. the rolling mean calculated for 1950-01-03 will
# need to be assigned to 1950-01-04, and so on
sphist = sphist.shift(periods=1, freq=None)

# creating new column for 365 days mean against stock closing price
sphist['365_days_mean'] = sphist.Close.rolling(365, win_type='triang', on='Date').mean()
sphist = sphist.shift(periods=1, freq=None)

# calculating ratio of 5 days mean and 365 days mean
sphist['mean_ratio'] = sphist['5_days_mean'] / sphist['365_days_mean']

# creating new column for 5 days standard deviation against stock closing price
sphist['5_days_std'] = sphist.Close.rolling(5, win_type='triang', on='Date').std()
sphist = sphist.shift(periods=1, freq=None)

# creating new column for 365 days standard deviation against stock closing price
sphist['365_days_std'] = sphist.Close.rolling(5, win_type='triang', on='Date').std()
sphist = sphist.shift(periods=1, freq=None)

# calculating ratio of 5 days std and 365 days std
sphist['std_ratio'] = sphist['5_days_std'] / sphist['365_days_std']

# dropping or ignoring values before Jan 03, 1951 as they don't
# have enough historical data to compute all the indicators. 
sphist_new = sphist[sphist['Date'] > datetime(year=1951, month=1, day=2)]

# dropping NaN values
sphist_new = sphist_new.dropna(axis=0).copy()

# splitting dataframe for training
train_sphist = sphist_new[sphist_new['Date'] < datetime(year=2013, month=1, day=1)]

# splitting dataframe for testing
test_sphist = sphist_new[sphist_new['Date'] >= datetime(year=2013, month=1, day=1)]

# instantiating a linear model
lr = LinearRegression()

# generating a list of required features, excluding all columns that 
# contain knowledge of the future that we don't want to feed the model.
features = list(train_sphist.columns[7:])

# fitting linear model
lr.fit(train_sphist[features], train_sphist['Close'])

# predicting using linear model
predicted_label = lr.predict(test_sphist[features])

# utilizing 'mean absolute error' as an error metric, because it will show
# how "close" we were to the price in intuitive terms.
mae = mean_absolute_error(test_sphist['Close'], predicted_label)

print('Mean Absolute Error:', mae)
print('Coefficient of determination (r^2) of the prediction:', lr.score(train_sphist[features], train_sphist['Close']))