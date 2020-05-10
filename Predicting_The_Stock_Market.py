#!/usr/bin/env python
# coding: utf-8

# ## Predicting_The_Stock_Market

# In this project, we worked with data from the S&P500 Index which is a stock market index. The S&P500 Index aggregates the stock prices of 500 large companies. Moreover, we used historical data on the price of the S&P500 Index to make predictions about future prices. Predicting whether an index will go up or down will help us forecast how the stock market as a whole will perform. Since stocks tend to correlate with how well the economy as a whole is performing, it can also help us make economic forecasts.
# 
# ### Note: You shouldn't make trades with any models developed in this mission. Trading stocks has risks, and nothing in this mission constitutes stock trading advice.
# 
# The file that we worked upon is a csv file containing index prices. Each row in the file contains a daily record of the price of the S&P500 Index from 1950 to 2015. The dataset is stored in [sphist.csv]().
# 
# The columns of the dataset are:
# 
# - `Date` -- The date of the record.
# - `Open` -- The opening price of the day (when trading starts).
# - `High` -- The highest trade price during the day.
# - `Low` -- The lowest trade price during the day.
# - `Close` -- The closing price for the day (when trading is finished).
# - `Volume` -- The number of shares traded.
# - `Adj Close` -- The daily closing price, adjusted retroactively to include any corporate actions. Read more [here](http://www.investopedia.com/terms/a/adjusted_closing_price.asp).
# 
# We used this dataset to develop a predictive model and trained the model with data from 1950-2012 to make predictions from 2013-2015.

# In[1]:


# importing libraries
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# reading file into dataframe
sphist = pd.read_csv('sphist.csv')
sphist.head()


# In[2]:


sphist.info()


# In[3]:


# converting Date column to pandas datetime
sphist['Date'] = pd.to_datetime(sphist['Date'])

sphist.info()


# In[4]:


# sorting values by Date column and in ascending order
sphist.sort_values('Date', inplace=True)


# In[5]:


# creating new column for 5 days mean against stock closing price
sphist['5_days_mean'] = sphist.Close.rolling(5, win_type='triang', on='Date').mean()

# rolling mean will use the current day's price, therefore we need to reindex the resulting series
# to shift all the values "forward" one day, i.e. the rolling mean calculated for 1950-01-03 will
# need to be assigned to 1950-01-04, and so on
sphist = sphist.shift(periods=1, freq=None)

sphist.head(10)


# In[6]:


# creating new column for 365 days mean against stock closing price
sphist['365_days_mean'] = sphist.Close.rolling(365, win_type='triang', on='Date').mean()

sphist = sphist.shift(periods=1, freq=None)

sphist.head(10)


# In[7]:


# calculating ratio of 5 days mean and 365 days mean
sphist['mean_ratio'] = sphist['5_days_mean'] / sphist['365_days_mean']


# In[8]:


# creating new column for 5 days standard deviation against stock closing price
sphist['5_days_std'] = sphist.Close.rolling(5, win_type='triang', on='Date').std()

sphist = sphist.shift(periods=1, freq=None)

sphist.head(10)


# In[9]:


# creating new column for 365 days standard deviation against stock closing price
sphist['365_days_std'] = sphist.Close.rolling(5, win_type='triang', on='Date').std()

sphist = sphist.shift(periods=1, freq=None)

sphist.head(10)


# In[10]:


# calculating ratio of 5 days std and 365 days std
sphist['std_ratio'] = sphist['5_days_std'] / sphist['365_days_std']


# In[11]:


# dropping or ignoring values before Jan 03, 1951 as they don't
# have enough historical data to compute all the indicators. 
sphist_new = sphist[sphist['Date'] > datetime(year=1951, month=1, day=2)]

sphist_new = sphist_new.dropna(axis=0).copy()

sphist_new.isnull().sum()


# In[12]:


# splitting dataframe for training
train_sphist = sphist_new[sphist_new['Date'] < datetime(year=2013, month=1, day=1)]

train_sphist.head()


# In[13]:


# splitting dataframe for testing
test_sphist = sphist_new[sphist_new['Date'] >= datetime(year=2013, month=1, day=1)]

test_sphist.head()


# In[28]:


# instantiating a linear model
lr = LinearRegression()

# generating a list of required features, excluding all columns that 
# contain knowledge of the future that we don't want to feed the model.
features = list(train_sphist.columns[8:])
target = ['Close']

# fitting linear model
lr.fit(train_sphist[features], train_sphist[target])


# In[31]:


# predicting using linear model
predicted_label = lr.predict(test_sphist[features])

# utilizing 'mean absolute error' as an error metric, because it will show
# how "close" we were to the price in intuitive terms.
mae = mean_absolute_error(test_sphist[target], predicted_label)
mae


# In[32]:


lr.score(train_sphist[features], train_sphist['Close'])

