#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import Series
from pandas import concat

from math import sqrt

from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error


# In[2]:


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


# In[3]:


series = read_csv('BOPGSTB.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.plot()
plt.show()


# In[4]:


# # Subtracting the Trend Component
# result_mul = seasonal_decompose(df['BOPGSTB'], model = 'additive', extrapolate_trend = 'freq')
# series = df.values.reshape(-1)
# detrended = series - result_mul.trend.values
# plt.plot(df.index, detrended)
# plt.title('Detrended Time Series', fontsize = 12)
# plt.xlabel('Date')
# plt.ylabel('Trade Balance (in Millions of $)')
# plt.show()


# In[5]:


# transform = log(series*-1)*-1

# plt.plot(df.index, transform)
# plt.title('Log-transformed Plot', fontsize = 12)
# plt.xlabel('Date')
# plt.ylabel('Log Trade Balance')
# plt.show()


# In[6]:


# Autocorrelation Function
plot_acf(series, lags=60); # => Series isn't stationary


# In[7]:


# Partial Autocorrelation Fucntion
plot_pacf(series);


# ### ACF is slow decay -> AR or ARMA
# ### PACF has no trend, just p-significant lags -> AR model

# In[8]:


series.index = series.index.to_period('M')


# In[9]:


model = ARIMA(series, order=(7,0,0))
model_fit = model.fit()

print(model_fit.summary())

# Only first 4 lags are significant


# In[10]:


model = ARIMA(series, order=(4,0,0))
model_fit = model.fit()

print(model_fit.summary())


# In[11]:


residuals = DataFrame(model_fit.resid)
residuals.plot(legend=False)
plt.show()


# In[12]:


residuals.plot(kind='kde', legend=False)
plt.show()


# In[13]:


# summary stats of residuals
print(residuals.describe())


# In[14]:


X = series
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
    
# Clearly, p-value > 0.05 => Series is non-stationary


# In[15]:


plot_acf(X);

# More than 10 lags show positive decaying autocorrelation
# => Series is non-stationary


# In[16]:


X = X.diff().dropna()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
    
# Clearly, p-value < 0.05 => Series has now become stationary
# One-differencing (d = 1) required for ARIMA model


# In[17]:


plot_acf(X);

# Clearly a stationary series now


# In[18]:


model = ARIMA(series, order=(4,1,0))
model_fit = model.fit()

print(model_fit.summary())

# 4th lag becomes insignificant now, let's remove and see if AIC/BIC reduces


# In[19]:


model = ARIMA(series, order=(3,1,0))
model_fit = model.fit()

print(model_fit.summary())

# A better model


# In[20]:


X = series.values
size = int(len(X) * 0.66)

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(3,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[21]:


# Plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# In[22]:


forecast = model_fit.forecast(steps=12)
forecast


# In[23]:


forecast[0]


# In[24]:


df_pred = model_fit.get_forecast(steps=12).summary_frame(alpha=0.05).drop('mean_se', axis=1)
df_pred


# In[25]:


plt.rcParams["legend.loc"] = 'best'
df_pred.plot(figsize=(8, 6));


# In[26]:


forecast_dates = ['2022-03-01', '2022-04-01', '2022-05-01', '2022-06-01',
                  '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01',
                  '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01']

s = Series([])
for date in forecast_dates:
    s = s.append(Series(parser(date))).reset_index(drop=True)

forecast_df = concat([s, Series(forecast)], axis=1).set_index(0)


# In[27]:


series = read_csv('BOPGSTB.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
train, test = series[0:size], series[size:len(X)]

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(test, label='In-Sample Observations')
ax.plot(forecast_df, label='Forecasts for next 12 months')
ax.legend()
plt.xticks(['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019','2020', '2021', '2022', '2023'], rotation=45);

