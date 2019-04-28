from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import gzip
import json

import matplotlib.pyplot as plt

import seaborn

import fix_yahoo_finance as yf
import quandl
quandl.ApiConfig.api_key = "Ymsx56XR6jmFmd5Nt8bn"



Df = yf.download('GLD', '2008-01-01', '2017-12-31')
# Df = quandl.get("OPEC/ORB", start_date="2001-12-31", end_date="2009-12-31")
Df=Df[['Close']] 
Df = Df.dropna()
 
Df.Close.plot(figsize=(10, 5))
 
plt.ylabel("Amazon Product Prices")
 
plt.show()
 
Df['S_3'] = Df['Close'].shift().rolling(window=3).mean()
Df['S_9'] = Df['Close'].shift(1).rolling(window=9).mean()
 
Df = Df.dropna()
 
X = Df[['S_3', 'S_9']]
 
X.head()
 
y = Df['Close']
 
y.head()
 
t = .8
 
t = int(t*len(Df))
 
X_train = X[:t]
 
Y_train = y[:t]
 
X_test = X[t:]
 
y_test = y[t:]
 
linear = LinearRegression().fit(X_train, Y_train)
 
print("Amazon Product Price =", round(linear.coef_[0], 2), \
      "3 Days Moving Average", round(linear.coef_[1], 2), \
      "* 9 Days Moving Average +", round(linear.intercept_, 2))
 
predicted_price = linear.predict(X_test)
 
predicted_price = pd.DataFrame(predicted_price, index = y_test.index, columns = ['price'])
 
predicted_price.plot(figsize=(10, 5))
 
y_test.plot()
 
plt.legend(['predicted_price', 'actual_price'])
 
plt.ylabel("Amazon product price")
 
plt.show()
