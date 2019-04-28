# import matplotlib
# import seaborn as sns
# import quandl
# import numpy as np
# import scipy as sp
# import pandas as pd
# import sklearn.linear_model
# import sklearn.metrics
# #import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# from quandl.errors.quandl_error import NotFoundError
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from sklearn.linear_model import LinearRegression
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.svm import SVR

#New Program
import warnings
import sys
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import quandl
quandl.ApiConfig.api_key = "Ymsx56XR6jmFmd5Nt8bn"
import json
import requests
from requests.auth import HTTPBasicAuth
import pickle as pl
import jsonpickle as jsonConvert 
import random
from collections import Mapping
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/xpred')
def xpred():
    print("API calls")
    #randomPrices = quandl.get("DOE/RWTC")#panda data frame
    # randomPrices = quandl.get("OPEC/ORB")
    random.seed(100)
    rangeSerie = pd.date_range(start='2008', periods = 130, freq='M')
    print("rangeSerie", rangeSerie)
    randomPrices = pd.Series(np.random.uniform(-5, 5, size=len(rangeSerie)), rangeSerie).cumsum() 
    randomPrices.plot(c='b', title='XReview Price Changes')
#     plt.show()
    # randomPrices.head(10)
    
    # randomPrices = quandl.get("OPEC/ORB", start_date="2001-12-31", end_date="2009-12-31")
    randomPrices.head()
    #randomPrices.plot()
    
    #randomPrices['Value']
    randomPrices = randomPrices.resample('MS').mean()
    randomPrices = randomPrices.fillna(randomPrices.bfill())
    
    print(randomPrices)
    
    randomPrices.plot(figsize=(15, 6)) 
    p = d = q = range(0 , 2)
      
    pdq = list(itertools.product(p, d, q))
      
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
      
#     print('Examples of parameter combinations for Seasonal ARIMA')
#     print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#     print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#     print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#     print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
      
          
    warnings.filterwarnings("ignore") # specify to ignore warning messages
      
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(randomPrices,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
       
                results = mod.fit()
       
#                 print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
       
    mod = sm.tsa.statespace.SARIMAX(randomPrices,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
       
    results = mod.fit()
       
#     print(results.summary().tables[1])
       
    # results.plot_diagnostics(figsize=(15, 12))
    # plt.show()
      
      
    # pred_dynamic = results.get_prediction(start=pd.to_datetime('2003-01-02').date(),end='2018-03-01', dynamic=False, full_results=True)
    pred_dynamic = results.get_prediction(start = pd.to_datetime('2008-01-01'),  dynamic=True, full_results=True)
    pred_dynamic_ci  = pred_dynamic.conf_int()
      
    y_forecasted = pred_dynamic.predicted_mean
    y_truth = randomPrices['2008-01-02':]
#     print(y_truth)
    print(y_truth.to_json(orient='index'))
      
    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
#     print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
      
         
    # Get forecast 100 steps ahead 
    pred_uc = results.get_forecast(steps=30)
#     print(pred_uc)
      
    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
#     print(pred_uc.predicted_mean)
    print(pred_uc.predicted_mean.to_json(orient='index'))
    
#     jsonStringA = y_truth.to_json(orient='index')
    jsonStringA = randomPrices.to_json(orient='index')
    jsonStringB = pred_uc.predicted_mean.to_json(orient='index')
    realData = json.loads(jsonStringA)
    forecastedData = json.loads(jsonStringB)
    dataSent = {'real': realData, 'forecasted':forecastedData}
    json.dumps(dataSent)
      
    ax = randomPrices.plot(label='Real Data', figsize=(20, 15))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecasted Data')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Amazon Product Price Changes')
#     print(ax)  
#     print(json.dumps(results.summary().tables[1]).to_json())
    plt.legend()
    plt.show()
    # plt.savefig('Json_example.png')
#     fig_handle = plt.figure()
#     pl.dump(fig_handle, open('sinus.pickle', 'wb'))
#     testJson = jsonConvert.encode(fig_handle)
#     print(testJson)
    
#     auth = HTTPBasicAuth('private_plotly', '43dfd44')
#     headers = {'Plotly-Client-Platform' : 'python'}
     
    #insert all the values of chart into the following code snippet
#     payload = {"data": { "cols": { "first column": \
#             {"data": ["a", "b", "c"], "order": 0}, \
#         "second column": {"data": [1, 2, 3], "order": 1} } } }
     
#     req = requests.post('IP address', auth=auth, headers = headers, json = payload)
    # 
#     print("req", req)
    return json.dumps(dataSent)


xpred()




if __name__=="__main__":
    app.run(host='127.0.0.1', port=4999)

