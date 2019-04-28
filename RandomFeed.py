import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import  shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import json
import pickle as pl
import jsonpickle as jsonConvert 

from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def main():
    return "Welcome!"

@app.route('/deeplearning')
def test():
    print("RandomSeedCreator")
    random.seed(100)
    rangeSerie = pd.date_range(start='2006', periods = 150, freq='M')
    print("rangeSerie", rangeSerie)
    ts = pd.Series(np.random.uniform(5, -5, size=len(rangeSerie)), rangeSerie).cumsum() 
    ts.plot(c='b', title='XReview Price Deviation')
    plt.show()
    ts.head(10)
    
    TS = np.array(ts)
    num_periods = 20
    f_horizon = 1 # forecast horizon
    
    x_data = TS[:(len(TS) - (len(TS) % num_periods))]
    x_batches = x_data.reshape(-1, 20, 1)
    
    y_data = TS[1:(len(TS) - (len(TS) % num_periods)) + f_horizon]
    y_batches = y_data.reshape(-1, 20, 1   )
    print(len(x_batches))
    print(x_batches.shape)
    print(x_batches[0:2])
    
    print(y_batches[0:1])
    print(y_batches.shape)
    
    def test_data(series, forecast, num_periods):
        test_x_setup = TS[-(num_periods + forecast):]
        testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
        testY =TS[-(num_periods):].reshape(-1, 20, 1)
        return testX, testY
    
    X_test, Y_test = test_data(TS, f_horizon, num_periods)
    print(X_test.shape)
    print(X_test)    
    
    tf.reset_default_graph()
    
    num_periods = 20
    inputs = 1
    hidden = 300
    output = 1
    
    X = tf.placeholder(tf.float32, [None, num_periods, inputs])
    Y = tf.placeholder(tf.float32, [None, num_periods, output])
    
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden, activation = tf.nn.relu)
    rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
    
    learning_rate = 0.001 #small learning rate
    
    stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])
    
    loss = tf.reduce_sum(tf.square(outputs-Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    epochs = 2000
    
    with tf.Session() as sess:
        init.run()
        for ep in range(epochs):
            sess.run(training_operation, feed_dict={X:x_batches, Y:y_batches})
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={X: x_batches, Y:y_batches})
                print(ep, "\tMSE:", mse)
        
        y_pred = sess.run(outputs, feed_dict={X:X_test})
#         print(y_pred)
    
    plt.title("Forecast and Real", fontsize=14)
    print("Y_test", Y_test)
    print("Y_pred", y_pred)
    plt.plot(pd.Series(np.ravel(Y_test)), "black", markersize=4, label="Real")
    plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=13, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods(Months)")
    plt.show()
    

    return "hello"

test()

if __name__=="__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)


