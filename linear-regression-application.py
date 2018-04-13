import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle


df = quandl.get('NSE/MRF')

# For features
df = df[['Open', 'High', 'Low', 'Close', 'Total Trade Quantity']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100
df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
df = df[['Close', 'HL_PCT', 'PCT_Change', 'Total Trade Quantity']]
df.fillna(-99999, inplace=True)

# For label
forecast_col = 'Close'
forecast_ahead = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_ahead)

# Numpy arrays for training, testing and prediction
x = np.array(df.drop('label', 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_ahead:]
x = x[:-forecast_ahead]

df.dropna(inplace=True)
y = np.array(df['label'])

#Classifier and prediction
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = LinearRegression()
clf.fit(x_train, y_train)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
predictions = clf.predict(x_lately)
print(predictions)

# Plotting the prediction
df['Forecast'] = np.nan
last_day = df.iloc[-1].name
last_unix = last_day.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for futurevalue in predictions:
    next_day = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_day] = [np.nan for _ in range(len(df.columns) - 1)] + [futurevalue]

df['Close'].plot()
df['Forecast'].plot()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
