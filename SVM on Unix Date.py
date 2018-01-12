import csv
import time
import datetime
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile)
        next(data) # to skip row 1
        for row in data:
            dates.append(int(row[1]))
            prices.append(float(row[5]))
    return
get_data('SENSEX1.csv')
print dates
print prices

def predict(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1))
    
    svr_lin = SVR(kernel = 'linear', C=1e3)
    svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

s = "01-05-1990"
n = int(time.mktime(time.strptime(s, "%d-%m-%Y")))
print n
predicted_price = predict(dates, prices, n)
print predicted_price