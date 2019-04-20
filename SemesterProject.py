# Bitcoin Price prediction with Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Importing the dataset
dataset = pd.read_csv('bitcoin_price.csv')

# Splitting Dataset
X_Open = dataset.iloc[:, 3:4].values

y_High = dataset.iloc[:, 1:2].values

y_low = dataset.iloc[:, 2:3].values

y_close = dataset.iloc[:, -1].values


def linear_prediction(open_price):
    testClosePrice = 7144.38
    # Fitting Linear Regression to the dataset

    lin_reg_open = LinearRegression()
    lin_reg_open.fit(X_Open, y_close)

    lin_reg_high = LinearRegression()
    lin_reg_high.fit(X_Open, y_High)

    lin_reg_low = LinearRegression()
    lin_reg_low.fit(X_Open, y_low)

    # Linear Predictions
    x_test_high = np.array(open_price)
    pred_high = lin_reg_high.predict(x_test_high.reshape(1, -1))  # High

    x_test_low = np.array(open_price)
    pred_low = lin_reg_low.predict(x_test_low.reshape(1, -1))  # Low

    x_test = np.array((open_price + pred_high + pred_low) / 3)  # Open
    y_pred = lin_reg_open.predict(x_test.reshape(1, -1))

    error = (((testClosePrice - y_pred) / testClosePrice) * 100).__abs__()

    return open_price, pred_high, pred_low, y_pred, error

    # print("Linear Regression")
    #
    # print("Open Price:", open_price)
    #
    # print("Predicted High Price: ", pred_high)
    #
    # print("Predicted Low Price: ", pred_low)
    #
    # print("Predicted Close Price: ", y_pred)
    #
    # print("Percent Error: ", error)


# def polynomial_prediction(open_price):
#
#     testClosePrice = 7144.38
#
#     #Fitting polynomial Regression to dataset
#
#     poly_high = PolynomialFeatures(degree=4)
#     X_poly_high = poly_high.fit_transform(X_Open)
#     poly_high.fit(X_poly_high, y_High)
#     lin_reg_high = LinearRegression()
#     lin_reg_high.fit(X_poly_high, y_High)
#
#     poly_low = PolynomialFeatures(degree=4)
#     X_poly_low = poly_low.fit_transform(X_Open)
#     poly_low.fit(X_poly_low, y_low)
#     lin_reg_low = LinearRegression()
#     lin_reg_low.fit(X_poly_low, y_low)
#
#     poly_open = PolynomialFeatures(degree=4)
#     X_poly_open = poly_open.fit_transform(X_Open)
#     poly_open.fit(X_poly_open, y_close)
#     lin_reg_open = LinearRegression()
#     lin_reg_open.fit(X_poly_open, y_close)
#
#
#     #Polynomial Predictions
#     x_test_high = np.array(open_price)
#     pred_high = lin_reg_high.predict(x_test_high.reshape(1, -1))  # High
#
#     x_test_low = np.array(open_price)
#     pred_low = lin_reg_low.predict(x_test_low.reshape(1, -1))  # Low
#
#     x_test = np.array((open_price + pred_low + pred_high) / 3)  # Open
#     y_pred = lin_reg_open.predict(x_test.reshape(1, -1))
#
#     error = (((testClosePrice - y_pred) / testClosePrice) * 100).__abs__()
#
#     print("Polynomial Predictions")
#
#     print("Open Price:", open_price)
#
#     print("Predicted High Price: ", pred_high)
#
#     print("Predicted Low Price: ", pred_low)
#
#     print("Predicted Close Price: ", y_pred)
#
#     print("Percent Error: ", error)


linear_prediction(7023.1)
