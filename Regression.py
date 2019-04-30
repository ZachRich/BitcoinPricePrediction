# Bitcoin Price prediction with Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('data.csv')

# Splitting Dataset
X_Open = dataset.iloc[:, 1:2].values  # Open Prices

y_High = dataset.iloc[:, 2:3].values  # High Prices

y_low = dataset.iloc[:, 3:4].values  # Low Prices

y_close = dataset.iloc[:, -1].values  # Close Prices

def linear_prediction(open_price):
    testClosePrice = 5140.38
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

    print("Linear Regression")

    print("Open Price:", open_price)

    print("Predicted High Price: ", pred_high)

    print("Predicted Low Price: ", pred_low)

    print("Predicted Close Price: ", y_pred)

    print("Percent Error: ", error)

    return open_price, pred_high, pred_low, y_pred, error


linear_prediction(5140.38)
