from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


master = Tk()
master.title("Bitcoin Price Predictor")

Label(master, text="Open Price:").grid(row=0)
Label(master, text="High Price Prediction:").grid(row=1)
Label(master, text="Low Price Prediction:").grid(row=2)
Label(master, text="Close Price Prediction:").grid(row=3)

#user input for open price
e1 = Entry(master)
e1.grid(row=0, column=1)

#where predicted prices will display
v1 = Entry(master, state=DISABLED)
v1.grid(row=1, column=1)

v2 = Entry(master, state=DISABLED)
v2.grid(row=2, column=1)

v3 = Entry(master, state=DISABLED)
v3.grid(row=3, column=1)


# Importing the dataset
dataset = pd.read_csv('bitcoin_price.csv')

# Splitting Dataset
X_Open = dataset.iloc[:, 3:4].values

y_High = dataset.iloc[:, 1:2].values

y_low = dataset.iloc[:, 2:3].values

y_close = dataset.iloc[:, -1].values


def linRegButton(open_price):
    
    enteredByUser = float(e1.get())
    testClosePrice = 7144.38
     # Fitting Linear Regression to the dataset

    lin_reg_open = LinearRegression()
    lin_reg_open.fit(X_Open, y_close)

    lin_reg_high = LinearRegression()
    lin_reg_high.fit(X_Open, y_High)

    lin_reg_low = LinearRegression()
    lin_reg_low.fit(X_Open, y_low)
    
    # Linear Predictions
    x_test_high = np.array(enteredByUser)
    pred_high = lin_reg_high.predict(x_test_high.reshape(1, -1))  # High

    x_test_low = np.array(enteredByUser)
    pred_low = lin_reg_low.predict(x_test_low.reshape(1, -1))  # Low

    x_test = np.array((enteredByUser + pred_high + pred_low) / 3)  # Close
    y_pred = lin_reg_open.predict(x_test.reshape(1, -1))
    
    #showing entered open price
    e1.configure(state=NORMAL)
    e1.delete(0,'end')
    e1.insert(0, str(open_price))
    e1.configure(state=DISABLED)
    
    #display high prediction
    v1.configure(state=NORMAL)
    v1.delete(0,'end')
    v1.insert(0,str(pred_high))
    v1.configure(state=DISABLED)
    
    #display low prediction
    v2.configure(state=NORMAL)
    v2.delete(0,'end')
    v2.insert(0,str(pred_low))
    v2.configure(state=DISABLED)
    
    #display close prediction
    v3.configure(state=NORMAL)
    v3.delete(0,'end')
    v3.insert(0,str(y_pred))
    v3.configure(state=DISABLED)

Button(master, text='Linear Regression', command=lambda: linRegButton(e1)).grid(row=6, column=0, sticky=W, pady=4)

mainloop()