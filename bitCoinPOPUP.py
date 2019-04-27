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
Label(master, text="Percent Error(%):").grid(row=4)

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

v4 = Entry(master, state=DISABLED)
v4.grid(row=4, column=1)

v5 = Entry(master, state=DISABLED)
v5.grid(row=5, column=1)


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

def linRegButton(linear_prediction, open_price, pred_high, pred_low, y_pred, error):
    linear_prediction(*open_price, pred_high, pred_low, y_pred, error)
   
    #showing entered open price
    v1.configure(state=NORMAL)
    v2.delete(0,'end')
    v3.insert(0, str(open_price))
    v4.configure(state=DISABLED)
    
    #display high prediction
    v2.configure(state=NORMAL)
    v2.delete(0,'end')
    v2.insert(0,str(pred_high))
    v2.configure(state=DISABLED)
    
    #display low prediction
    v3.configure(state=NORMAL)
    v3.delete(0,'end')
    v3.insert(0,str(pred_low))
    v3.configure(state=DISABLED)
    
    #display close prediction
    v4.configure(state=NORMAL)
    v4.delete(0,'end')
    v4.insert(0,str(y_pred))
    v4.configure(state=DISABLED)
    
    #display error prediction
    v5.configure(state=NORMAL)
    v5.delete(0,'end')
    v5.insert(0,str(error))
    v5.configure(state=DISABLED)

#def calc(K, Cn):
#    # get the user input as floats
#    C0 = float(e1.get())
#    p = float(e2.get())
#    n = float(e3.get())
#    # < put your input validation here >
#
#    #Amount of interest for repayment:
#    K.configure(state=NORMAL) # make the field editable
#    K.delete(0, 'end') # remove old content
#    K.insert(0, str((C0 * p * n) / 100)) # write new content
#    K.configure(state=DISABLED) # make the field read only
#
#    #Final Debt Amount:
#    Cn.configure(state=NORMAL) # make the field editable
#    Cn.delete(0, 'end') # remove old content
#    Cn.insert(0, str(C0 * (1 + (p * n) / 100))) # write new content
#    Cn.configure(state=DISABLED) # make the field read only


#Button(master, text='Quit', command=master.quit).grid(row=5, column=0, sticky=E, pady=4)
Button(master, text='Linear Regression', command=lambda: linRegButton(e1, v1, v2, v3, v4)).grid(row=6, column=0, sticky=W, pady=4)
Button(master, text='Non-Linear Regression', command=lambda: calc(K, Cn)).grid(row=7, column=0, sticky=W, pady=4)

mainloop()