
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
from sklearn.svm import SVR
import datetime
import csv


#Importing Data

path = '/Users/zacharyrich/Desktop/bitcoin_cash_price.csv'

with open(path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        tempRow = row[0].split(",")
        DOW = tempRow[0]


        if DOW == 'Sunday':
            DOW = 0
        elif DOW == 'Monday':
            DOW = 1
        elif DOW == 'Tuesday':
            DOW = 2
        elif DOW == 'Wednesday':
            DOW = 3
        elif DOW == 'Thursday':
            DOW = 4
        elif DOW == 'Friday':
            DOW = 5
        elif DOW == 'Saturday':
            DOW = 6

        print(DOW)
    


data.dropna

data.head()
data.info()
data.describe()

X = data[['Date', 'Open', 'High', 'Low']]
y = data['Close']
