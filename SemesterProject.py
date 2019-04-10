
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries
import sklearn.svm
import datetime
import csv
import sklearn.cross_validation

# Importing Data

path = '/Users/zacharyrich/Desktop/bitcoin_cash_price.csv'

X = []
y = []

with open(path) as csv_file:  # Reading in data from CSV
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        tempRow = row[0].split(",")  # Splitting data by ","
        DOW = tempRow[0]

        if DOW == 'Sunday':  # 0-6 for days of the week starting at Sunday
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

        row[0] = DOW  # Replacing dates with Quantifiers

        X.append([row[0], row[1], row[2], row[3]])
        y.append(row[4])

# Make sure to specify test_size also
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75,
                                                    test_size=0.25)

print(X_train)

