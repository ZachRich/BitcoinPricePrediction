

import csv
import matplotlib.pyplot as plt

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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizaing the Linear Regression Results
plt.scatter(X, y, color='red')  # Plots the actual data
plt.plot(X, lin_reg.predict(X), color='blue')  # Plots the predicted data
plt.title('Truth or Bluff (Linear Regresion)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression Results
plt.scatter(X, y, color='red')  # Plots the actual data
plt.plot(X, lin_reg2.predict(X_poly), color='blue')  # Predicted data
plt.title('Truth or Bluff(Polynomial Regression, degree 4)')
plt.xlabel('Posiiton Level')
plt.ylabel('Salary')
plt.show()
