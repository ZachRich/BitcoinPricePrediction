# Bitcoin Price prediction with Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = '/Users/zacharyrich/Desktop/bitcoin_price.csv'

dataset = pd.read_csv(path)

# Replace Days of week with their Numbers
replacements = {
    'Sunday': 0,
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6
}

dataset.replace(replacements, inplace=True, )

X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, -1].values

# Importing the dataset
dataset = pd.read_csv('bitcoin_price.csv')
X = dataset.iloc[:, 3:4].values
y = dataset.iloc[:, -1].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizaing the Linear Regression Results
plt.scatter(X, y, color='red')  # Plots the actual data
plt.plot(X, lin_reg.predict(X), color='blue')  # Plots the predicted data
plt.title('Close price using open(Linear Regresion)')
plt.xlabel('Open')
plt.ylabel('Close')
plt.show()

# Visualizing the Polynomial Regression Results
plt.scatter(X, y, color='red')  # Plots the actual data
plt.plot(X, lin_reg2.predict(X_poly), color='blue')  # Predicted data
plt.title('Close price using open(Polynomial Regression, degree 4)')
plt.xlabel('Open')
plt.ylabel('Close')
plt.show()

# Improving the Visualizing the Polynomial Regression Result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')  # Plots the actual data
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')  # Predicted data
plt.title('Close price using open(Polynomial Regression, degree 4 Continuous Curve)')
plt.xlabel('Open')
plt.ylabel('Close')
plt.show()
