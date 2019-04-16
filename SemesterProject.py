# Bitcoin Price prediction with Polynomial Regression

# Imports
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

path = '/Users/zacharyrich/Desktop/bitcoin_cash_price.csv'

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

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:4].values

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y_train)

# # Visualizaing the Linear Regression Results
plt.scatter(y_test, lin_reg.predict(X_test), color='red')  # Plots the actual data
plt.plot(X, lin_reg.predict(X), color='blue')  # Plots the predicted data
plt.title('Bitcoin Price Prediction(Linear Regresion)')
plt.xlabel('Features(Day of week, High, Low, Open)')
plt.ylabel('Close Price')
plt.show()

# Visualizing the Polynomial Regression Results
plt.scatter(y_test, lin_reg2.predict(X_test), color='red')  # Plots the actual data
plt.plot(X, lin_reg2.predict(X_poly), color='blue')  # Predicted data
plt.title('Bitcoin Price Prediction(Polynomial Regresion)')
plt.xlabel('Features(Day of week, High, Low, Open)')
plt.ylabel('Close Price')
plt.show()
