# Bitcoin Price prediction with Polynomial Regression

# Imports
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

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

plt.scatter(dataset.iloc[:, 0:1].values, dataset.iloc[:, 3:4].values, color='green')  # Plots the actual data
plt.title('Day Of the Week + Close Price')
plt.xlabel('Day of the week')
plt.ylabel('Close Price')
plt.show()

plt.scatter(dataset.iloc[:, 1:2].values, dataset.iloc[:, 3:4].values, color='red')  # Plots the actual data
plt.title('High + Close Price')
plt.xlabel('High Price')
plt.ylabel('Close Price')
plt.show()

plt.scatter(dataset.iloc[:, 2:3].values, dataset.iloc[:, 3:4].values, color='blue')  # Plots the actual data
plt.title('Low + Close Price')
plt.xlabel('Low Price')
plt.ylabel('Close Price')
plt.show()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y_train)

y_pred = lin_reg2.predict(X_poly)

confidence = lin_reg2.score(X_test, y_test)
print("confidence: ", confidence)
