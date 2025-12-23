import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get dataset
dataset = pd.read_csv('./data/nba/3ptEfficiency-advanced.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, -1].values

# Split data into test and training sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fit the training data to a LinearRegression algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

# Predict the test set results
y_pred = regressor.predict(X_test)

# show results
plt.scatter(X_train, y_train, color = 'red', s = 1)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('3pt Shot Attempts (Training set)')
plt.xlabel('Season')
plt.ylabel('Attempt %')
plt.show()
