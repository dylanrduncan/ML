""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes

db = load_diabetes()

print(db)


# how many samples and How many features?
print(db.data.shape)

# What does feature s6 represent?
print(db.DESCR)

# print out the coefficient
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    db.data, db.target, random_state=11
)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X=data_train, y=target_train)

print(linear_regression.coef_)

# print out the intercept
print(linear_regression.intercept_)

# create a scatterplot with regression line
predicted = linear_regression.predict(data_train)

expected = target_train

plt.plot(expected, predicted, ".")

x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()
