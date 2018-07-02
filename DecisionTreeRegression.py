# Decision Tree Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Import the dataset using pandas library
dataset = pd.read_csv('C:\\Users\\Gustavo\\Documents\\Big data e Cloud\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 8 - Decision Tree Regression\Decision_Tree_Regression\\Position_Salaries.csv')

print(dataset)
# Dividing the dataset in dependent and independent variables
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

print('X matrix \n')
print(X)
print('y vector \n')
print(y)

# Fitting decision tree regression to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualising the decision tree regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()