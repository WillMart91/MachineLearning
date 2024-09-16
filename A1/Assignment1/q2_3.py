import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from q2_2 import cross_validation_linear_regression

# Define a range of alpha values for hyperparameter search
hyperparams = np.logspace(-4, 4, 50)
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values
kfolds = 5

# Using the functions you implemented, perform a hyperparameter search on the regularization term 
# of the Ridge Regression. Plot the RMSE as a function of the hyperparameter value. How can your 
# observations be theoretically explained? Please provide your code in the q2 3.py file and provide 
# your plots and explanation in the .pdf report file.

best_hyperparam, best_mean_squared_error, mean_squared_errors = cross_validation_linear_regression(kfolds, hyperparams, X_train, y_train)

plt.plot(hyperparams, mean_squared_errors)
plt.xscale('log')
plt.xlabel('Hyperparameter')
plt.ylabel('RMSE')
plt.title('Hyperparameter Search')
plt.show()


