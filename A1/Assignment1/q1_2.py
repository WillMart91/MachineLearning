import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from q1_1 import data_matrix_bias, linear_regression_predict, linear_regression_optimize, rmse

# Loading the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

# Using the functions you have implemented, fit and evaluate a linear regression model in 
# q1 2.py. You should find the optimal parameters using the training data. We have provided 
# part of the code and you need to complete it. Report the RMSE value on the test data, and 
# plot the predicted salary and the actual salary on the test set in one graph as a function 
# of both variables: experience and test score. You can plot two separate scatter plots, one 
# per variable. Show the resulting plots in your .pdf report file
# Write your code here:

X_train_bias = data_matrix_bias(X_train) 
X_test_bias = data_matrix_bias(X_test)

X_experience_train_bias = data_matrix_bias(X_train[:, 0].reshape(-1, 1))
X_experience_test_bias = data_matrix_bias(X_test[:, 0].reshape(-1, 1))

X_test_score_train_bias = data_matrix_bias(X_train[:, 1].reshape(-1, 1))
X_test_score_test_bias = data_matrix_bias(X_test[:, 1].reshape(-1, 1))

w = linear_regression_optimize(y_train, X_train_bias)
y_hat = linear_regression_predict(X_test_bias, w)

w_experience = linear_regression_optimize(y_train, X_experience_train_bias)
y_hat_experience = linear_regression_predict(X_experience_test_bias, w_experience)
w_test_score = linear_regression_optimize(y_train, X_test_score_train_bias)
y_hat_test_score = linear_regression_predict(X_test_score_test_bias, w_test_score)


rmse_val = rmse(y_test, y_hat)
print(f"RMSE for a linear regression of two features (experience and test score): {rmse_val}")
rmse_val_experience = rmse(y_test, y_hat_experience)
print(f"RMSE for a linear regression of experience: {rmse_val_experience}")
rmse_val_test_score = rmse(y_test, y_hat_test_score)
print(f"RMSE for a linear regression of test score: {rmse_val_test_score}")

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(X_test[:, 0], y_test, label='Actual')
plt.scatter(X_test[:, 0], y_hat_experience, label='Predicted')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary : Linear Regression of one features')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(X_test[:, 1], y_test, label='Actual')
plt.scatter(X_test[:, 1], y_hat_test_score, label='Predicted')
plt.xlabel('Test Score')
plt.ylabel('Salary')
plt.title('Test Score vs Salary : Linear Regression of one features')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(X_test[:, 0], y_test, label='Actual')
plt.scatter(X_test[:, 0], y_hat, label='Predicted')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary : Linear Regression of two features')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(X_test[:, 1], y_test, label='Actual')
plt.scatter(X_test[:, 1], y_hat, label='Predicted')
plt.xlabel('Test Score')
plt.ylabel('Salary')
plt.title('Test Score vs Salary : Linear Regression of two features')
plt.legend()

plt.tight_layout()
plt.show()