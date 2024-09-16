import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from q1_1 import rmse
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import gradient_descent_regression
from q3_2 import computeLoss

# Load the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

# Hyperparameters
num_epochs = 1000
ridge_hyperparameter = 0.05
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]  # Different learning rates to try

# Now, try different values for ‘learning rate‘ hyperparameter. Plot the training loss after each epoch,
# and the RMSE versus the ‘learning rate‘ for both linear and ridge regression on the test dataset. 
# Explain how changing this hyperparameter affects the training process

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
## Maybe add bias from 1_1 ???????????

training_loss_data_data_simple = []
training_loss_data_data_ridge = []
simple_rmse_data = []
ridge_rmse_data = []

for learning_rate in learning_rates:
    w_simple, b_simple, training_loss_data_simple = gradient_descent_regression(X_train, y_train, reg_type='simple', learning_rate=learning_rate, num_epochs=num_epochs)
    w_ridge, b_ridge, training_loss_data_ridge = gradient_descent_regression(X_train, y_train, reg_type='ridge', hyperparameter=ridge_hyperparameter, learning_rate=learning_rate, num_epochs=num_epochs)
    
    training_loss_data_data_simple.append(training_loss_data_simple)
    training_loss_data_data_ridge.append(training_loss_data_ridge)

    simple_rmse_data.append(rmse(y_test, np.dot(X_test, w_simple) + b_simple))
    ridge_rmse_data.append(rmse(y_test, np.dot(X_test, w_ridge) + b_ridge))


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for i, learning_rate in enumerate(learning_rates):
    plt.plot(training_loss_data_data_simple[i], label=f'Learning Rate: {learning_rate}')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss troughout epoch for Simple Linear Regression')
plt.legend()

plt.subplot(1, 2, 2)
for i, learning_rate in enumerate(learning_rates):
    plt.plot(training_loss_data_data_ridge[i], label=f'Learning Rate: {learning_rate}')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss troughout epoch for Rigde Linear Regression')
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(learning_rates, simple_rmse_data, marker='o', linestyle='-', color='b', label='Simple Linear Regression')
plt.plot(learning_rates, ridge_rmse_data, marker='o', linestyle='-', color='r', label='Ridge Regression')
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.title('RMSE vs Learning Rate for Simple and Ridge Regression')
plt.legend()
plt.grid(True)
plt.ylim(0, 40000)
plt.show()