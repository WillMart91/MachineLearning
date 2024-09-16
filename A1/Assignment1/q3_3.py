import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from q1_1 import rmse
from q3_1 import compute_gradient_ridge, compute_gradient_simple
from q3_2 import gradient_descent_regression

# Load the dataset
X_train = pd.read_csv('Data/X_train.csv').values
y_train = pd.read_csv('Data/y_train.csv').values
X_test = pd.read_csv('Data/X_test.csv').values
y_test = pd.read_csv('Data/y_test.csv').values

np.random.seed(42)  # For reproducibility
n_features = X_train.shape[1]
initial_w = np.random.normal(0, 1, size=n_features)
initial_b = 0.0

learning_rate = 0.001  # You can change this value to get better results
num_epochs = 1000
ridge_hyperparameter = 0.5 # You can change this value to get better results

# Now run the above function for both linear and ridge regression on the given dataset. Initialize 
# the initial weight vector using ‘np.random.normal‘. Write the code to find the solution weights for 
# each type of regression. After each epoch, plot the training loss
# Provide your code here ...

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

w_simple, b_simple, training_loss_data_simple = gradient_descent_regression(X_train, y_train, reg_type='simple', learning_rate=learning_rate, num_epochs=num_epochs)
w_ridge, b_ridge, training_loss_data_ridge= gradient_descent_regression(X_train, y_train, reg_type='ridge', hyperparameter=ridge_hyperparameter, learning_rate=learning_rate, num_epochs=num_epochs)

plt.plot(training_loss_data_simple, label='Simple Linear Regression loss')
plt.plot(training_loss_data_ridge, label='Ridge Regression loss')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss by epoch Simple Linear and Ridge Regression')
plt.legend()
plt.show()