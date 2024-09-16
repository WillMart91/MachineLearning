import numpy as np

# Returns the design matrix with an all one column appended
# Args:     X (np.ndarray): Numpy array of shape [observations, num_features]
# Returns:  np.ndarray: Numpy array of shape [observations, num_features + 1]
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    bias = np.ones((X.shape[0], 1))
    X_bias = np.hstack((bias, X))
    return X_bias


# Computes $y = Xw$
# Args:     X (np.ndarray): Numpy array of shape [observations, features]
#           w (np.ndarray): Numpy array of shape [features, 1]
# Returns:  
#           np.ndarray: Numpy array of shape [observations, 1]
def linear_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = np.dot(X, w)
    return y

# Optimizes MSE fit of $y = Xw$
# Args:     y (np.ndarray): Numpy array of shape [observations, 1]
#           X (np.ndarray): Numpy array of shape [observations, features]
# Returns:  Numpy array of shape [features, 1]
def linear_regression_optimize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    # closed form solution : w = (X^T * X)^-1 * X^T * y
    X_transpose = X.T
    left_part = np.linalg.inv(np.dot(X_transpose, X))
    right_part = np.dot(X_transpose, y)
    w = np.dot(left_part, right_part)
    return w


# Evaluate the RMSE between actual and predicted values.
# Parameters:
#           y (list or np.array): The actual values.
#           y_hat (list or np.array): The predicted values.
# Returns:
#           float: The RMSE value.
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:    
    return np.sqrt(mse(y, y_hat))


# Evaluate the MSE between actual and predicted values.
# Parameters:
#           y (list or np.array): The actual values.
#           y_hat (list or np.array): The predicted values.
# Returns:
#           float: The MSE value.
def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.array(y)  
    y_hat = np.array(y_hat)
    mse_err = np.mean(np.square(y_hat - y))
    return mse_err
