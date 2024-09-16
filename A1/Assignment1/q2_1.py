import numpy as np

# Optimizes MSE fit of y = Xw with L2 regularization.
# Args:
#       y (np.ndarray): Salary, Numpy array of shape [observations, 1].
#       X (np.ndarray): Features (e.g., experience, test_score), Numpy array of shape [observations, features].
#       hyperparameter (float): Lambda used in L2 regularization.
# Returns:    
#       np.ndarray: Optimal parameters (w), Numpy array of shape [features, 1].
def ridge_regression_optimize(y: np.ndarray, X: np.ndarray, hyperparameter: float) -> np.ndarray:
    # Ridge Regression closed-form solution : w = (X^T * X + λ * I)^-1 * X^T * y
    X_transpose = X.T
    I = np.eye(X.shape[1])
    left_part = np.linalg.inv(np.dot(X_transpose, X) + hyperparameter * I)
    right_part = np.dot(X_transpose, y)
    w = np.dot(left_part, right_part)

    return w


