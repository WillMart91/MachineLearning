import numpy as np
import q2_1
import q1_1
from typing import List, Tuple

# Perform k-fold cross-validation to find the best hyperparameter for Ridge Regression.
# Args:
#     k_folds (int): Number of folds to use.
#     hyperparameters (List[float]): List of floats containing the hyperparameter values to search.
#     X (np.ndarray): Numpy array of shape [observations, features].
#     y (np.ndarray): Numpy array of shape [observations, 1].
# Returns:
#     best_hyperparam (float): Value of the best hyperparameter found.
#     best_mean_squared_error (float): Best mean squared error corresponding to the best hyperparameter.
#     mean_squared_errors (List[float]): List of mean squared errors for each hyperparameter.
def cross_validation_linear_regression(k_folds: int, hyperparameters: List[float],
                                       X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:

    n = X.shape[0]
    size_of_folds = n / k_folds
    mean_squared_errors = []
    best_hyperparam = None
    best_mean_squared_error = np.inf
    
    for hyperparam in hyperparameters:
        mse = 0

        for i in range(k_folds):
            start_index = int(np.round(size_of_folds * i))
            end_index = int(np.round(size_of_folds * (i + 1)))

            validation_pos = list(range(start_index, end_index))
            train_pos = list(set(range(n)) - set(validation_pos))

            X_train = X[train_pos]
            y_train = y[train_pos]
            X_val = X[validation_pos]
            y_val = y[validation_pos]

            w = q2_1.ridge_regression_optimize(y_train, X_train, hyperparam)
            y_hat = np.dot(X_val, w)
            mse += q1_1.mse(y_val, y_hat)

        mse /= k_folds
        mean_squared_errors.append(mse)

        if mse < best_mean_squared_error:
            best_hyperparam = hyperparam
            best_mean_squared_error = mse

    return best_hyperparam, best_mean_squared_error, mean_squared_errors
