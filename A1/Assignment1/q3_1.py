import numpy as np

# In this section, we will optimize our Regression model with gradient descent instead of the 
# closed-form solution. As the first step, we should calculate the gradients. Implement functions 
# that compute the gradient for both simple and ridge regression
# Compute the gradients of the loss function with respect to w and b for simple linear regression.
# Args:
#     X (np.ndarray): Input features matrix of shape (n, m).
#     y (np.ndarray): Target vector of shape (n, ).
#     w (np.ndarray): Weights vector of shape (m, ).
#     b (float): Bias term.
# Returns:
#     grad_w (np.ndarray): Gradient with respect to weights.
#     grad_b (float): Gradient with respect to bias.
def compute_gradient_simple(X, y, w, b):

    y_pred = np.dot(X, w) + b
    n = X.shape[0]
    X_transpose = X.T
    error = y - y_pred
    grad_w = -2/n * np.dot(X_transpose, error)
    grad_b = -2/n * np.sum(error)

    return grad_w, grad_b


# Compute the gradients of the loss function with respect to w and b for ridge regression.
# Args:
#     X (np.ndarray): Input features matrix of shape (n, m).
#     y (np.ndarray): Target vector of shape (n, ).
#     w (np.ndarray): Weights vector of shape (m, ).
#     b (float): Bias term.
#     lambda_reg (float): Regularization parameter.
# Returns:
#     grad_w (np.ndarray): Gradient with respect to weights.
#     grad_b (float): Gradient with respect to bias.
def compute_gradient_ridge(X, y, w, b, lambda_reg):

    y_pred = np.dot(X, w) + b
    X_transpose = X.T
    n = X.shape[0]
    error = y - y_pred
    grad_w = -2/n * np.dot(X_transpose, error) + 2*lambda_reg * w
    grad_b = -2/n * np.sum(error)

    return grad_w, grad_b