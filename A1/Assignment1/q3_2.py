import numpy as np
from q3_1 import compute_gradient_ridge,compute_gradient_simple
from q1_1 import rmse

# Solves regression tasks using full-batch gradient descent.
# Parameters:
        # X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        # y (np.ndarray): Target values of shape (n_samples,).
        # reg_type (str): Type of regression ('simple' for simple linear, 'ridge' for ridge regression).
        # hyperparameter (float): Regularization parameter, used only for ridge regression.
        # learning_rate (float): Learning rate for gradient descent.
        # num_epochs (int): Number of epochs for gradient descent.
# Returns:
        # w (np.ndarray): Final weights after gradient descent optimization.
        # b (float): Final bias after gradient descent optimization.
        # loss_data (list): List of loss values at each epoch.
def gradient_descent_regression(X, y, reg_type='simple', hyperparameter=0.0, learning_rate=0.01, num_epochs=100):

    np.random.seed(42)  
    n, m = X.shape
    w = np.random.normal(0, 1, size=(m, 1))  #w = np.zeros((m,1)) to start from the "zero" point

    b = 0
    loss_data = []

    for _ in range(num_epochs):
        if reg_type == 'simple':
            grad_w, grad_b = compute_gradient_simple(X, y, w, b)
        elif reg_type == 'ridge':
            grad_w, grad_b = compute_gradient_ridge(X, y, w, b, hyperparameter)

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        loss_data.append(computeLoss(X, y, reg_type, w, b, hyperparameter))

    return w, b, loss_data

def computeLoss(X, y, reg_type, w, b, hyperparameter):
    y_pred = np.dot(X, w) + b
    if reg_type == 'simple':
        loss = np.mean((y_pred - y) ** 2) # 2/n ? or just *2 cuz mean divides by n
    elif reg_type == 'ridge':
        loss = np.mean((y_pred - y) ** 2) + hyperparameter * np.sum(w ** 2)  # 2/n ?

    return loss