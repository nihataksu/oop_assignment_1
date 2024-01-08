from src.multiple_linear_regression import MultipleLinearRegressor
import numpy as np


class RidgeRegressor(MultipleLinearRegressor):
    def __init__(self, default_intercept=0, default_coefficients=None, ridge_lambda=1.0):
        super().__init__(default_intercept, default_coefficients)
        self.ridge_lambda = ridge_lambda

    def _ridge_gradient(self, X, y, w):
        n = len(y)
        error = y - np.dot(X, w[1:]) - w[0]
        gradient = -2/n * np.dot(X.T, error) + 2 * self.ridge_lambda * w[1:]
        return gradient

    def train(self, X: np.ndarray, y: np.ndarray, learning_rate=0.01, num_iterations=1000):
        try:
            self._features = X
            self._target = y

            X_with_intercept = np.c_[np.ones(X.shape[0]), X]
            w = np.zeros(X_with_intercept.shape[1])

            for _ in range(num_iterations):
                gradient = self._ridge_gradient(X_with_intercept, y, w)
                w -= learning_rate * gradient

            self._intercept = w[0]
            self._coefficients = w[1:]

        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion error: The design matrix X is singular, indicating multicollinearity in the predictors.")