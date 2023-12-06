import numpy as np

class MultipleLinearRegressor:
    #initializer, sets the intercept, coefficients, features and target to corresponding values. features and target are initialized as None, as they have not yet been calculated
    def __init__(self, default_intercept=0, default_coefficients=None):
        self._intercept = default_intercept
        self._coefficients = default_coefficients
        self._features = None
        self._target = None

    #train takes features data array x and target data array y. It then calculates the
    def train(self, X : np.ndarray, y : np.ndarray):
        try:
            # X is a numpy array with each row representing an observation and each column representing a feature
            # y is a 1D numpy array containing the ground truth values
            self._features = X
            self._target = y

            #calculate X with the intercept
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]

            # Calculate coefficients using the given 10th equation:
            # (X^T * X)^(-1) * X^T * y
            X_transpose = np.transpose(X_with_intercept)
            X_transpose_dot_X = np.dot(X_transpose, X_with_intercept)
            inverse_X_transpose_dot_X = np.linalg.inv(X_transpose_dot_X)
            X_transpose_dot_y = np.dot(X_transpose, y)
            coefficients = np.dot(inverse_X_transpose_dot_X, X_transpose_dot_y)

            # Set intercept and coefficients to calculated value(s)
            self._intercept = coefficients[0]
            self._coefficients = coefficients[1:]
            
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion error: The design matrix X is singular, indicating multicollinearity in the predictors.")

    def predict(self, X : np.ndarray):
        # X is a 2D numpy array with each row representing an observation and each column representing a feature

        # Add a column of ones to X for the intercept term
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]

        # Calculate predictions using nump.dot, giving the dot product between X with the intercept and a concatenation of the intercept and coefficient(s)
        _predictions = np.dot(X_with_intercept, np.concatenate([[self._intercept], self._coefficients]))

        return _predictions

