import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class RegressionPlotter:
    #initializing, setting the model to passed argument
    def __init__(self, model):
        self.model = model

    #plot function uses polymorphism, by identifying the shape of the feature array that is passed we can identify the users intentions 
    def plot(self, X : np.ndarray, y : np.ndarray, y_pred=None, sequence=False):
        
        #if no predicted values were passed, use the linear regression model to predict these values
        if y_pred is None:
            y_pred = self.model.predict(X)

        #number of features can be found by looking at the second entry of the shape of X
        num_features = X.shape[1]

        #if number of features is 1 plot a single regression plot
        if num_features == 1:
            # Simple linear regression
            plt.scatter(X, y, color='blue')
            plt.plot(X, y_pred, color='red')
            plt.xlabel('Feature')
            plt.ylabel('Target')
            plt.title('Linear Regression Plot')
            plt.show()

        #if number of features is 2, the user likely wants a 3-D plot with corresponding regression plane
        elif num_features == 2 and sequence == False:
            # Multiple linear regression with 2 features
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #add the datapoints:
            ax.scatter(X[:, 0], X[:, 1], y, color='blue')
            
            # Create a meshgrid for the plane
            x0, x1 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 10),
                                 np.linspace(X[:, 1].min(), X[:, 1].max(), 10))
            y_pred_plane = self.model.predict(np.column_stack((x0.ravel(), x1.ravel()))).reshape(x0.shape)

            #give namex to the axes, plot the regression plane and set a title
            ax.plot_surface(x0, x1, y_pred_plane, alpha=0.3, color='red')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Target')
            plt.title('Multiple Linear Regression Plot')
            plt.show()

        #more than 2 features? multiple single regression plots showing the datapoints and a regression line trained on all datapoints.
        else:
            for i in range(num_features):
                plt.figure(figsize=(8, 6))
                plt.scatter(X[:, i], y, color='blue', label='Actual Data')

                # Calculating regression values for plotting
                reg_values = self.model.coefficients[0] + self.model.coefficients[i + 1] * X[:, i]
                
                plt.plot(X[:, i], reg_values, color='red', label='Regression Line')
                plt.xlabel(f'Feature {i + 1}')
                plt.ylabel('Target')
                plt.title(f'Feature {i + 1} vs Target')
                plt.legend()
                plt.show()
