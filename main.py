from src import multiple_linear_regression
from src import regression_plotter 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

def main():
    
    #Any variable or function/method preceded by a underscore, like: "_variable" will be "public" and accessible to the user, though as we're using python this of course makes no difference
    np.random.seed(42) 
    # -------------------------------------------------------------------------- #
    
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    np.array(X).astype(float)
    np.array(y).astype(float)

    # model = multiple_linear_regression.MultipleLinearRegressor()
    # model.train(X,y)
    # y_pred = model.predict(X)
    
    # plotter = regression_plotter.RegressionPlotter(model)

    #regression plotter example
    # plotter.plot(X,y,y_pred)
    #two features MultiLinearRegression plotter
    # print("Xshape")
    # print(X.shape)
    # plotter.plot()
    
    
    diabetes_reduced = X[:, :2]
    model = multiple_linear_regression.MultipleLinearRegressor()
    model.train(diabetes_reduced, y)
    y_pred = model.predict(diabetes_reduced)
 
    plotter = regression_plotter.RegressionPlotter(model)
    plotter.plot(diabetes_reduced, y, y_pred)


if __name__ == "__main__":
    main()