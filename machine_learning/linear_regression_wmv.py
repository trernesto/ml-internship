import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error

def cost_function(y_pred: np.array, y_true: np.array) -> float:
    return 1/y_true.shape[0] * (y_pred - y_true)**2 

def func(X: np.array, theta: np.array):
    return X @ theta.T

if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv('Real-estate1.csv')
    df.drop('No', inplace=True, axis = 1)

    #print(df.head())
    #print(df.columns)

    X: pd.DataFrame = df.drop('Y house price of unit area', axis = 1)
    #preprocess
    for column in X.columns: 
        X[column] = X[column]  / X[column].abs().max() 
    
    y: pd.DataFrame = df['Y house price of unit area']
    X: np.array = np.append(arr = np.ones([X.shape[0],1]).astype(int), values = X,axis=1)
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rng = np.random.default_rng(42)
    #learning rate
    alpha: float = 5e-4
    #number of steps
    N: int = 5000
    #random theta
    theta: np.array = rng.random(X_train.shape[1])
    for i in range(N):
        y_pred: np.array = func(X_train, theta)
        for j in range(theta.shape[0]):
            theta[j] = theta[j] - alpha * 2 * ((y_pred - y_train) * X_train[:, j]).sum()/y_pred.shape[0]
           
    
    prediction: np.array = func(X_test, theta)
    print('My lr mean_squared_error : ', mean_squared_error(y_test, prediction)) 
    print('My lr mean_absolute_error : ', mean_absolute_error(y_test, prediction)) 
    
    # creating a regression model 
    model = LinearRegression() 
    
    # fitting the model 
    model.fit(X_train, y_train) 
    
    # making predictions 
    prediction = model.predict(X_test) 
    
    # model evaluation 
    print('Sklearn mean_squared_error : ', mean_squared_error(y_test, prediction)) 
    print('Sklearn mean_absolute_error : ', mean_absolute_error(y_test, prediction)) 
            