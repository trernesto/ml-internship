import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 

class PolynomialRegression():
    def __init__(self, degree = 1):
        super(PolynomialRegression, self).__init__()
        rng = np.random.default_rng(42)
        self.degree = degree
        self.theta = rng.random(degree + 1)
        
    def forward(self, x):
        x = self.transform_X(x)
        out = x @ self.theta
        return out
    
    def backwards(self, alpha, x, error):
        x = self.transform_X(x)
        for j in range(self.theta.shape[0]):
            self.theta[j] = self.theta[j] - alpha * 2 * (error * x[:, j]).sum()/error.shape[0]
        
    def transform_X(self, X):
        X_transform = np.ones((X.shape[0], 1))
        for i in range(1, self.degree + 1):
            x_pow = np.power(X, i)
            X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis = 1)
        X_transform[:, 1:] = (X_transform[:, 1:] - np.mean(X_transform[:, 1:], axis = 0)) / np.std(X_transform[:, 1:], axis=0)
        return X_transform

def normalize_y(y):
    return (y - np.mean(y)) / np.std(y)
    
if __name__ == "__main__":
    # Data generation
    rng = np.random.default_rng(42)
    train_data_X = np.arange(100)
    train_data_Y = 7 * rng.random(100) + np.power(train_data_X - 50, 2) * rng.random(100)
    test_data_X = np.arange(100)
    test_data_Y = 7 * rng.random(100) + np.power(test_data_X - 50, 2) * rng.random(100)
    
    #normalization
    train_data_Y = normalize_y(train_data_Y)
    test_data_Y = normalize_y(test_data_Y)
    
    
    #learning rate
    alpha: float = 1e-1
    #number of steps
    num_epochs: int = 1000
    
    model = PolynomialRegression(degree = 2)
    
    for epoch in range(num_epochs):
        y_pred = model.forward(train_data_X)
        
        error = y_pred - train_data_Y
        
        model.backwards(alpha, train_data_X, error)
    
    y_pred = model.forward(test_data_X)
    
    # Data visualization
    plt.scatter(train_data_X, train_data_Y, c = 'blue', label = "Train data")
    plt.scatter(test_data_X, test_data_Y, c = 'red', label = "Test data")
    plt.plot(test_data_X, y_pred, c = 'green', label = 'predicted')
    plt.xlabel("X")
    plt.ylabel("data")
    #plt.xlim(right = 100)
    #plt.ylim(bottom = 0)
    #plt.ylim(top = 110)
    plt.legend()
    plt.show()