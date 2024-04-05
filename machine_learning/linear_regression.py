import matplotlib.pyplot as plt
import numpy as np


def func(x: np.array, theta: np.array) -> np.array:
    #y = theta_0 + theta_1 * x - linear function
    return theta[0] + theta[1] * x

def cost_function(y_pred: np.array, y_true: np.array) -> float:
    return 1/y_true.shape[0] * (y_pred - y_true)**2

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    train_data_X = np.arange(100)
    train_data_Y = 7 * rng.random(100) + train_data_X
    test_data_X = np.arange(100)
    test_data_Y = 7 * rng.random(100) + test_data_X
    
    #learning rate
    alpha: float = 0.0001
    #number of steps
    N: int = 100
    #random theta
    theta: np.array = rng.random(2)
    
    
    for i in range(N):
        y_pred: np.array = func(train_data_X, theta)
        theta[0] = theta[0] - 2 * alpha * (y_pred - train_data_Y).sum() / y_pred.shape[0]
        theta[1] = theta[1] - 2 * alpha * ((y_pred - train_data_Y) * train_data_X).sum() / y_pred.shape[0]
        print("iteration:", i + 1, "cost:", cost_function(y_pred, test_data_Y).sum())
        
    y_pred = func(test_data_X, theta)
    plt.scatter(train_data_X, train_data_Y, c = 'blue', label = "Train data")
    plt.scatter(test_data_X, test_data_Y, c = 'red', label = "Test data")
    plt.plot(test_data_X, y_pred, c = 'green', label = 'predicted')
    plt.xlabel("X")
    plt.ylabel("data")
    plt.xlim(right = 100)
    plt.ylim(bottom = 0)
    plt.ylim(top = 110)
    plt.legend()
    plt.show()