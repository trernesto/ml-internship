import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def func(x: np.array, theta: np.array) -> np.array:
    return 1 / (1 + np.exp(-(x @ theta.T)))

def cost_function(y_pred: np.array, y_true: np.array) -> float:
    return 1/y_true.shape[0] * (y_pred - y_true)**2

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    # load the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    X = np.append(arr = np.ones([X.shape[0],1]).astype(int), values = X,axis=1)
    
    # split the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                       test_size=0.20,
                                       random_state=42)
    
    #learning rate
    alpha: float = 0.0001
    #number of steps
    N: int = 1000
    #random theta
    theta: np.array = rng.random(31)
    
    
    for i in range(N):
        y_pred: np.array = func(X_train, theta)
        for j in range(theta.shape[0]):
            theta[j] = theta[j] - 2 * alpha * ((y_pred - y_train) * X_train[:, j]).sum() / y_pred.shape[0]
        #eval
        y_pred: np.array = func(X_test, theta)
        if ((i + 1) % 50 == 0):
            print("iteration:", i + 1, "cost:", cost_function(y_pred, y_test).sum())
        
    y_pred = func(X_test, theta) > 0.5
    acc = accuracy_score(y_test, y_pred)
    print("my logistic Regression model accuracy (in %):", acc*100)
    
    # LogisticRegression
    clf = LogisticRegression(random_state=0, max_iter=50)
    clf.fit(X_train, y_train)
    
    # Prediction
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("Sklearn logistic Regression model accuracy (in %):", acc*100)