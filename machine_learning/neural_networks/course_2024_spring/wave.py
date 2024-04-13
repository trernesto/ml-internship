import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

class NNmodel(nn.Module):
    def __init__(self, in_layer = 2, hidden_layer = 8, encoder_layer = 8, out_layer = 1) -> None:
        super(NNmodel, self).__init__()
        self.lin1 = nn.Linear(in_layer, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, encoder_layer)
        self.encoder = nn.Linear(encoder_layer, encoder_layer)
        self.lin3 = nn.Linear(encoder_layer, out_layer)
        
    def forward(self, x):
        out = self.lin1(x)
        out = nn.functional.sigmoid(out)
        out = self.lin2(out)
        out = nn.functional.sigmoid(out)
        out = self.encoder(out)
        out = nn.functional.sigmoid(out)
        out = self.lin3(out)
        return out

def func(x: np.array, t: np.array) -> np.array:
    return 2 * np.sin(2*np.pi*x) * np.sin(1*np.pi*t)

def heatmap2d(x: np.array, t: np.array, y: np.array):
    plt.scatter(t, x, c=y, cmap='plasma_r')
    plt.colorbar()
    
def get_unique_train_points(number_of_points: int, size: int):
    #returns idx
    right_border = size**2
    points = random.sample(range(0, right_border), number_of_points)
    x_points = [(int)(point / size) for point in points]
    t_points = [(int)(point % size) for point in points]
    return np.array(x_points), np.array(t_points)
    
if __name__ == "__main__":
    # Data generation
    # Train data
    size = 101
    train_X = torch.linspace(0, 1, size).view(size, 1)
    train_t = torch.linspace(0, 2, size).view(size, 1)
    train_X, train_t = np.meshgrid(train_X, train_t, sparse=True)
    train_y = func(train_X, train_t)
    x_idx, t_idx = get_unique_train_points(4, size)
    train_X = train_X[:, x_idx]
    train_t = train_t[t_idx, :]
    
    #size_test = 201
    #test_X = torch.linspace(0, 10, size_test).view(size_test, 1)
    #test_t = torch.linspace(0, 10, size_test).view(size_test, 1)
    #test_data = np.array([[x0, y0] for x0 in test_X for y0 in test_t])
    #test_y = func(test_data)
    
    '''
    #X = torch.tensor(X, dtype = torch.float32)
    #y = torch.tensor(y, dtype = torch.float32)
    
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size= 0.2, random_state=42)
    
    batch_size = 16
    
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = NNmodel()
    
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # Train cycle
    num_epochs = 10_000 #640
    
    # Boundary
    X_BOUNDARY_left = torch.tensor([0], dtype=torch.float32)
    Y_BOUNDARY_left = torch.tensor([1], dtype=torch.float32)
    
    X_BOUNDARY_right = torch.tensor([10], dtype=torch.float32)
    Y_BOUNDARY_right = torch.tensor([11/np.exp(10)], dtype=torch.float32)
    
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            boundary_loss_left = model(X_BOUNDARY_left) - Y_BOUNDARY_left
            boundary_loss_right = model(X_BOUNDARY_right) - Y_BOUNDARY_right
            loss = criterion(y_batch, y_pred) +\
                criterion(boundary_loss_left, torch.zeros_like(boundary_loss_left)) +\
                criterion(boundary_loss_right, torch.zeros_like(boundary_loss_right))
            
            optim.zero_grad()
            loss.backward()
            optim.step()
    
    with torch.no_grad():
        mse_loss = 0
        for x_val, y_val in val_loader:
            y_pred = model(x_val)
            mse_loss += mean_squared_error(y_pred, y_val)
        print("val loss: {:.2f}".format(mse_loss))
        
    with torch.no_grad():
        mse_loss = 0
        for x_test, y_test in test_loader:
            y_pred = model(x_test)
            plt.scatter(x_test, y_pred, c = 'b', linewidths=0.1)
            mse_loss += mean_squared_error(y_pred, y_test)
        print("test loss: {:.2f}".format(mse_loss))
     
    '''   
        
    #heatmap2d(train_data, y)
    #plt.plot(X, y, label = 'train data', c = 'g')
    #plt.legend()
    #plt.show()
    #X = train_data[0:size, 0].reshape(-1, size)
    #t = train_data[0::size, 1].reshape(size, -1)
    #print(X)
    #print(t)
    #fig = plt.figure(figsize=(10,6))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, t, y, cmap='viridis')
    #ax.set_xlabel('X axis')
    #ax.set_ylabel('time axis')
    #ax.set_zlabel('function axis')
    #plt.scatter(X, t, c=y, cmap='viridis')
    #plt.xlabel('t')
    #plt.ylabel('x')
    #plt.colorbar()
    #plt.show()