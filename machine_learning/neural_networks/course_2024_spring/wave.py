import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def func(data: np.array) -> np.array:
    x = data[:, 0]
    t = data[:, 1]
    return (x * t) + np.sin(x + t) - (1 - np.cosh(t)) * np.exp(x)

def heatmap2d(data: np.ndarray, y: np.array):
    #data[0] = x, data[1] = t
    plt.scatter(data[:, 1], data[:, 0], c=y, cmap='plasma_r')
    plt.colorbar()

if __name__ == "__main__":
    # Data generation
    size = 51
    X = torch.linspace(0, 1, size).view(size, 1)
    t = torch.linspace(0, 1, size).view(size, 1)
    train_data = np.array([[x0, y0] for x0 in X for y0 in t])
    y = func(train_data)
    
    size_test = 201
    test_X = torch.linspace(0, 10, size_test).view(size_test, 1)
    test_t = torch.linspace(0, 10, size_test).view(size_test, 1)
    test_data = np.array([[x0, y0] for x0 in test_X for y0 in test_t])
    test_y = func(test_data)
    
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
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(train_data[:, 1], train_data[:, 0], y, cmap='viridis')
    plt.show()