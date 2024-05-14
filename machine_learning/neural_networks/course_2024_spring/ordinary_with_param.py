from collections import OrderedDict
import torch
from torch import autograd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
        
    def forward(self, x, c):
        x = torch.cat((x, c), 1)
        out = self.lin1(x)
        out = nn.functional.sigmoid(out)
        out = self.lin2(out)
        out = nn.functional.sigmoid(out)
        out = self.encoder(out)
        out = nn.functional.sigmoid(out)
        out = self.lin3(out)
        return out

def grad(y, xs, create_graph=True, retain_graph=True):
    return autograd.grad(y.sum(),
                         xs,
                         create_graph=create_graph,
                         retain_graph=retain_graph)

def pde_residual(y_pred, x):
    dydx = grad(y_pred, x)
    return dydx[0]

def func(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return c * (x + 1) / torch.exp(x)


def loss_func(model: nn.Module, x: torch.tensor, c: torch.tensor):
    #! c is a constant of our diff equation
    criterion = nn.MSELoss()
    #x = torch.squeeze(x)
    #equilibrium condition
    dydx = pde_residual(model(x, c), x)
    
    equilibrium = dydx + x * model(x, c) / (x + 1)
    
    # Boundary
    X_BOUNDARY_left = 0 * torch.ones(c.shape[0], dtype=torch.float32).view(c.shape[0], 1)
    Y_BOUNDARY_left = c * (X_BOUNDARY_left + 1)/np.exp(X_BOUNDARY_left)
    
    X_BOUNDARY_right = 10 * torch.ones(c.shape[0], dtype=torch.float32).view(c.shape[0], 1)
    Y_BOUNDARY_right = c * (X_BOUNDARY_right + 1)/np.exp(X_BOUNDARY_right)
    
    boundary_loss_left = model(X_BOUNDARY_left, c) - Y_BOUNDARY_left
    boundary_loss_right = model(X_BOUNDARY_right, c) - Y_BOUNDARY_right
    loss = criterion(equilibrium, torch.zeros_like(equilibrium)) +\
            criterion(boundary_loss_left, torch.zeros_like(boundary_loss_left)) +\
            criterion(boundary_loss_right, torch.zeros_like(boundary_loss_right))
                
    return loss
    
if __name__ == "__main__":
    # Data generation
    size = 16
    train_X = torch.linspace(0, 10, size, requires_grad=True).view(size, 1)
    lower_bound = 1
    upper_bound = 5
    random = torch.randint(lower_bound, upper_bound, (size, 1))
    train_C = random * torch.ones((size, 1)).view(size, 1)
    #train_X = torch.cat((train_X, train_C), 1)
    train_y = func(train_X, train_C)
    
    size_test = 101
    test_X = torch.linspace(0, 10, size_test).view(size_test, 1)
    test_C = 2.5 * torch.ones(size_test).view(size_test, 1)
    test_y = func(test_X, test_C)
    
    #train_X, val_X, train_y, val_y = train_test_split(X, y, test_size= 0.1, random_state=42)
    
    batch_size = 16
    
    train_dataset = TensorDataset(train_X, train_C, train_y)
    test_dataset = TensorDataset(test_X, test_C, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = NNmodel()
    
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # Train cycle
    num_epochs = 8_000 #640
    
    for epoch in range(num_epochs):
        for x_batch, c_batch, y_batch in train_loader:
            loss = loss_func(model, x_batch, c_batch)
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
    
    train_plot_x = np.array([])
    train_plot_y_true = np.array([])
    train_plot_y_pred = np.array([])
    with torch.no_grad():
        train_loss = torch.tensor([0], dtype = torch.float32)
        for x_batch, c_batch, y_batch in train_loader:
            y_pred = model(x_batch, c_batch)
            train_plot_x = np.append(train_plot_x, x_batch)
            train_plot_y_true = np.append(train_plot_y_true, y_batch)
            train_plot_y_pred = np.append(train_plot_y_pred, y_pred)
            train_loss = torch.max(loss, torch.max(y_pred - y_batch))
        print("train loss: {:.3f}".format(train_loss.item()))
        
    test_plot_x = np.array([])
    test_plot_y = np.array([])
    with torch.no_grad():
        loss_mse = 0
        loss = torch.tensor([0], dtype = torch.float32)
        for x_test, c_test, y_test in test_loader:
            y_pred = model(x_test, c_test)
            test_plot_x = np.append(test_plot_x, x_test)
            test_plot_y = np.append(test_plot_y, y_pred)
            loss_mse += torch.abs(y_pred - y_test).sum()
            loss = torch.max(loss, torch.max(torch.abs(y_pred - y_test)))
        loss_mse = loss_mse / size_test
        print("test loss abs: {:.3f}".format(loss_mse))
        print("max diff loss: {:.3f}".format(loss.item()))
        
    
    #plt.plot(test_X, test_y, label = 'data', c = 'g')
    #plt.scatter(test_plot_x, test_plot_y, c = 'b', linewidths=0.1, label = 'test')
    #plt.scatter(train_plot_x, train_plot_y_pred, c = 'r', linewidths=0.1, label = 'train pred')
    #plt.scatter(train_plot_x, train_plot_y_true, c = 'g', linewidths=0.1, label = 'train true')
    #plt.legend()
    #plt.show()