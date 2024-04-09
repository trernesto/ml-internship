import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary 
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.model_selection import train_test_split 

class simpleNetwork(nn.Module):
    def __init__(self, input_nodes = 16, hidden_size = 16, output_nodes=10) -> None:
        super(simpleNetwork, self).__init__()
        self.lin1 = nn.Linear(input_nodes, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_nodes)

    def forward(self, x):
        out = self.lin1(x)
        out = self.lin2(out)
        out = nn.functional.softmax(out, dim = 1)
        return out
    
    
if __name__ == "__main__":
    # Import data
    digits = load_digits()
    X = torch.tensor(digits.data, dtype = torch.float32)
    y = torch.tensor(digits.target, dtype = torch.long)
    
    # Normalization
    #X = (X - torch.mean(X, dim = 0)) / torch.std(X, dim = 0)
    
    # Split -> DataSet -> DataLoader
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size= 0.2, random_state=42)
    
    batch_size = 16
    
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model init
    model = simpleNetwork(input_nodes=64, output_nodes=10)
    optimization = torch.optim.Adam(model.parameters(), lr = 3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train cycle
    num_epochs = 100
    
    for epoch in range(num_epochs):
        for x_train, y_true in train_loader:
            y_pred = model(x_train)
            
            loss = criterion(y_pred, y_true)
            
            optimization.zero_grad()
            loss.backward()
            optimization.step()
        
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
    # Inference
    with torch.no_grad():
        correct = 0
        total = 0
        for x_test, y_test in test_loader:
            y_pred = model(x_test)
            _, predicted = torch.max(y_pred.data, 1)
            
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
        print("Test Accuracy: {:.2f}%".format(correct/total * 100))
    
    