from sklearn.datasets import load_iris 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn 
from torchinfo import summary 

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.softmax(out, dim = 1)
        return out

if __name__ == "__main__":
    # Preparing dataset
    iris: tuple = load_iris()
    
    X = torch.tensor(iris.data, dtype = torch.float32)
    y = torch.tensor(iris.target, dtype = torch.long)
    
    # Normalization
    mean = torch.mean(X, dim = 0)
    std = torch.std(X, dim = 0)
    X = (X - mean) / std
    
    # Split the dataset into training and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # To Torch dataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Dataloader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    model = LogisticRegression(input_size = 4, num_classes = 3)
    #print(summary(model, input_size=(32,4)))
    
    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    
    # Training cycle
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            
            # Forward
            out = model(inputs)
            loss = criterion(out, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            out = model(inputs)
            _, pred = torch.max(out.data, 1)
            
            total +=  labels.size(0)
            correct += (pred == labels).sum().item()
        
        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
