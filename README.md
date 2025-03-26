# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![Screenshot 2025-03-21 135602](https://github.com/user-attachments/assets/d831b970-71c2-4acd-ac6c-2b4700625982)


## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.

### STEP 2:
Input Layer: Number of neurons = features. Hidden Layers: 2 layers with ReLU activation. Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.

### STEP 3:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

### STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

### STEP 5:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

### STEP 6:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

## PROGRAM

### Name: DIVYA DHARSHINI R 
### Register Number: 212223040042

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # 4 output classes
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x       

```
```python
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

## Dataset Information

![image](https://github.com/user-attachments/assets/aeb47fcd-cc20-43d2-b152-6eddb5ce9ea8)


## OUTPUT
### Confusion Matrix
![image](https://github.com/user-attachments/assets/b073d6fa-0580-4cbb-9543-cb384f0c80fd)


### Classification Report

![image](https://github.com/user-attachments/assets/4d456cd4-0905-4c8a-a8a2-be8020133dc9)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/bf41fadf-2d68-48ce-a3aa-6ab749026c46)


## RESULT
The development of a neural network classification model for the given dataset is executed successfully.
