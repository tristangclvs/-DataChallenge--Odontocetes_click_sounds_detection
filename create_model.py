import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"
print(f"Using {DEVICE} device")

# load the data
def load_data(data_path: str) -> tuple:
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    # convert lists to np.ndarray
    X = torch.Tensor(data["abs_stft"]).to(DEVICE)
    y = torch.Tensor(data["labels"]).to(torch.long).to(DEVICE)
    
    return X, y


# create the model
def create_model():
    # Define the model architecture
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    
    # Define the loss function
    criterion = nn.BCELoss()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer

model, criterion, optimizer = create_model()

BATCH_SIZE = 10
NUM_EPOCHS = 50

x_train, y_train = load_data("data.json")

# create the data loaders and Tensor Dataset
dataset    = torch.utils.data.TensorDataset(x_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

losses = []
for e in range(NUM_EPOCHS):

    epoch_loss = []

    for x_i, y_i in dataloader:  # for each batch of data
        # Forward pass
        y_pred = model(x_i)
        
        # training loop !
        loss = criterion(y_pred, y_i).cpu() # compute loss on cpu bc cannot use mps on Mac with numpy & detach
        
        if e % 100 == 99:
            print(e, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save loss values somewhere to plot the learning curve
        loss = loss.detach().numpy()
        epoch_loss.append(loss)
        
        print(f"Epoch {e} - Loss {np.mean(epoch_loss):.3f}", end="\r")

    losses.append(np.mean(epoch_loss))

