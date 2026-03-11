### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Part_A_2
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            # YOUR CODE HERE
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    # YOUR CODE HERE
    def forward(self, x):
        return self.mlp_stack(x)

class CustomDataset(Dataset):
    # YOUR CODE HERE
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure y is a 2D tensor (column vector) for BCELoss
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def initialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test):
    # YOUR CODE HERE
    # Create instances of the CustomDataset
    train_dataset = CustomDataset(X_train_scaled, y_train)
    test_dataset = CustomDataset(X_test_scaled, y_test)

    # Initialize DataLoaders
    # Note: Training loader uses shuffle=True for better gradient descent
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader

# Part_A_3
class DynamicMLP(nn.Module):
    def __init__(self, input_dim, arch):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in arch:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp_stack = nn.Sequential(*layers)
    def forward(self, x): return self.mlp_stack(x)

def train(model, X_train_scaled, y_train2, X_val_scaled, y_val2, learning_rate):

    # YOUR CODE HERE
    train_loader, val_loader = initialise_loaders(X_train_scaled, y_train2, X_val_scaled, y_val2)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    loss_fn = nn.BCELoss()
    stopper = EarlyStopper(patience=5) # Using the provided EarlyStopper

    train_accuracies, train_losses = [], []
    test_accuracies, test_losses = [], []
    times = []

    for epoch in range(1, 201):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += ((outputs > 0.5).float() == batch_y).sum().item()
            total += batch_y.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vX, vy in val_loader:
                v_out = model(vX)
                val_loss += loss_fn(v_out, vy).item()
                val_correct += ((v_out > 0.5).float() == vy).sum().item()
                val_total += vy.size(0)

        avg_val_loss = val_loss / len(val_loader)
        test_losses.append(avg_val_loss)
        test_accuracies.append(val_correct / val_total)
        times.append(time.time() - start_time)

        # Early Stopping
        if stopper.early_stop(avg_val_loss):
            break

    return train_accuracies, train_losses, test_accuracies, test_losses, times

