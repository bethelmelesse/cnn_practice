import torch 
from torch import nn 
 

def newline():
    print()

# Hyperparameters
batch_size = 64
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3

