import torch 
from torch import nn 
 

def newline():
    print()

# Hyperparameters
batch_size = 256
ce_loss = nn.CrossEntropyLoss()
learning_rate = 0.01
weight_decay = 0.0001
momentum = 0.9
epoch = 10


