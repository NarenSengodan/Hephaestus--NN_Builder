import torch
import torch.nn as nn

def create_dnn(input_size, hidden_size, num_layers):
    model = nn.Sequential()
    
    for i in range(num_layers):
        model.add_module(f'fc{i}', nn.Linear(input_size, hidden_size))
        input_size = hidden_size
    
    model.add_module(f'fc{num_layers}', nn.Linear(hidden_size, 10))
    model.add_module(f'softmax', nn.LogSoftmax(dim=1))
        
    return model