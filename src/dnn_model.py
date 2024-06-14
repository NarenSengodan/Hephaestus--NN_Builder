import torch
import torch.nn as nn

def create_dnn(input_size, hidden_size, num_layers):
    model = nn.Sequential()
    
    for i in range(num_layers):
        model.add_module(f'fc{i}', nn.Linear(input_size, hidden_size))
        input_size = hidden_size
        
    return model