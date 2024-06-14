import torch
import torch.nn as nn

def create_cnn(num_layers, activation_layers):
    model = nn.Sequential()
    
    for i in range(num_layers):
        if i == 0:
            model.add_module(f'conv{i}', nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1))
        else:
            model.add_module(f'conv{i}', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        
        model.add_module(f'activation{i}', activation_layers[i])
    
    model.add_module(f'pool', nn.MaxPool2d(kernel_size=2, stride=2))
    model.add_module(f'flatten', nn.Flatten())
    model.add_module(f'fc', nn.Linear(32 * 14 * 14, 10))
    model.add_module(f'softmax', nn.LogSoftmax(dim=1))
    
    
    return model