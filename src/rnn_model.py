import torch
import torch.nn as nn

def create_rnn(input_size, hidden_size, num_layers, bidirectional):
    model = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    
    return model