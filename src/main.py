import torch
import torch.nn as nn
from cnn_model import create_cnn
from dnn_model import create_dnn


model = torch.nn.Sequential()

model_ask = input('Enter model type (cnn/dnn): ')
if model_ask == 'cnn':
    cnn_num = int(input('Enter number of cnn layers: '))
    activation_layers = []

    for i in range(cnn_num):
        activation_layers.append(nn.ReLU())

    model = create_cnn(cnn_num, activation_layers)

elif model_ask == 'dnn':
    dnn_num = int(input('Enter number of dnn layers: '))
    activation_layers = []

    for i in range(dnn_num):
        activation_layers.append(nn.ReLU())

    model = create_dnn(784, 256, dnn_num)

else:
    print('Invalid model type')
    exit()

print(model)