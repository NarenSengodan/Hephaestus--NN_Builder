import torch.nn as nn
from cnn_model import create_cnn
from rnn_model import create_rnn
from dnn_model import create_dnn

def main():
    # CNN
    num_cnn_layers = int(input("Enter the number of CNN layers: "))
    activation_layers_cnn = []
    
    for i in range(num_cnn_layers):
        layer_type = input(f"Enter the type of activation layer for CNN layer {i+1}: ")
        
        if layer_type == 'relu':
            activation_layers_cnn.append(nn.ReLU())
        elif layer_type == 'sigmoid':
            activation_layers_cnn.append(nn.Sigmoid())
        elif layer_type == 'tanh':
            activation_layers_cnn.append(nn.Tanh())
        else:
            print(f"Invalid activation layer type: {layer_type}")
            return
    
    cnn_model = create_cnn(num_cnn_layers, activation_layers_cnn)
    print("CNN Model:")
    print(cnn_model)
    
    # RNN
    input_size_rnn = int(input("Enter the input size for RNN: "))
    hidden_size_rnn = int(input("Enter the hidden size for RNN: "))
    num_layers_rnn = int(input("Enter the number of RNN layers: "))
    bidirectional_rnn = input("Enter 'true' for bidirectional RNN, 'false' otherwise: ")
    
    if bidirectional_rnn.lower() == 'true':
        bidirectional_rnn = True
    else:
        bidirectional_rnn = False
    
    rnn_model = create_rnn(input_size_rnn, hidden_size_rnn, num_layers_rnn, bidirectional_rnn)
    print("RNN Model:")
    print(rnn_model)
    
    # DNN
    input_size_dnn = int(input("Enter the input size for DNN: "))
    hidden_size_dnn = int(input("Enter the hidden size for DNN: "))
    num_layers_dnn = int(input("Enter the number of DNN layers: "))
    
    dnn_model = create_dnn(input_size_dnn, hidden_size_dnn, num_layers_dnn)
    print("DNN Model:")
    print(dnn_model)

if __name__ == "__main__":
    main()