import torch 

def create_cnn(model,input_size,num_layers_cnn,out_activation,in_chan, out_chan, ks, stride_, padding_):

    model.append(torch.nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=ks, stride=stride_, padding=padding_))
    model.append(torch.nn.ReLU())

    for i in range(num_layers_cnn-1):
        model.append(torch.nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=ks, stride=stride_, padding=padding_))
        model.append(torch.nn.ReLU())

    model.append(torch.nn.Linear(in_features=in_chan*input_size*input_size, out_features=128))
    model.append(torch.nn.Softmax2d())


