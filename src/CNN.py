import torch 

def create_cnn(model,input_size,num_layers_cnn,in_chan, out_chan, ks, stride_, padding_):

    
    model.append(torch.nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=ks, stride=stride_, padding=padding_))
    model.append(torch.nn.ReLU())

    for i in range(num_layers_cnn-1):
        model.append(torch.nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=ks, stride=stride_, padding=padding_))
        model.append(torch.nn.ReLU())

    Pool_layer_inp = input("\n1.MaxPool2d\n2.AvgPool2d\n3.AdaptiveAvgPool2d\nEnter your choice for pooling layer: ")

    if Pool_layer_inp == 1:
        model.append(torch.nn.MaxPool2d(kernel_size=ks, stride=None, padding=padding_))
    elif Pool_layer_inp == 2:
        model.append(torch.nn.AvgPool2d(kernel_size=ks, stride=None, padding=padding_))
    elif Pool_layer_inp == 3:
        model.append(torch.nn.AdaptiveAvgPool2d((1,1)))

    model.append(torch.nn.Linear(in_features=in_chan*input_size*input_size, out_features=128))
    model.append(torch.nn.Softmax2d())

    return model


#Example:
'''model = torch.nn.Sequential()
create_cnn(model,28,2,1,16,3,1,1)
print(model)'''


