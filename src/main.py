import torch
from CNN import create_cnn

def main():

    layer_type = input("Enter layer type: ")

    if layer_type.lower() == "cnn":
        model = CNN_main()
        print(model)
        return model

def CNN_main():

    model = torch.nn.Sequential()
    input_size = int(input("Enter input size: "))
    num_layers_cnn = int(input("Enter number of layers in CNN: "))
    in_chan = int(input("Enter number of input channels: "))
    out_chan = int(input("Enter number of output channels: "))
    ks = int(input("Enter kernel size: "))
    stride_ = int(input("Enter stride: "))
    padding_ = int(input("Enter padding: "))

    model = create_cnn(model,input_size,num_layers_cnn,in_chan, out_chan, ks, stride_, padding_)

    return model

def model_save():
    file_path = input("Enter file path:")
    torch.save(model, file_path)
    print("Model saved")
    
if __name__ == "__main__":
    model = main()

    save_prompt = input("Do you want to save the model? (y/n): ")
    if save_prompt.lower() == "y":
        model_save()
    else:
        pass
