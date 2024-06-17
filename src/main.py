import torch
import tensorflow_datasets as tfds
from CNN import create_cnn

def import_dataset():

    dataset_option = input("Which dataset do you want to import\n(Note: Tensorflow Datasets only)\n")
    dataset, info = tfds.load(dataset_option, with_info=True, as_supervised=True)

    x_train, x_test = dataset["train"], dataset["test"]
    return x_train, x_test

def CNN_main():

    x_train, x_test = import_dataset()
    in_chan = None
    input_prompt = input("Do you want to obtain input size from dataset? (y/n): ")
    if input_prompt.lower() == "y":
        input_size = x_train.element_spec.shape[1:]
    else:
        input_size = None

    model = torch.nn.Sequential()
    
    if input_size:
        pass
    else:
        input_size = int(input("Enter input size: "))

    num_layers_cnn = int(input("Enter number of layers in CNN: "))

    in_chan = int(input("Enter number of input channels: "))
    out_chan = int(input("Enter number of output channels: "))
    ks = int(input("Enter kernel size: "))
    stride_ = int(input("Enter stride: "))
    padding_ = int(input("Enter padding: "))

    model = create_cnn(model,input_size,num_layers_cnn,in_chan, out_chan, ks, stride_, padding_)

    return model

def train_model(model,x_train,x_test):

    training_loader = torch.utils.data.DataLoader(x_train, batch_size=32, shuffle=True)
    evaluation_loader = torch.utils.data.DataLoader(x_test, batch_size=32, shuffle=False)

    return training_loader, evaluation_loader
    

def model_save():
    file_path = input("Enter file path:")
    torch.save(model.state_dict(), file_path)
    print("Model saved")

def main():

    
    print("Welcome to Hephaestus ")
    layer_type = input("Enter layer type:\n")

    if layer_type.lower() == "cnn":
        model = CNN_main()
        print(model)
        return model
    
    adam = torch.optimizers.Adam
    loss = torch.nn.CrossEntropyLoss
    metrics = ["accuracy"]
  
    save_prompt = input("Do you want to save the model? (y/n): ")
    if save_prompt.lower() == "y":
        model_save()
    else:
        pass
    
if __name__ == "__main__":
    model = main()