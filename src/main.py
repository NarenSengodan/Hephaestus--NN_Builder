import torch
import tensorflow_datasets as tfds
from CNN import create_cnn

def import_dataset():

    dataset_option = input("Which dataset do you want to import\n(Note: Tensorflow Datasets only)\n")
    dataset, info = tfds.load(dataset_option, with_info=True, as_supervised=True)

    x_train, x_test = dataset["train"], dataset["test"]
    return x_train, x_test

'''   train_dataset, test_dataset = tfds.as_numpy(dataset['train']), tfds.as_numpy(dataset['test'])

    x_train, y_train = train_dataset['features'], train_dataset['labels']   
    x_test, y_test = test_dataset['features'], test_dataset['labels']


    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)


    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, x_train, x_test, y_train, y_test'''

def CNN_main():

    x_train_tensor, x_test_tensor, x_train, x_test = import_dataset()

    input_prompt = input("Do you want to obtain input size from dataset? (y/n): ")
    if input_prompt.lower() == "y":
        input_size = x_train.shape[1:]
    else:
        input_size = None

    in_chan_prompt = input("Do you want to obtain number of in channels from dataset? (y/n): ")
    if in_chan_prompt.lower() == "y":
        in_chan = x_train_tensor.shape[-1]
    else:
        in_chan = None

    model = torch.nn.Sequential()
    
    if input_size:
        pass
    else:
        input_size = int(input("Enter input size: "))

    num_layers_cnn = int(input("Enter number of layers in CNN: "))

    if in_chan:
        pass
    else:
        in_chan = int(input("Enter number of input channels: "))

    out_chan = int(input("Enter number of output channels: "))
    ks = int(input("Enter kernel size: "))
    stride_ = int(input("Enter stride: "))
    padding_ = int(input("Enter padding: "))

    model = create_cnn(model,input_size,num_layers_cnn,in_chan, out_chan, ks, stride_, padding_)

    return model

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
    save_prompt = input("Do you want to save the model? (y/n): ")
    if save_prompt.lower() == "y":
        model_save()
    else:
        pass
    
if __name__ == "__main__":
    model = main()