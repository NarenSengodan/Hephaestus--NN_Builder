import torch
import tensorflow_datasets as tfds
from CNN import create_cnn

def import_dataset():

    dataset_option = input("Which dataset do you want to import\n(Note: Tensorflow Datasets only) ")
    dataset, info = tfds.load(dataset_option, with_info=True, as_supervised=True)

    x_train, y_train = tfds.as_numpy(dataset['train'])
    x_test, y_test = tfds.as_numpy(dataset['test'])

    x_train_tensor = torch.tensor(x_train)
    y_train_tensor = torch.tensor(y_train)
    x_test_tensor = torch.tensor(x_test)
    y_test_tensor = torch.tensor(y_test)

    input_prompt = input("Do you want to obtain input size from dataset? (y/n): ")
    if input_prompt.lower() == "y":
        input_size = x_train[0]
    else:
        None

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, input_size

def main():

    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, input_size = import_dataset()

    layer_type = input("Enter layer type: ")

    if layer_type.lower() == "cnn":
        model = CNN_main()
        print(model)
        return model

def CNN_main(input_size):

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

def model_save():
    file_path = input("Enter file path:")
    torch.save(model.state_dict(), file_path)
    print("Model saved")
    
if __name__ == "__main__":
    model = main()

    save_prompt = input("Do you want to save the model? (y/n): ")
    if save_prompt.lower() == "y":
        model_save()
    else:
        pass
