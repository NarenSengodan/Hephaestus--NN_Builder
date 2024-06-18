import torch
import tensorflow_datasets as tfds
from CNN import create_cnn

def import_dataset():

    dataset_option = input("Which dataset do you want to import\n(Note: Tensorflow Datasets only)\n")
    dataset, info = tfds.load(dataset_option, with_info=True, as_supervised=True)

    x_train, x_test = dataset["train"], dataset["test"]
    return x_train, x_test

def CNN_main(x_train, x_test):

    
    in_chan = None
    input_prompt = input("Do you want to obtain input size from dataset? (y/n): ")
    if input_prompt.lower() == "y":
        input_size = x_train.element_spec[0].shape[1:]
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

    model = create_cnn(model,tuple(input_size),num_layers_cnn,in_chan, out_chan, ks, stride_, padding_)

    return model

def train_model(model,x_train,x_test):

    training_loader = torch.utils.data.DataLoader(x_train, batch_size=32, shuffle=True)
    evaluation_loader = torch.utils.data.DataLoader(x_test, batch_size=32, shuffle=False)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),learning_rate=0.001) 

    num_epochs = int(input("Number of epochs: "))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    model.to(device) 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i,(input, labels) in enumerate(training_loader):
            input, labels = input.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss_value = loss(output, labels)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()
            if i % 100 == 99:
                print(f"Epoch: {epoch+1}, Loss: {running_loss/100}")
                running_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (input, labels) in evaluation_loader:
                input, labels = input.to(device), labels.to(device)
                output = model(input)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
        print(f"Epoch: {epoch+1}, Accuracy: {correct/total}")

def model_save():
    file_path = input("Enter file path:")
    torch.save(model.state_dict(), file_path)
    print("Model saved")

def main():

    x_train, x_test = import_dataset()
    print("Welcome to Hephaestus ")
    layer_type = input("Enter layer type:\n")

    if layer_type.lower() == "cnn":
        model = CNN_main(x_train, x_test)
        print(model)
        return model
    
    train_prompt = input("Do you want to train the model? (y/n): ")
    if train_prompt.lower() == "y":
        train_model(model,x_train,x_test)
    else:
        pass
    
    save_prompt = input("Do you want to save the model? (y/n): ")
    if save_prompt.lower() == "y":
        model_save()
    else:
        pass
    
if __name__ == "__main__":
    model = main()