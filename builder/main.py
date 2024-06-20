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
    input_prompt = input("Do you want to obtain input size and in channels from dataset? (y/n): ")
    if input_prompt.lower() == "y":
        input_size = (x_train.element_spec[0].shape[1:])
        print(input_size[0])
        
    else:
        input_size = None

    model = torch.nn.Sequential()
    
    if input_size:
        pass
    else:
        input_size = int(input("Enter input size: "))
        in_chan = int(input("Enter number of input channels: "))
    num_layers_cnn = int(input("Enter number of layers in CNN: "))
    out_chan = int(input("Enter number of output channels: "))
    ks = int(input("Enter kernel size: "))
    stride_ = int(input("Enter stride: "))
    padding_ = int(input("Enter padding: "))

    model = create_cnn(model,input_size,num_layers_cnn,in_chan, out_chan, ks, stride_, padding_)

    return model

def train_model(model,x_train,x_test):

    training_loader = torch.utils.data.DataLoader(x_train, batch_size=32, shuffle=True)
    evaluation_loader = torch.utils.data.DataLoader(x_test, batch_size=32, shuffle=False)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01) 

    num_epochs = int(input("Number of epochs: "))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    model.to(device) 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i,(input_, labels) in enumerate(training_loader):
            input_, labels = input_.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(input_)
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
            for (input_, labels) in evaluation_loader:
                input_, labels = input_.to(device), labels.to(device)
                output = model(input_)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
        print(f"Epoch: {epoch+1}, Accuracy: {correct/total}")

def model_save(model):
    file_path = input("Enter file path:")
    torch.save(model.state_dict(), file_path)
    print("Model saved")

def main():

    x_train, x_test = import_dataset()
    print("Welcome to Hephaestus ")
    
    train_prompt = input("Do you want to train the model? (y/n): ")
    save_prompt = input("Do you want to save the model? (y/n): ")

    
    model = CNN_main(x_train, x_test)
    print(model)

    if save_prompt.lower() == "y":
        model_save(model)
        print("Model saved")
    else:
        print("Model not saved")
        pass
        
    if train_prompt.lower() == "y":
        train_model(model,x_train,x_test)
        print("Model trained")
    else:
        print("Model not trained")
    

    
if __name__ == "__main__":
    model = main()