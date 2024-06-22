import torch
import tensorflow_datasets as tfds
import tensorflow as tf
from CNN import create_cnn
import numpy as np
from tqdm import tqdm

def import_dataset():
    dataset_option = input("Which dataset do you want to import\n(Note: Tensorflow Datasets only)\n")
    try:
        dataset, info = tfds.load(dataset_option, with_info=True, as_supervised=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

    x_train, x_test = dataset["train"], dataset["test"]
    num_classes = info.features['label'].num_classes
    return x_train, x_test, num_classes

def tfds_to_torch(dataset):
    data = [(image, label) for image, label in tfds.as_numpy(dataset)]
    return [(torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)) for image, label in data]

def CNN_main(x_train, x_test, num_classes):
    if x_train is None or x_test is None:
        print("Invalid dataset. Exiting...")
        return None

    input_prompt = input("Do you want to obtain input size and in channels from dataset? (y/n): ")
    if input_prompt.lower() == "y":
        input_size = x_train.element_spec[0].shape[1:]
        in_chan = input_size[-1] if len(input_size) == 3 else 1  # Assuming grayscale if channel info is not present
        input_size = input_size[0] if len(input_size) == 3 else input_size[0]
        print(f"Input size: {input_size}, Input channels: {in_chan}")
    else:
        input_size = int(input("Enter input size: "))
        in_chan = int(input("Enter number of input channels: "))

    model = torch.nn.Sequential()
    
    num_layers_cnn = int(input("Enter number of layers in CNN: "))
    out_chan = num_classes
    print(f"Number of output channels (classes): {out_chan}")
    ks = int(input("Enter kernel size: "))
    stride_ = int(input("Enter stride: "))
    padding_ = int(input("Enter padding (integer): "))

    model = create_cnn(model, input_size, num_layers_cnn, in_chan, out_chan, ks, stride_, padding_)

    # Add a flatten layer before the fully connected layers
    model.add_module("flatten", torch.nn.Flatten())

    # Calculating the output size after convolutional layers
    conv_output_size = calculate_conv_output_size(input_size, num_layers_cnn, ks, stride_, padding_)
    fc_input_size = conv_output_size * conv_output_size * in_chan

    # Adding fully connected layers
    model.add_module("fc", torch.nn.Linear(fc_input_size, 128))
    model.add_module("output", torch.nn.Linear(128, out_chan))

    return model

def calculate_conv_output_size(input_size, num_layers_cnn, ks, stride_, padding_):
    size = input_size
    for _ in range(num_layers_cnn):
        size = (size - ks + 2 * padding_) // stride_ + 1
    return size

def train_model(model, x_train, x_test):
    x_train = tfds_to_torch(x_train)
    x_test = tfds_to_torch(x_test)

    training_loader = torch.utils.data.DataLoader(x_train, batch_size=32, shuffle=True)
    evaluation_loader = torch.utils.data.DataLoader(x_test, batch_size=32, shuffle=False)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = int(input("Number of epochs: "))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(training_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, (input_, labels) in enumerate(train_bar):
            input_ = input_.permute(0, 3, 1, 2).to(device)  # Permute dimensions
            labels = labels.to(device)
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
        eval_bar = tqdm(evaluation_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for input_, labels in eval_bar:
                input_ = input_.permute(0, 3, 1, 2).to(device)  # Permute dimensions
                labels = labels.to(device)
                output = model(input_)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch: {epoch+1}, Accuracy: {correct/total}")

def model_save(model):
    file_path = input("Enter file path:")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved at {file_path}")

def main():
    x_train, x_test, num_classes = import_dataset()

    if x_train is None or x_test is None or num_classes is None:
        return

    model = CNN_main(x_train, x_test, num_classes)
    if model is None:
        return

    print(model)

    save_prompt = input("Do you want to save the model? (y/n): ")
    if save_prompt.lower() == "y":
        model_save(model)

    train_prompt = input("Do you want to train the model? (y/n): ")
    if train_prompt.lower() == "y":
        train_model(model, x_train, x_test)

if __name__ == "__main__":
    main()
