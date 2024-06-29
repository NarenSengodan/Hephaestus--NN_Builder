```markdown
# Hephaestus CNN Model Builder

Hephaestus CNN Model Builder is a Python script that allows users to build and train Convolutional Neural Network (CNN) models using PyTorch with TensorFlow Datasets.

## Features

- **Dataset Import**: Imports datasets from TensorFlow Datasets (TFDS) for training and testing.
- **Dynamic Model Creation**: Builds a CNN model dynamically based on user inputs for layer configurations.
- **Training**: Trains the CNN model using the imported dataset with options for specifying epochs and GPU utilization.
- **Model Saving**: Optionally saves the trained model to a specified file path.

## Requirements

- Python
- PyTorch
- TensorFlow Datasets (tfds)
- GPU (optional, for faster training)

## Usage

1. **Clone the Repository**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/NarenSengodan/hephaestus.cnn.git
   cd hephaestus.cnn
   ```

2. **Install Dependencies**: Install the required Python libraries.
   ```bash
   pip install torch tensorflow-datasets
   ```

3. **Run the Script**: Execute the main Python script to build and train your CNN model.
   ```bash
   python main.py
   ```

4. **Follow the Prompts**:
   - Choose the dataset from TensorFlow Datasets that you want to use.
   - Specify the CNN architecture by entering the number of layers, input and output channel numbers, kernel size, stride, and padding.
   - Optionally train the model by specifying the number of epochs.
   - Optionally save the trained model to a specified file path.
   - Monitor Training: During training, the script will display loss values and accuracy metrics for each epoch.

## Example Usage

Here's an example of how to use the Hephaestus CNN Model Builder script:

```bash
Which dataset do you want to import
(Note: TensorFlow Datasets only)
mnist
Do you want to obtain input size from dataset? (y/n): y
Enter number of layers in CNN: 2
Enter number of input channels: 1
Enter number of output channels: 32
Enter kernel size: 3
Enter stride: 1
Enter padding: 1
Number of epochs: 10
Do you want to train the model? (y/n): y
Do you want to save the model? (y/n): y
Enter file path: models/mnist_cnn_model.pth
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## Acknowledgments

- Built with PyTorch and TensorFlow Datasets.
- Inspired by the need for a flexible CNN model builder with TensorFlow dataset integration.
```
