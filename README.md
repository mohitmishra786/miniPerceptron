# MLP Line Classification
![](https://komarev.com/ghpvc/?username=mohitmishra786&color=green)
This project implements a Multi-Layer Perceptron (MLP) from scratch in C to classify points relative to a curve. The MLP learns to distinguish between points above and below a configurable curve, which can be linear, quadratic, or sinusoidal.

## Features

- **Customizable MLP Architecture:**
    - Configure the number of hidden layers.
    - Specify the number of neurons in each hidden layer.
- **Activation Function Selection:**
    - Choose from Sigmoid, ReLU, or Tanh activation functions for hidden layers.
    - Output layer uses Sigmoid activation for binary classification.
- **Regularization:**
    - Optionally enable L2 regularization to prevent overfitting.
- **Visualization with Cairo:**
    - Generates PNG images of the target curve, the MLP's learned decision boundary, and the training data points.
    - Images are organized into subdirectories based on the activation function used.
- **Command-line Interface:**
    - Control hyperparameters (hidden layers, neurons, activation, regularization, learning rate) from the command line.
    - Specify the type of curve for classification.

## Requirements

- **GCC Compiler:** For compiling the C code.
- **Cairo Graphics Library:** For generating the visualization plots.
- **libpng:**  For saving images in PNG format.

## Installation

1. **Install Required Libraries:**
   On Debian/Ubuntu based systems, you can install these dependencies using:
   ```bash
   sudo apt-get install build-essential libcairo2-dev libpng-dev
   ```
   For other systems, please refer to the package manager of your distribution.

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/miniPerceptron.git
   cd miniPerceptron
   ```

## Compilation
Compile the project using the following command:
```bash
gcc mlp.c -o mlp -I/usr/include/cairo -lcairo -lpng -lm
```

## Usage
Run the compiled program with various command-line arguments to customize the MLP and the classification task:
```bash
./mlp [--hidden-layers <value>] [--hidden-neurons <value>] [--activation <value>] [--regularization <value>] [--learning-rate <value>] [--l2-lambda <value>] [--curve <value>]
```

**Arguments:**
* hidden-layers <value>: Number of hidden layers in the MLP (default: 2).
* hidden-neurons <value>: Number of neurons in each hidden layer (default: 4).
* activation <value>: Activation function to use ("sigmoid", "relu", or "tanh"; default: "sigmoid").
* regularization <value>: Whether to use L2 regularization (1 for true, 0 for false; default: 0).
* learning-rate <value>: Learning rate for the backpropagation algorithm (default: 0.1).
* l2-lambda <value>: Lambda value for L2 regularization (default: 0.01).
* curve <value>: The type of curve to use ("linear", "quadratic", or "sine"; default: "linear").

**Examples:**
1. Run with ReLU activation, 3 hidden layers, 5 neurons per layer, and L2 regularization:
  ```bash
  ./mlp --activation relu --hidden-layers 3 --hidden-neurons 5 --regularization 1
  ```

2. Run with Tanh activation and a quadratic curve, using default values for other parameters:
  ```bash
  ./mlp --activation tanh --curve quadratic
  ```

## Output
The program will print the training progress (epoch number, error, time) to the console. It will generate PNG image files in the images directory, organized into subdirectories named "sigmoid," "relu," and "tanh," based on the chosen activation function.

## Future Improvements

* Implement mini-batch gradient descent for better performance on larger datasets
* Add support for multiple output neurons to handle multi-class classification
* Implement early stopping to prevent overfitting
* Add dropout regularization as an alternative to L2 regularization
* Implement adaptive learning rate techniques (e.g., Adam, RMSprop)
* Add support for loading and saving trained models
* Implement k-fold cross-validation for more robust model evaluation
* Add more activation functions (e.g., LeakyReLU, ELU)
* Implement a simple GUI for visualizing the training process in real-time
* Add support for categorical features through one-hot encoding
* Implement feature scaling (e.g., normalization, standardization) for input data
* Add support for custom loss functions
* Implement batch normalization for faster training and better generalization
* Add support for generating and saving multiple plots to compare different configurations
* Implement a simple API to use the trained model for predictions on new data

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is open source and available under the [MIT License](README#License).
