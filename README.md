# MLP Line Classification

This project implements a Multi-Layer Perceptron (MLP) from scratch in C to classify points relative to a line. The MLP learns to distinguish between points above and below the line y = 0.5x + 0.1.

## Features

- Custom MLP implementation with configurable layers and neurons
- Training data generation
- Forward and backward propagation
- Visualization of results using Cairo graphics library

## Requirements

- GCC compiler
- Cairo graphics library
- libpng

## Installation

1. Install the required libraries:

```bash
sudo apt-get install libcairo2-dev libpng-dev
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/miniPerceptron.git
cd miniPerceptron
```

## Compilation

Compile the project using the following command:

```bash
gcc -o mlp_classification mlp.c -lm -lcairo -lpng
```

## Usage

Run the compiled program:

```bash
./mlp_classification
```

The program will train the MLP and generate plot images showing the classification results at different epochs.

## Output

The program generates PNG images named `mlp_plot_epoch_X.png`, where X is the epoch number. These images show the true line, the MLP's approximation, and the classified points.

## Future Improvements

1. Implement additional activation functions (e.g., ReLU, tanh)
2. Add regularization techniques to prevent overfitting
3. Extend the model to classify points relative to more complex curves
4. Implement a command-line interface for customizing MLP parameters
5. Optimize performance for larger datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
