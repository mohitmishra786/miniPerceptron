#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1
#define MAX_EPOCHS 10000
#define EPSILON 1e-4

typedef struct {
    double **weights;
    double *biases;
    double *outputs;
    int inputs;
    int neurons;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} MLP;

double random_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x);
}

Layer create_layer(int inputs, int neurons) {
    printf("Creating layer with %d inputs and %d neurons\n", inputs, neurons);
    Layer layer;
    layer.inputs = inputs;
    layer.neurons = neurons;
    layer.weights = (double **)malloc(neurons * sizeof(double *));
    layer.biases = (double *)malloc(neurons * sizeof(double));
    layer.outputs = (double *)malloc(neurons * sizeof(double));

    if (!layer.weights || !layer.biases || !layer.outputs) {
        fprintf(stderr, "Memory allocation failed in create_layer\n");
        exit(1);
    }

    for (int i = 0; i < neurons; i++) {
        layer.weights[i] = (double *)malloc(inputs * sizeof(double));
        if (!layer.weights[i]) {
            fprintf(stderr, "Memory allocation failed for weights in create_layer\n");
            exit(1);
        }
        for (int j = 0; j < inputs; j++) {
            layer.weights[i][j] = random_weight();
        }
        layer.biases[i] = random_weight();
    }

    printf("Layer created successfully\n");
    return layer;
}

MLP create_mlp() {
    printf("Creating MLP\n");
    MLP mlp;
    mlp.num_layers = 3;
    mlp.layers = (Layer *)malloc(mlp.num_layers * sizeof(Layer));

    if (!mlp.layers) {
        fprintf(stderr, "Memory allocation failed in create_mlp\n");
        exit(1);
    }

    mlp.layers[0] = create_layer(INPUT_NEURONS, HIDDEN_NEURONS);
    mlp.layers[1] = create_layer(HIDDEN_NEURONS, HIDDEN_NEURONS);
    mlp.layers[2] = create_layer(HIDDEN_NEURONS, OUTPUT_NEURONS);

    printf("MLP created successfully\n");
    return mlp;
}

void forward_propagation(MLP *mlp, double *input) {
    printf("Starting forward propagation\n");
    for (int l = 0; l < mlp->num_layers; l++) {
        Layer *layer = &mlp->layers[l];
        double *prev_outputs = l == 0 ? input : mlp->layers[l - 1].outputs;
        int prev_size = l == 0 ? INPUT_NEURONS : mlp->layers[l - 1].neurons;

        for (int i = 0; i < layer->neurons; i++) {
            double sum = layer->biases[i];
            for (int j = 0; j < prev_size; j++) {
                sum += prev_outputs[j] * layer->weights[i][j];
            }
            layer->outputs[i] = sigmoid(sum);
        }
    }
    printf("Forward propagation completed\n");
}

void backward_propagation(MLP *mlp, double *input, double target) {
    printf("Starting backward propagation\n");
    // Allocate memory for deltas
    double **deltas = (double **)malloc(mlp->num_layers * sizeof(double *));
    for (int l = 0; l < mlp->num_layers; l++) {
        deltas[l] = (double *)malloc(mlp->layers[l].neurons * sizeof(double));
        if (!deltas[l]) {
            fprintf(stderr, "Memory allocation failed in backward_propagation\n");
            exit(1);
        }
    }

    // Output layer delta
    Layer *output_layer = &mlp->layers[mlp->num_layers - 1];
    double output_error = target - output_layer->outputs[0];
    deltas[mlp->num_layers - 1][0] = output_error * sigmoid_derivative(output_layer->outputs[0]);

    // Hidden layers deltas
    for (int l = mlp->num_layers - 2; l >= 0; l--) {
        Layer *layer = &mlp->layers[l];
        Layer *next_layer = &mlp->layers[l + 1];

        for (int i = 0; i < layer->neurons; i++) {
            double error = 0.0;
            for (int j = 0; j < next_layer->neurons; j++) {
                error += next_layer->weights[j][i] * deltas[l + 1][j];
            }
            deltas[l][i] = error * sigmoid_derivative(layer->outputs[i]);
        }
    }

    // Update weights and biases 
    for (int l = 0; l < mlp->num_layers; l++) {
        Layer *layer = &mlp->layers[l];
        double *prev_outputs = l == 0 ? input : mlp->layers[l - 1].outputs;
        int prev_size = l == 0 ? INPUT_NEURONS : mlp->layers[l - 1].neurons;

        for (int i = 0; i < layer->neurons; i++) {
            for (int j = 0; j < prev_size; j++) {
                layer->weights[i][j] += LEARNING_RATE * deltas[l][i] * prev_outputs[j];
            }
            layer->biases[i] += LEARNING_RATE * deltas[l][i];
        }
    }

    // Free deltas
    for (int l = 0; l < mlp->num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
    printf("Backward propagation completed\n");
}

double train(MLP *mlp, double inputs[][INPUT_NEURONS], double *targets, int num_samples) {
    printf("Starting training\n");
    double total_error = 0.0;
    for (int i = 0; i < num_samples; i++) {
        forward_propagation(mlp, inputs[i]);
        backward_propagation(mlp, inputs[i], targets[i]);
        double error = targets[i] - mlp->layers[mlp->num_layers - 1].outputs[0];
        total_error += error * error;
    }
    printf("Training completed\n");
    return total_error / num_samples;
}


void free_mlp(MLP *mlp) {
    printf("Freeing MLP memory\n");
    for (int l = 0; l < mlp->num_layers; l++) {
        for (int i = 0; i < mlp->layers[l].neurons; i++) {
            free(mlp->layers[l].weights[i]);
        }
        free(mlp->layers[l].weights);
        free(mlp->layers[l].biases);
        free(mlp->layers[l].outputs);
    }
    free(mlp->layers);
    printf("MLP memory freed\n");
}

int main() {
    printf("Starting main function\n");
    srand((unsigned int)time(NULL));

    MLP mlp = create_mlp();

    double inputs[4][INPUT_NEURONS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};

    printf("Training MLP for XOR problem...\n");

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double error = train(&mlp, inputs, targets, 4);
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Error = %f\n", epoch, error);
        }
        if (error < EPSILON) {
            printf("Converged at epoch %d\n", epoch);
            break;
        }
    }

    printf("\nTesting MLP:\n");
    for (int i = 0; i < 4; i++) {
        forward_propagation(&mlp, inputs[i]);
        printf("Input: (%g, %g), Output: %g, Target: %g\n",
               inputs[i][0], inputs[i][1],
               mlp.layers[mlp.num_layers - 1].outputs[0], targets[i]);
    }

    free_mlp(&mlp);
    printf("Program completed successfully\n");
    return 0;
}
