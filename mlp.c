#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cairo.h>
#include <string.h>

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1
#define MAX_EPOCHS 10000
#define EPSILON 1e-3
#define NUM_SAMPLES 10000
#define TEST_SAMPLES 100
#define PLOT_SIZE 20
#define PLOT_WIDTH 800
#define PLOT_HEIGHT 800

#define SIGMOID 0
#define RELU 1
#define TANH 2
#define L2_LAMBDA 0.01

typedef struct {
    double **weights;
    double *biases;
    double *outputs;
    int inputs;
    int neurons;
    int activation_function;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
    int use_regularization;
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

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double tanh_activation(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    return 1 - x * x;
}

double (*activation_functions[])(double) = {sigmoid, relu, tanh_activation};
double (*activation_derivatives[])(double) = {sigmoid_derivative, relu_derivative, tanh_derivative};

double true_line(double x);
double mlp_line(double x);
void forward_propagation(MLP *mlp, double *input);

MLP* g_mlp;

double true_line(double x) {
    return 0.5 * x + 0.1;
}

double mlp_line(double x) {
    double input[2] = {x, 0};
    forward_propagation(g_mlp, input);
    double y = g_mlp->layers[g_mlp->num_layers - 1].outputs[0];
    return (y - 0.5) * 2; // Scale the output to [-1, 1]
}

Layer create_layer(int inputs, int neurons, int activation_function) {
    Layer layer;
    layer.inputs = inputs;
    layer.neurons = neurons;
    layer.activation_function = activation_function;
    layer.weights = (double **)malloc(neurons * sizeof(double *));
    layer.biases = (double *)malloc(neurons * sizeof(double));
    layer.outputs = (double *)malloc(neurons * sizeof(double));

    for (int i = 0; i < neurons; i++) {
        layer.weights[i] = (double *)malloc(inputs * sizeof(double));
        for (int j = 0; j < inputs; j++) {
            layer.weights[i][j] = random_weight();
        }
        layer.biases[i] = random_weight();
    }

    return layer;
}

MLP create_mlp(int activation_function, int use_regularization) {
    MLP mlp;
    mlp.num_layers = 3;
    mlp.layers = (Layer *)malloc(mlp.num_layers * sizeof(Layer));
    mlp.use_regularization = use_regularization;

    mlp.layers[0] = create_layer(INPUT_NEURONS, HIDDEN_NEURONS, activation_function);
    mlp.layers[1] = create_layer(HIDDEN_NEURONS, HIDDEN_NEURONS, activation_function);
    mlp.layers[2] = create_layer(HIDDEN_NEURONS, OUTPUT_NEURONS, SIGMOID);  // Output layer always uses sigmoid

    return mlp;
}

void forward_propagation(MLP *mlp, double *input) {
    for (int l = 0; l < mlp->num_layers; l++) {
        Layer *layer = &mlp->layers[l];
        double *prev_outputs = l == 0 ? input : mlp->layers[l - 1].outputs;
        int prev_size = l == 0 ? INPUT_NEURONS : mlp->layers[l - 1].neurons;

        for (int i = 0; i < layer->neurons; i++) {
            double sum = layer->biases[i];
            for (int j = 0; j < prev_size; j++) {
                sum += prev_outputs[j] * layer->weights[i][j];
            }
            layer->outputs[i] = activation_functions[layer->activation_function](sum);
        }
    }
}

void backward_propagation(MLP *mlp, double *input, double target) {
    double **deltas = (double **)malloc(mlp->num_layers * sizeof(double *));
    for (int l = 0; l < mlp->num_layers; l++) {
        deltas[l] = (double *)malloc(mlp->layers[l].neurons * sizeof(double));
    }

    Layer *output_layer = &mlp->layers[mlp->num_layers - 1];
    double output_error = target - output_layer->outputs[0];
    deltas[mlp->num_layers - 1][0] = output_error * activation_derivatives[output_layer->activation_function](output_layer->outputs[0]);

    for (int l = mlp->num_layers - 2; l >= 0; l--) {
        Layer *layer = &mlp->layers[l];
        Layer *next_layer = &mlp->layers[l + 1];

        for (int i = 0; i < layer->neurons; i++) {
            double error = 0.0;
            for (int j = 0; j < next_layer->neurons; j++) {
                error += next_layer->weights[j][i] * deltas[l + 1][j];
            }
            deltas[l][i] = error * activation_derivatives[layer->activation_function](layer->outputs[i]);
        }
    }

    for (int l = 0; l < mlp->num_layers; l++) {
        Layer *layer = &mlp->layers[l];
        double *prev_outputs = l == 0 ? input : mlp->layers[l - 1].outputs;
        int prev_size = l == 0 ? INPUT_NEURONS : mlp->layers[l - 1].neurons;

        for (int i = 0; i < layer->neurons; i++) {
            for (int j = 0; j < prev_size; j++) {
                double weight_update = LEARNING_RATE * deltas[l][i] * prev_outputs[j];
                if (mlp->use_regularization) {
                    weight_update -= LEARNING_RATE * L2_LAMBDA * layer->weights[i][j];
                }
                layer->weights[i][j] += weight_update;
            }
            layer->biases[i] += LEARNING_RATE * deltas[l][i];
        }
    }

    for (int l = 0; l < mlp->num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

double train(MLP *mlp, double inputs[][INPUT_NEURONS], double *targets, int num_samples) {
    double total_error = 0.0;
    for (int i = 0; i < num_samples; i++) {
        forward_propagation(mlp, inputs[i]);
        backward_propagation(mlp, inputs[i], targets[i]);
        double error = targets[i] - mlp->layers[mlp->num_layers - 1].outputs[0];
        total_error += error * error;
    }
    return total_error / num_samples;
}

void free_mlp(MLP *mlp) {
    for (int l = 0; l < mlp->num_layers; l++) {
        for (int i = 0; i < mlp->layers[l].neurons; i++) {
            free(mlp->layers[l].weights[i]);
        }
        free(mlp->layers[l].weights);
        free(mlp->layers[l].biases);
        free(mlp->layers[l].outputs);
    }
    free(mlp->layers);
}

void generate_point(double *x, double *y) {
    *x = ((double)rand() / RAND_MAX) * 2 - 1;
    *y = ((double)rand() / RAND_MAX) * 2 - 1;
}

int is_above_line(double x, double y) {
    return y > 0.5 * x + 0.1;
}

void create_plot(MLP *mlp, double inputs[][INPUT_NEURONS], double *targets, int num_samples, const char *filename) {
    g_mlp = mlp;

    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, PLOT_WIDTH, PLOT_HEIGHT);
    cairo_t *cr = cairo_create(surface);

    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);

    cairo_translate(cr, PLOT_WIDTH / 2, PLOT_HEIGHT / 2);
    cairo_scale(cr, PLOT_WIDTH / 2, -PLOT_HEIGHT / 2);

    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_set_line_width(cr, 0.005);
    cairo_move_to(cr, -1, 0);
    cairo_line_to(cr, 1, 0);
    cairo_move_to(cr, 0, -1);
    cairo_line_to(cr, 0, 1);
    cairo_stroke(cr);

    cairo_set_source_rgba(cr, 0, 0, 1, 0.5);
    for (int i = 0; i < num_samples; i++) {
        cairo_arc(cr, inputs[i][0], inputs[i][1], 0.01, 0, 2 * M_PI);
        cairo_fill(cr);
    }

    cairo_set_source_rgb(cr, 0, 0, 0);
    cairo_move_to(cr, -1, true_line(-1));
    cairo_line_to(cr, 1, true_line(1));
    cairo_stroke(cr);

    cairo_set_source_rgb(cr, 1, 0, 0);
    cairo_move_to(cr, -1, mlp_line(-1));
    for (double x = -0.95; x <= 1; x += 0.05) {
        cairo_line_to(cr, x, mlp_line(x));
    }
    cairo_stroke(cr);

    cairo_surface_write_to_png(surface, filename);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}

int main() {
    srand((unsigned int)time(NULL));

    double inputs[NUM_SAMPLES][INPUT_NEURONS];
    double targets[NUM_SAMPLES];

    // Generating training data
    for (int i = 0; i < NUM_SAMPLES; i++) {
        generate_point(&inputs[i][0], &inputs[i][1]);
        targets[i] = is_above_line(inputs[i][0], inputs[i][1]) ? 1.0 : 0.0;
    }

    const char* activation_names[] = {"Sigmoid", "ReLU", "Tanh"};
    int activation_functions[] = {SIGMOID, RELU, TANH};
    int num_activations = sizeof(activation_functions) / sizeof(activation_functions[0]);

    for (int use_regularization = 0; use_regularization <= 1; use_regularization++) {
        printf("\n%s Regularization:\n", use_regularization ? "With" : "Without");

        for (int a = 0; a < num_activations; a++) {
            MLP mlp = create_mlp(activation_functions[a], use_regularization);
            g_mlp = &mlp;

            printf("\nTraining MLP with %s activation...\n", activation_names[a]);

            clock_t start_time = clock();
            double total_training_time = 0.0;

            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                clock_t epoch_start = clock();
                double error = train(&mlp, inputs, targets, NUM_SAMPLES);
                clock_t epoch_end = clock();
                double epoch_time = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
                total_training_time += epoch_time;

                if (epoch % 1000 == 0) {
                    printf("Epoch %d: Error = %f, Time = %.4f seconds\n", epoch, error, epoch_time);
                }
                if (error < EPSILON) {
                    printf("Converged at epoch %d\n", epoch);
                    break;
                }
            }

            clock_t end_time = clock();
            double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

            printf("\nTotal training time: %.4f seconds\n", total_training_time);
            printf("Total execution time: %.4f seconds\n", total_time);

            char filename[100];
            snprintf(filename, sizeof(filename), "mlp_plot_%s_%s_regularization.png", 
                     activation_names[a], use_regularization ? "with" : "without");
            create_plot(&mlp, inputs, targets, NUM_SAMPLES, filename);

            printf("\nTesting MLP with %s activation:\n", activation_names[a]);
            int correct = 0;
            for (int i = 0; i < TEST_SAMPLES; i++) {
                double x, y;
                generate_point(&x, &y);
                double input[2] = {x, y};
                forward_propagation(&mlp, input);
                double output = mlp.layers[mlp.num_layers - 1].outputs[0];
                int predicted = output > 0.5 ? 1 : 0;
                int actual = is_above_line(x, y);
                if (predicted == actual) correct++;
            }
            printf("Accuracy: %.2f%%\n", (double)correct / TEST_SAMPLES * 100);

            free_mlp(&mlp);
        }
    }

