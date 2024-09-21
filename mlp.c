#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cairo.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_EPOCHS 10000
#define EPSILON 1e-3
#define NUM_SAMPLES 1000
#define TEST_SAMPLES 100
#define PLOT_WIDTH 800
#define PLOT_HEIGHT 800

#define SIGMOID 0
#define RELU 1
#define TANH 2

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
    double learning_rate;
    double l2_lambda;
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

typedef double (*CurveFunction)(double);

double linear_curve(double x) {
    return 0.5 * x + 0.1;
}

double quadratic_curve(double x) {
    return 0.5 * x * x - 0.2;
}

double sine_curve(double x) {
    return 0.5 * sin(3 * x);
}

CurveFunction g_curve_function;
MLP* g_mlp;

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

MLP create_mlp(int input_neurons, int hidden_layers, int hidden_neurons, int output_neurons, 
               int activation_function, int use_regularization, double learning_rate, double l2_lambda) {
    MLP mlp;
    mlp.num_layers = hidden_layers + 2;
    mlp.layers = (Layer *)malloc(mlp.num_layers * sizeof(Layer));
    mlp.use_regularization = use_regularization;
    mlp.learning_rate = learning_rate;
    mlp.l2_lambda = l2_lambda;

    mlp.layers[0] = create_layer(input_neurons, hidden_neurons, activation_function);
    for (int i = 1; i < hidden_layers + 1; i++) {
        mlp.layers[i] = create_layer(hidden_neurons, hidden_neurons, activation_function);
    }
    mlp.layers[mlp.num_layers - 1] = create_layer(hidden_neurons, output_neurons, SIGMOID);

    return mlp;
}

void forward_propagation(MLP *mlp, double *input) {
    for (int l = 0; l < mlp->num_layers; l++) {
        Layer *layer = &mlp->layers[l];
        double *prev_outputs = l == 0 ? input : mlp->layers[l - 1].outputs;
        int prev_size = l == 0 ? layer->inputs : mlp->layers[l - 1].neurons;

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
        int prev_size = l == 0 ? layer->inputs : mlp->layers[l - 1].neurons;

        for (int i = 0; i < layer->neurons; i++) {
            for (int j = 0; j < prev_size; j++) {
                double weight_update = mlp->learning_rate * deltas[l][i] * prev_outputs[j];
                if (mlp->use_regularization) {
                    weight_update -= mlp->learning_rate * mlp->l2_lambda * layer->weights[i][j];
                }
                layer->weights[i][j] += weight_update;
            }
            layer->biases[i] += mlp->learning_rate * deltas[l][i];
        }
    }

    for (int l = 0; l < mlp->num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

double train(MLP *mlp, double inputs[][2], double *targets, int num_samples) {
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

int is_above_curve(double x, double y) {
    return y > g_curve_function(x);
}

double mlp_curve(double x) {
    double input[2] = {x, 0};
    forward_propagation(g_mlp, input);
    double y = g_mlp->layers[g_mlp->num_layers - 1].outputs[0];
    return (y - 0.5) * 2;
}

void create_plot(MLP *mlp, double inputs[][2], double *targets, int num_samples, const char *filename, int activation_function) {
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
    cairo_move_to(cr, -1, g_curve_function(-1));
    for (double x = -0.95; x <= 1; x += 0.05) {
        cairo_line_to(cr, x, g_curve_function(x));
    }
    cairo_stroke(cr);

    cairo_set_source_rgb(cr, 1, 0, 0);
    cairo_move_to(cr, -1, mlp_curve(-1));
    for (double x = -0.95; x <= 1; x += 0.05) {
        cairo_line_to(cr, x, mlp_curve(x));
    }
    cairo_stroke(cr);

    // Create directories for images
    char directory[100];
    if (activation_function == SIGMOID) {
        snprintf(directory, sizeof(directory), "images/sigmoid");
    } else if (activation_function == RELU) {
        snprintf(directory, sizeof(directory), "images/relu");
    } else { // TANH
        snprintf(directory, sizeof(directory), "images/tanh");
    }
    mkdir(directory, 0777); // Create directory if it doesn't exist

    char filepath[200]; 
    snprintf(filepath, sizeof(filepath), "%s/%s", directory, filename);

    cairo_surface_write_to_png(surface, filepath);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    int input_neurons = 2;
    int hidden_layers = 2;
    int hidden_neurons = 4;
    int output_neurons = 1;
    int activation_function = SIGMOID;
    int use_regularization = 0;
    double learning_rate = 0.1;
    double l2_lambda = 0.01;
    CurveFunction curve_function = linear_curve;

    // Parse command-line arguments
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--hidden-layers") == 0) hidden_layers = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--hidden-neurons") == 0) hidden_neurons = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--activation") == 0) {
                if (strcmp(argv[i + 1], "sigmoid") == 0) activation_function = SIGMOID;
                else if (strcmp(argv[i + 1], "relu") == 0) activation_function = RELU;
                else if (strcmp(argv[i + 1], "tanh") == 0) activation_function = TANH;
            }
            else if (strcmp(argv[i], "--regularization") == 0) use_regularization = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "--learning-rate") == 0) learning_rate = atof(argv[i + 1]);
            else if (strcmp(argv[i], "--l2-lambda") == 0) l2_lambda = atof(argv[i + 1]);
            else if (strcmp(argv[i], "--curve") == 0) {
                if (strcmp(argv[i + 1], "linear") == 0) curve_function = linear_curve;
                else if (strcmp(argv[i + 1], "quadratic") == 0) curve_function = quadratic_curve;
                else if (strcmp(argv[i + 1], "sine") == 0) curve_function = sine_curve;
            }
        }
    }

    g_curve_function = curve_function;

    double inputs[NUM_SAMPLES][2];
    double targets[NUM_SAMPLES];

    // Generating training data
    for (int i = 0; i < NUM_SAMPLES; i++) {
        generate_point(&inputs[i][0], &inputs[i][1]);
        targets[i] = is_above_curve(inputs[i][0], inputs[i][1]) ? 1.0 : 0.0;
    }

    const char* activation_names[] = {"Sigmoid", "ReLU", "Tanh"};

    printf("\n%s Regularization:\n", use_regularization ? "With" : "Without");

    MLP mlp = create_mlp(input_neurons, hidden_layers, hidden_neurons, output_neurons, 
                         activation_function, use_regularization, learning_rate, l2_lambda);
    g_mlp = &mlp;

    printf("\nTraining MLP with %s activation...\n", activation_names[activation_function]);

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
            char filename[100];
            snprintf(filename, sizeof(filename), "mlp_plot_%s_%s_regularization_epoch_%d.png", 
                     activation_names[activation_function], use_regularization ? "with" : "without", epoch);
            create_plot(&mlp, inputs, targets, NUM_SAMPLES, filename, activation_function);
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

    // Create the "images" directory if it doesn't exist
    mkdir("images", 0777); 

    char filename[100];
    snprintf(filename, sizeof(filename), "mlp_plot_%s_%s_regularization.png", 
             activation_names[activation_function], use_regularization ? "with" : "without");
    create_plot(&mlp, inputs, targets, NUM_SAMPLES, filename, activation_function);

    printf("\nTesting MLP with %s activation:\n", activation_names[activation_function]);
    int correct = 0;
    for (int i = 0; i < TEST_SAMPLES; i++) {
        double x, y;
        generate_point(&x, &y);
        double input[2] = {x, y};
        forward_propagation(&mlp, input);
        double output = mlp.layers[mlp.num_layers - 1].outputs[0];
        int predicted = output > 0.5 ? 1 : 0;
        int actual = is_above_curve(x, y);
        if (predicted == actual) correct++;
    }
    printf("Accuracy: %.2f%%\n", (double)correct / TEST_SAMPLES * 100);

    free_mlp(&mlp);

    return 0;
}
