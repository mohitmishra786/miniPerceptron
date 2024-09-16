#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cairo.h>

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1
#define MAX_EPOCHS 10000
#define EPSILON 1e-3
#define NUM_SAMPLES 1000
#define TEST_SAMPLES 100
#define PLOT_SIZE 20
#define PLOT_WIDTH 800
#define PLOT_HEIGHT 800

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

Layer create_layer(int inputs, int neurons) {
    Layer layer;
    layer.inputs = inputs;
    layer.neurons = neurons;
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

MLP create_mlp() {
    MLP mlp;
    mlp.num_layers = 3;
    mlp.layers = (Layer *)malloc(mlp.num_layers * sizeof(Layer));

    mlp.layers[0] = create_layer(INPUT_NEURONS, HIDDEN_NEURONS);
    mlp.layers[1] = create_layer(HIDDEN_NEURONS, HIDDEN_NEURONS);
    mlp.layers[2] = create_layer(HIDDEN_NEURONS, OUTPUT_NEURONS);

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
            layer->outputs[i] = sigmoid(sum);
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
    deltas[mlp->num_layers - 1][0] = output_error * sigmoid_derivative(output_layer->outputs[0]);

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

void init_plot(char plot[PLOT_SIZE][PLOT_SIZE]) {
    for (int i = 0; i < PLOT_SIZE; i++) {
        for (int j = 0; j < PLOT_SIZE; j++) {
            plot[i][j] = ' ';
        }
    }
}

void add_point_to_plot(char plot[PLOT_SIZE][PLOT_SIZE], double x, double y, char symbol) {
    int plot_x = (int)((x + 1) / 2 * (PLOT_SIZE - 1));
    int plot_y = (int)((1 - (y + 1) / 2) * (PLOT_SIZE - 1));
    if (plot_x >= 0 && plot_x < PLOT_SIZE && plot_y >= 0 && plot_y < PLOT_SIZE) {
        plot[plot_y][plot_x] = symbol;
    }
}

void add_line_to_plot(char plot[PLOT_SIZE][PLOT_SIZE], double (*line_func)(double)) {
    for (int x = 0; x < PLOT_SIZE; x++) {
        double plot_x = (double)x / (PLOT_SIZE - 1) * 2 - 1;
        double plot_y = line_func(plot_x);
        int plot_y_int = (int)((1 - (plot_y + 1) / 2) * (PLOT_SIZE - 1));
        if (plot_y_int >= 0 && plot_y_int < PLOT_SIZE) {
            plot[plot_y_int][x] = '-';
        }
    }
}

void print_plot(char plot[PLOT_SIZE][PLOT_SIZE]) {
    for (int i = 0; i < PLOT_SIZE; i++) {
        for (int j = 0; j < PLOT_SIZE; j++) {
            printf("%c", plot[j][i]);
        }
        printf("\n");
    }
}

void create_plot(MLP *mlp, double inputs[][INPUT_NEURONS], double *targets, int num_samples) {
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

    cairo_set_source_rgb(cr, 0, 0, 1);
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

    cairo_surface_write_to_png(surface, "mlp_plot.png");
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}

int main() {
    srand((unsigned int)time(NULL));

    MLP mlp = create_mlp();
    g_mlp = &mlp;

    double inputs[NUM_SAMPLES][INPUT_NEURONS];
    double targets[NUM_SAMPLES];

    // Generating training data
    for (int i = 0; i < NUM_SAMPLES; i++) {
        generate_point(&inputs[i][0], &inputs[i][1]);
        targets[i] = is_above_line(inputs[i][0], inputs[i][1]) ? 1.0 : 0.0;
    }

    char plot[PLOT_SIZE][PLOT_SIZE];

    printf("Plot before training:\n");
    init_plot(plot);
    add_line_to_plot(plot, true_line);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        add_point_to_plot(plot, inputs[i][0], inputs[i][1], targets[i] > 0.5 ? '+' : '.');
    }
    print_plot(plot);

    printf("\nTraining MLP for line classification...\n");

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double error = train(&mlp, inputs, targets, NUM_SAMPLES);
        if (epoch % 1000 == 0) {
            printf("Epoch %d: Error = %f\n", epoch, error);
        }
        if (error < EPSILON) {
            printf("Converged at epoch %d\n", epoch);
            break;
        }
    }

    create_plot(&mlp, inputs, targets, NUM_SAMPLES);

    printf("\nPlot after training:\n");
    init_plot(plot);
    add_line_to_plot(plot, true_line);
    add_line_to_plot(plot, mlp_line);
    for (int i = 0; i < TEST_SAMPLES; i++) {
        double x, y;
        generate_point(&x, &y);
        double input[2] = {x, y};
        forward_propagation(&mlp, input);
        double output = mlp.layers[mlp.num_layers - 1].outputs[0];
        add_point_to_plot(plot, x, y, output > 0.5 ? '+' : '.');
    }
    print_plot(plot);

    printf("\nTesting MLP:\n");
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
        printf("Point (%.2f, %.2f): Predicted = %d, Actual = %d\n", x, y, predicted, actual);
    }
    printf("Accuracy: %.2f%%\n", (double)correct / TEST_SAMPLES * 100);

    free_mlp(&mlp);
    return 0;
}
