#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_EPOCHS 69
#define LEARNING_RATE 0.1
#define EPSILON 1e-5

typedef struct {
    double *weights;
    int num_features;
    int max_epochs;
} Perceptron;

Perceptron* create_perceptron(int num_features, int max_epochs) {
    Perceptron *p = (Perceptron*)malloc(sizeof(Perceptron));
    p->weights = (double*)malloc((num_features + 1) * sizeof(double));
    p->num_features = num_features;
    p->max_epochs = max_epochs;

    srand((unsigned int)time(NULL));
    for (int i = 0; i <= num_features; i++) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    return p;
}

void free_perceptron(Perceptron *p) {
    free(p->weights);
    free(p);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double predict(Perceptron *p, double *x) {
    double sum = p->weights[p->num_features];
    for (int i = 0; i < p->num_features; i++) {
        sum += p->weights[i] * x[i];
    }
    return sigmoid(sum);
}

double calculate_loss(double predicted, double actual) {
    return -actual * log(predicted) - (1 - actual) * log(1 - predicted);
}

void fit(Perceptron *p, double **X, int *y, int num_samples) {
    int epoch;
    for (epoch = 0; epoch < p->max_epochs; epoch++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < num_samples; i++) {
            double prediction = predict(p, X[i]);
            double error = y[i] - prediction;
            total_loss += calculate_loss(prediction, y[i]);
            
            // Update weights
            for (int j = 0; j < p->num_features; j++) {
                p->weights[j] += LEARNING_RATE * error * X[i][j];
            }
            p->weights[p->num_features] += LEARNING_RATE * error; // Update bias
        }
        
        double mean_loss = total_loss / num_samples;
        
        printf("Epoch %d: Loss = %f\n", epoch + 1, mean_loss);
        
        if (mean_loss < EPSILON) break;
    }
    printf("Training completed in %d epochs\n", epoch + 1);
}

int main() {
    // Example usage
    int num_features = 2;
    int num_samples = 4;
    
    double **X = (double**)malloc(num_samples * sizeof(double*));
    for (int i = 0; i < num_samples; i++) {
        X[i] = (double*)malloc(num_features * sizeof(double));
    }
    
    // XOR problem
    X[0][0] = 0; X[0][1] = 0;
    X[1][0] = 0; X[1][1] = 1;
    X[2][0] = 1; X[2][1] = 0;
    X[3][0] = 1; X[3][1] = 1;
    
    int y[] = {0, 1, 1, 0};

    Perceptron *p = create_perceptron(num_features, MAX_EPOCHS);
    
    fit(p, X, y, num_samples);

    // Test predictions
    printf("\nFinal predictions:\n");
    for (int i = 0; i < num_samples; i++) {
        double pred = predict(p, X[i]);
        printf("Input: (%.0f, %.0f), Predicted: %.4f, Actual: %d\n", X[i][0], X[i][1], pred, y[i]);
    }

    // Clean up
    for (int i = 0; i < num_samples; i++) {
        free(X[i]);
    }
    free(X);
    free_perceptron(p);

    return 0;
}
