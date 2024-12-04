#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "activation.h"
#include "csv.h"
#include "hyperparameters.h"
#include "timer.h"

#include <iostream>
#include <iomanip>
#include <array>
#include <limits>
#include <chrono>

using namespace std::chrono_literals;

class network {
public:
    // Learning limits
    static constexpr float accuracy_threshold = 0.882;
    static constexpr std::chrono::duration time_limit = 9min;

    // Inputs
    static constexpr unsigned validation_set_size = 5'000;
    static constexpr unsigned training_set_size = 60'000 - validation_set_size;
    static constexpr unsigned test_set_size = 10'000;

    static constexpr unsigned input_size = 28 * 28;
    static constexpr unsigned output_size = 10;

    using input_t = matrix<cfg::batch_size, input_size>;
    using labels_t = std::array<unsigned, cfg::batch_size>;

private:
    /**
     * @brief Templated layer structure for neural network layers
     *
     * @tparam PREV_NEURONS Number of neurons in the previous layer
     * @tparam NEURONS Number of neurons in the current layer
     * @tparam WEIGHT_INIT_METHOD Method for initializing weights
     */
    template<
        unsigned PREV_NEURONS,
        unsigned NEURONS,
        init::random_t WEIGHT_INIT_METHOD
    >
    struct layer_t {
        static constexpr unsigned neurons = NEURONS;

        using weights_t = matrix<PREV_NEURONS, NEURONS>;
        using biases_t = matrix<1, NEURONS>;
        using neurons_t = matrix<cfg::batch_size, NEURONS>;

        layer_t()
          : weights{init::random<weights_t::cols, PREV_NEURONS, NEURONS>(WEIGHT_INIT_METHOD)},
            biases{init::random<biases_t::cols, PREV_NEURONS, NEURONS>(WEIGHT_INIT_METHOD)} {}

        void compute_inner_potentials(const matrix<cfg::batch_size, PREV_NEURONS>& inputs) {
            potentials = weights * inputs;
            for (std::size_t i = 0; i < neurons; ++i)
                for (std::size_t j = 0; j < cfg::batch_size; ++j)
                    potentials[j, i] += biases[i];
        }

        weights_t weights, weights_gradient;
        biases_t biases, biases_gradient;
        neurons_t potentials, activations;
    };

    using hidden1_layer_t = layer_t<input_size, cfg::hidden1_neurons, init::random_t::normal_he>;
    using hidden2_layer_t = layer_t<cfg::hidden1_neurons, cfg::hidden2_neurons, init::random_t::normal_he>;
    using output_layer_t = layer_t<cfg::hidden2_neurons, output_size, init::random_t::normal_glorot>;

    /**
     * @brief Perform forward propagation through the network.
     * Compute inner potentials and activations.
     * The activation functions are ReLU in the hidden layers and softmax in the output layer.
     *
     * @param inputs Input batch to propagate through the network
     */
    void forward_pass(const input_t& inputs) {
        hidden1.compute_inner_potentials(inputs);
        hidden1.activations = apply<ReLU>(hidden1.potentials);
        hidden2.compute_inner_potentials(hidden1.activations);
        hidden2.activations = apply<ReLU>(hidden2.potentials);
        output.compute_inner_potentials(hidden2.activations);
        output.activations = softmax(output.potentials);
    }

    /**
     * @brief Perform backpropagation to update network weights.
     *
     * The error function is categorical cross-entropy.
     * The learning algorithm is SGD with momentum and weight decay.
     *
     * @param inputs Input batch
     * @param labels Corresponding label batch
     */
    void backpropagation(const input_t& inputs, const labels_t& labels) {
        forward_pass(inputs);

        // Output layer gradient
        auto output_dE_dPotential = output.activations;
        for (std::size_t k = 0; k < cfg::batch_size; ++k)
            output_dE_dPotential[k, labels[k]] -= 1.f;
        output_layer_t::weights_t output_dE_dw = output_dE_dPotential * hidden2.activations.transpose();
        auto ones = matrix<1, cfg::batch_size>{1.f};
        output_layer_t::biases_t output_dE_dbias = output_dE_dPotential * ones;

        // Second hidden layer gradient
        auto hidden2_dE_dy = output.weights.transpose() * output_dE_dPotential;
        auto hidden2_dE_dy_dsigma = pmult(apply<dReLU>(hidden2.potentials), hidden2_dE_dy);
        hidden2_layer_t::weights_t hidden2_dE_dw = hidden2_dE_dy_dsigma * hidden1.activations.transpose();
        hidden2_layer_t::biases_t hidden2_dE_dbias = hidden2_dE_dy_dsigma * ones;

        // First hidden layer gradient
        auto hidden1_dE_dy = hidden2.weights.transpose() * hidden2_dE_dy_dsigma;
        auto hidden1_dE_dy_dsigma = pmult(apply<dReLU>(hidden1.potentials), hidden1_dE_dy);
        hidden1_layer_t::weights_t hidden1_dE_dw = hidden1_dE_dy_dsigma * inputs.transpose();
        hidden1_layer_t::biases_t hidden1_dE_dbias = hidden1_dE_dy_dsigma * ones;

        // Update gradients with momentum
        output.weights_gradient = cfg::momentum * output.weights_gradient - cfg::learning_rate * output_dE_dw;
        output.biases_gradient = cfg::momentum * output.biases_gradient - cfg::learning_rate * output_dE_dbias;
        hidden2.weights_gradient = cfg::momentum * hidden2.weights_gradient - cfg::learning_rate * hidden2_dE_dw;
        hidden2.biases_gradient = cfg::momentum * hidden2.biases_gradient - cfg::learning_rate * hidden2_dE_dbias;
        hidden1.weights_gradient = cfg::momentum * hidden1.weights_gradient - cfg::learning_rate * hidden1_dE_dw;
        hidden1.biases_gradient = cfg::momentum * hidden1.biases_gradient - cfg::learning_rate * hidden1_dE_dbias;

        // Update weights
        (output.weights *= cfg::weight_decay) += output.weights_gradient;
        (output.biases *= cfg::weight_decay) += output.biases_gradient;
        (hidden2.weights *= cfg::weight_decay) += hidden2.weights_gradient;
        (hidden2.biases *= cfg::weight_decay) += hidden2.biases_gradient;
        (hidden1.weights *= cfg::weight_decay) += hidden1.weights_gradient;
        (hidden1.biases *= cfg::weight_decay) += hidden1.biases_gradient;
    }

    /**
     * @brief Predict classes for a batch of inputs
     *
     * @param inputs Input batch to predict
     * @return labels_t Predicted labels for the input batch
     */
    labels_t predict_batch(const input_t& inputs) {
        forward_pass(inputs);

        labels_t result;
        for (std::size_t k = 0; k < cfg::batch_size; ++k) {
            float max_value = 0;
            for (std::size_t i = 0; i < output_layer_t::neurons; ++i) {
                if (output.activations[k, i] > max_value) {
                    max_value = output.activations[k, i];
                    result[k] = i;
                }
            }
        }
        return result;
    }

    /**
     * @brief Calculate accuracy on the validation set
     *
     * @param inputs Validation input data
     * @param labels Validation labels
     * @return float Accuracy of the model on the validation set
     */
    float validation_set_accuracy(csv& inputs, csv& labels) {
        unsigned hits = 0;
        for (std::size_t i = 0; i < validation_set_size / cfg::batch_size; ++i) {
            auto input = inputs.read_batch<cfg::batch_size, input_size>();
            auto label = labels.read_batch_labels<labels_t, cfg::batch_size>();
            auto prediction = predict_batch(input);

            for (std::size_t k = 0; k < cfg::batch_size; ++k)
                if (prediction[k] == label[k])
                    ++hits;
        }

        return static_cast<float>(hits) / static_cast<float>(validation_set_size);
    }

public:
    /**
     * @brief Train the neural network
     *
     * Performs batch training with early stopping based on validation accuracy.
     * The learning algorithm is SGD with momentum and weight decay.
     *
     * @param inputs Training input data
     * @param labels Training labels
     */
    void learn(csv& inputs, csv& labels) {
        timer t{std::cout, "Learning time"};

        std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << "Hyperparameters:"
                << "\n    learning_rate = " << cfg::learning_rate
                << "\n    momentum = " << cfg::momentum
                << "\n    weight_decay = " << cfg::weight_decay
                << "\n    hidden1_neurons = " << cfg::hidden1_neurons
                << "\n    hidden2_neurons = " << cfg::hidden2_neurons
                << "\n    batch_size = " << cfg::batch_size << '\n';

        std::cout << std::fixed << std::setprecision(2);

        float accuracy = 0, prev_accuracy = 0;
        for (
            unsigned epoch = 0;
            (prev_accuracy < accuracy_threshold || accuracy >= prev_accuracy) && t.duration() < time_limit;
            ++epoch
        ) {
            for (std::size_t i = 0; i < training_set_size / cfg::batch_size; ++i) {
                auto input = inputs.read_batch<cfg::batch_size, input_size>();
                auto label = labels.read_batch_labels<labels_t, cfg::batch_size>();

                backpropagation(input, label);
            }

            prev_accuracy = accuracy;
            accuracy = validation_set_accuracy(inputs, labels);
            inputs.seek_begin();
            labels.seek_begin();

            std::cout << "Accuracy after epoch " << epoch << ": " << accuracy * 100.f << "%\n";
        }
    }

    /**
     * @brief Generate predictions for a dataset
     *
     * @tparam SIZE Number of inputs in the dataset
     * @param inputs CSV file with input data
     * @param out CSV file to write predictions to
     */
    template<unsigned SIZE>
    void predict(csv& inputs, csv& out) {
        for (std::size_t i = 0; i < SIZE / cfg::batch_size; ++i) {
            auto input = inputs.read_batch<cfg::batch_size, input_size>();
            out.write_batch(predict_batch(input));
        }
    }

private:
    hidden1_layer_t hidden1;
    hidden2_layer_t hidden2;
    output_layer_t output;
};



#endif //NETWORK_H
