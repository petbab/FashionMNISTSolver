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
    static constexpr std::chrono::duration time_limit = 9min;
    static constexpr unsigned epochs_after_threshold = 10;

    // Inputs
    static constexpr unsigned validation_set_size = 5'000;
    static constexpr unsigned training_set_size = 60'000 - validation_set_size;
    static constexpr unsigned test_set_size = 10'000;

    static_assert(validation_set_size % cfg::batch_size == 0);
    static_assert(training_set_size % cfg::batch_size == 0);
    static_assert(test_set_size % cfg::batch_size == 0);

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

        /**
         * @brief Compute gradient for backpropagation
         *
         * Calculates gradients for weights, biases, and propagates error to previous layer
         *
         * @tparam NEXT_NEURONS Number of neurons in the next layer
         * @param next_dE_dPotential Gradient of error with respect to potentials in the next layer
         * @param next_weights Weights matrix of the next layer
         * @param input Input matrix from the previous layer
         *
         * @return Gradient of error with respect to potentials in the current layer
         */
        template<unsigned NEXT_NEURONS>
        neurons_t compute_gradient(const matrix<cfg::batch_size, NEXT_NEURONS>& next_dE_dPotential,
                                   const matrix<NEURONS, NEXT_NEURONS>& next_weights,
                                   const matrix<cfg::batch_size, PREV_NEURONS>& input) {
            // Compute gradient
            neurons_t dE_dy = next_weights.transpose() * next_dE_dPotential;
            neurons_t dE_dPotential = pmult(apply<dReLU>(potentials), dE_dy);
            weights_t dE_dw = dE_dPotential * input.transpose();
            biases_t dE_dbias = dE_dPotential * matrix<1, cfg::batch_size>{1.f};

            // Update gradients with momentum
            weights_gradient = cfg::momentum * weights_gradient + cfg::learning_rate * dE_dw;
            biases_gradient = cfg::momentum * biases_gradient + cfg::learning_rate * dE_dbias;

            return dE_dPotential;
        }

        void update_weights() {
            (weights *= cfg::weight_decay) -= weights_gradient;
            (biases *= cfg::weight_decay) -= biases_gradient;
        }

        void save_optimal_weights() {
            optimal_weights = weights;
        }

        void load_optimal_weights() {
            weights = optimal_weights;
        }

        weights_t weights, weights_gradient, optimal_weights;
        biases_t biases, biases_gradient;
        neurons_t potentials, activations;
    };

    using hidden1_layer_t = layer_t<input_size, cfg::hidden1_neurons, init::random_t::normal_he>;
    using hidden2_layer_t = layer_t<cfg::hidden1_neurons, cfg::hidden2_neurons, init::random_t::normal_he>;
    using output_layer_t = layer_t<cfg::hidden2_neurons, output_size, init::random_t::normal_glorot>;

    /**
     * @brief Perform forward propagation through the network.
     *
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
        output_layer_t::biases_t output_dE_dbias = output_dE_dPotential * matrix<1, cfg::batch_size>{1.f};

        auto hidden2_dE_dPotential = hidden2.compute_gradient(output_dE_dPotential, output.weights, hidden1.activations);
        hidden1.compute_gradient(hidden2_dE_dPotential, hidden2.weights, inputs);

        // Update gradients with momentum
        output.weights_gradient = cfg::momentum * output.weights_gradient + cfg::learning_rate * output_dE_dw;
        output.biases_gradient = cfg::momentum * output.biases_gradient + cfg::learning_rate * output_dE_dbias;

        output.update_weights();
        hidden2.update_weights();
        hidden1.update_weights();
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
            double max_value = 0;
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
     * @return double Accuracy of the model on the validation set
     */
    double validation_set_accuracy(csv_image_reader& inputs, csv_label_reader& labels) {
        unsigned hits = 0;
        for (std::size_t i = 0; i < validation_set_size / cfg::batch_size; ++i) {
            auto input = inputs.read_batch<cfg::batch_size, input_size>();
            auto label = labels.read_batch<labels_t, cfg::batch_size>();
            auto prediction = predict_batch(input);

            for (std::size_t k = 0; k < cfg::batch_size; ++k)
                if (prediction[k] == label[k])
                    ++hits;
        }

        return static_cast<double>(hits) / static_cast<double>(validation_set_size);
    }

    /**
     * @brief Save the current weights as optimal
     */
    void save_optimal_weights() {
        output.save_optimal_weights();
        hidden2.save_optimal_weights();
        hidden1.save_optimal_weights();
    }

    /**
     * @brief Load the optimal weights after learning
     */
    void load_optimal_weights() {
        output.load_optimal_weights();
        hidden2.load_optimal_weights();
        hidden1.load_optimal_weights();
    }

public:
    /**
     * @brief Train the neural network
     *
     * The learning algorithm is SGD with momentum and weight decay.
     * When the network reaches the `accuracy_threshold`, the learning
     * algorithm computes `epochs_after_threshold` epochs and then stops.
     * If the network surpasses the `best_accuracy` in the extra epochs,
     * the algorithm continues with another round of extra epochs.
     *
     * @param inputs Training input data
     * @param labels Training labels
     */
    void learn(csv_image_reader& inputs, csv_label_reader& labels) {
        timer t{std::cout, "Learning time"};
        std::cout << std::fixed << std::setprecision(2);

        unsigned best_epoch = 0;
        double best_accuracy = 0;
        for (
            unsigned epoch = 0;
            t.duration() < time_limit && epoch - best_epoch <= epochs_after_threshold;
            ++epoch
        ) {
            for (std::size_t i = 0; i < training_set_size / cfg::batch_size; ++i) {
                auto input = inputs.read_batch<cfg::batch_size, input_size>();
                auto label = labels.read_batch<labels_t, cfg::batch_size>();
                backpropagation(input, label);
            }

            auto accuracy = validation_set_accuracy(inputs, labels);
            if (accuracy > best_accuracy) {
                save_optimal_weights();
                best_accuracy = accuracy;
                best_epoch = epoch;
            }

            std::cout << "Accuracy after epoch " << epoch << ": " << accuracy * 100.f << "%\n";
            
            inputs.seek_begin();
            labels.seek_begin();
        }
        
        load_optimal_weights();
    }

    /**
     * @brief Generate predictions for a dataset
     *
     * @tparam SIZE Number of inputs in the dataset
     * @param inputs CSV file with input data
     * @param out CSV file to write predictions to
     */
    template<unsigned SIZE>
    void predict(csv_image_reader& inputs, csv_writer& out) {
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
