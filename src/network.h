#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "activation.h"
#include "csv.h"


class network {
public:
    // Hyper-parameters
    static constexpr unsigned batch_size = 16;
    static constexpr float learning_rate = 0.001;
    static constexpr unsigned hidden1_neurons = 64;
    static constexpr unsigned hidden2_neurons = 32;

    static constexpr unsigned training_set_size = 60'000;
    static constexpr unsigned test_set_size = 10'000;
    static constexpr unsigned input_size = 28 * 28;
    static constexpr unsigned output_size = 10;

    using input_t = matrix<batch_size, input_size>;
    using labels_t = std::array<unsigned, batch_size>;

private:
    template<
        unsigned PREV_NEURONS,
        unsigned NEURONS,
        init::random_t WEIGHT_INIT_METHOD
    >
    struct layer_t {
        static constexpr unsigned neurons = NEURONS;

        using weights_t = matrix<PREV_NEURONS, NEURONS>;
        using biases_t = matrix<1, NEURONS>;
        using neurons_t = matrix<batch_size, NEURONS>;

        layer_t()
          : weights{init::random<weights_t::cols, PREV_NEURONS, NEURONS>(WEIGHT_INIT_METHOD)},
            biases{init::random<biases_t::cols, PREV_NEURONS, NEURONS>(WEIGHT_INIT_METHOD)} {}

        void compute_inner_potentials(const matrix<batch_size, PREV_NEURONS>& inputs) {
            potentials = weights * inputs;
            for (std::size_t i = 0; i < neurons; ++i)
                for (std::size_t j = 0; j < batch_size; ++j)
                    potentials[j, i] += biases[i];
        }

        weights_t weights;
        biases_t biases;
        neurons_t potentials, activations;
    };

    using hidden1_layer_t = layer_t<input_size, hidden1_neurons, init::random_t::normal_he>;
    using hidden2_layer_t = layer_t<hidden1_neurons, hidden2_neurons, init::random_t::normal_he>;
    using output_layer_t = layer_t<hidden2_neurons, output_size, init::random_t::normal_glorot>;

    void forward_pass(const input_t& inputs) {
        hidden1.compute_inner_potentials(inputs);
        hidden1.activations = apply<ReLU>(hidden1.potentials);
        hidden2.compute_inner_potentials(hidden1.activations);
        hidden2.activations = apply<ReLU>(hidden2.potentials);
        output.compute_inner_potentials(hidden2.activations);
        output.activations = softmax(output.potentials);
    }

    void write_batch(csv& out) {
        labels_t result;
        for (std::size_t k = 0; k < batch_size; ++k) {
            float max_value = 0;
            for (std::size_t i = 0; i < output_layer_t::neurons; ++i) {
                if (output.activations[k, i] > max_value) {
                    max_value = output.activations[k, i];
                    result[k] = i;
                }
            }
        }
        out.write_batch(result);
    }
    
    float error(const input_t& inputs, const labels_t& labels) {
        forward_pass(inputs);
        float err = 0;
        for (std::size_t k = 0; k < batch_size; ++k)
            err -= std::log(output.activations[k, labels[k]]);
        return err / static_cast<float>(batch_size);
    }

    void backpropagation(const input_t& inputs, const labels_t& labels) {
        forward_pass(inputs);

        // Output layer gradient
        auto output_dE_dPotential = output.activations;
        for (std::size_t k = 0; k < batch_size; ++k)
            output_dE_dPotential[k, labels[k]] -= 1.f;
        output_layer_t::weights_t output_dE_dw = output_dE_dPotential * hidden2.activations.transpose();
        auto ones = matrix<1, batch_size>{1.f};
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

        // Update weights
        output.weights -= learning_rate * output_dE_dw;
        output.biases -= learning_rate * output_dE_dbias;
        hidden2.weights -= learning_rate * hidden2_dE_dw;
        hidden2.biases -= learning_rate * hidden2_dE_dbias;
        hidden1.weights -= learning_rate * hidden1_dE_dw;
        hidden1.biases -= learning_rate * hidden1_dE_dbias;
    }

public:
    void learn(csv& inputs, csv& labels, csv& train_out) {
        for (std::size_t i = 0; i < training_set_size / network::batch_size; ++i) {
            auto input = inputs.read_batch<batch_size, input_size>();
            auto label = labels.read_batch_labels<labels_t, batch_size>();

            backpropagation(input, label);
            write_batch(train_out);
        }
    }

    void predict(csv& inputs, csv& test_out) {
        for (std::size_t i = 0; i < test_set_size / network::batch_size; ++i) {
            auto input = inputs.read_batch<batch_size, input_size>();
            forward_pass(input);
            write_batch(test_out);
        }
    }

private:
    hidden1_layer_t hidden1;
    hidden2_layer_t hidden2;
    output_layer_t output;
};



#endif //NETWORK_H
