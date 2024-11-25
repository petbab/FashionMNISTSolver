#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "csv_reader.h"
#include "activation.h"


class network {
public:
    static constexpr unsigned batch_size = 4;
    static constexpr unsigned input_size = 2;
    static constexpr unsigned output_size = 1;

    using input_t = matrix<batch_size, input_size>;
    using output_t = matrix<batch_size, output_size>;

private:
    template<
        unsigned PREV_NEURONS,
        unsigned NEURONS,
        random_t WEIGHT_INIT_METHOD
    >
    struct layer_t {
        static constexpr unsigned neurons = NEURONS;

        using weights_t = matrix<PREV_NEURONS, NEURONS>;
        using biases_t = matrix<1, NEURONS>;
        using neurons_t = matrix<batch_size, NEURONS>;

        layer_t() : weights{weights_t::random(WEIGHT_INIT_METHOD)} {}

        void compute_inner_potentials(const matrix<batch_size, PREV_NEURONS>& inputs) {
            potentials = weights * inputs;
            for (std::size_t i = 0; i < neurons; ++i)
                for (std::size_t j = 0; j < batch_size; ++j)
                    potentials[j, i] += biases.data[i];
        }

        weights_t weights;
        biases_t biases;
        neurons_t potentials, activations;
    };

    using hidden_layer_t = layer_t<input_size, 4, random_t::normal_he>;
    using output_layer_t = layer_t<hidden_layer_t::neurons, output_size, random_t::normal_glorot>;

public:
    output_t evaluate(const input_t& inputs) {
        hidden.compute_inner_potentials(inputs);
        hidden.activations = apply<ReLU>(hidden.potentials);
        output.compute_inner_potentials(hidden.activations);
        output.activations = apply<sigmoid>(output.potentials);
        return output.activations;
    }

    void backpropagation(const input_t& inputs, const matrix<batch_size, 1>& labels, float learning_rate) {
        // Forward pass
        output_t outputs = evaluate(inputs);

        // Backward pass
        output_t output_dE_dy;
        for (unsigned i = 0; i < batch_size; ++i)
            output_dE_dy.data[i] = (1 - labels.data[i]) / (1 - outputs.data[i]) - labels.data[i] / outputs.data[i];

        auto output_dE_dy_dsigma = pmult(apply<dsigmoid>(output.potentials), output_dE_dy);
        auto hidden_dE_dy = output.weights.transpose() * output_dE_dy_dsigma;

        output_layer_t::weights_t output_dE_dw = output_dE_dy_dsigma * hidden.activations.transpose();
        auto ones = matrix<1, batch_size>::ones();
        output_layer_t::biases_t output_dE_dbias = output_dE_dy_dsigma * ones;

        auto hidden_dE_dy_dsigma = pmult(apply<dReLU>(hidden.potentials), hidden_dE_dy);
        hidden_layer_t::weights_t hidden_dE_dw = hidden_dE_dy_dsigma * inputs.transpose();
        hidden_layer_t::biases_t hidden_dE_dbias = hidden_dE_dy_dsigma * ones;

        output.weights -= learning_rate * output_dE_dw;
        output.biases -= learning_rate * output_dE_dbias;
        hidden.weights -= learning_rate * hidden_dE_dw;
        hidden.biases -= learning_rate * hidden_dE_dbias;
    }

private:
    hidden_layer_t hidden;
    output_layer_t output;
};



#endif //NETWORK_H
