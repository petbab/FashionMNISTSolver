#ifndef MATH_H
#define MATH_H

#include "matrix.h"

#include <cmath>


inline float linear(float x) {
    return x;
}

inline float dlinear(float) {
    return 1;
}

inline float ReLU(float x) {
    return x > 0 ? x : 0;
}

inline float dReLU(float x) {
    return x > 0 ? 1 : 0;
}

inline float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

inline float dsigmoid(float x) {
    float e_x = std::exp(-x);
    return e_x / ((1 + e_x) * (1 + e_x));
}

typedef float(*unary_f)(float);

template<unary_f F, unsigned COLS, unsigned ROWS>
matrix<COLS, ROWS> apply(const matrix<COLS, ROWS>& xs) {
    matrix<COLS, ROWS> result;
    for (std::size_t i = 0; i < ROWS * COLS; ++i)
        result[i] = F(xs[i]);
    return result;
}

template<unsigned BATCH_SIZE, unsigned NEURONS>
matrix<BATCH_SIZE, NEURONS> softmax(const matrix<BATCH_SIZE, NEURONS> &potentials) {
    matrix<BATCH_SIZE, NEURONS> result;
    for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch) {
        float denom = 0.;
        for (std::size_t neuron = 0; neuron < NEURONS; ++neuron)
            denom += std::exp(potentials[batch, neuron]);

        for (std::size_t neuron = 0; neuron < NEURONS; ++neuron)
            result[batch, neuron] = std::exp(potentials[batch, neuron]) / denom;
    }
    return result;
}

template<unsigned BATCH_SIZE, unsigned NEURONS>
matrix<BATCH_SIZE, NEURONS> dsoftmax(const matrix<BATCH_SIZE, NEURONS> &potentials) {
    matrix<BATCH_SIZE, NEURONS> result;
    for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch) {
        float sum = 0.;
        for (std::size_t neuron = 0; neuron < NEURONS; ++neuron)
            sum += std::exp(potentials[batch, neuron]);

        for (std::size_t neuron = 0; neuron < NEURONS; ++neuron)
            result[batch, neuron] = (sum - std::exp(potentials[batch, neuron])) / (sum * sum);
    }
    return result;
}

#endif //MATH_H
