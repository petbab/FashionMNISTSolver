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
    for (unsigned i = 0; i < ROWS * COLS; ++i)
        result.data[i] = F(xs.data[i]);
    return result;
}

void softmax(auto &inner_potentials) {
    float denom = 0.;
    for (const auto &p : inner_potentials)
        denom += std::exp(p);

    for (float &ip : inner_potentials)
        ip = std::exp(ip) / denom;
}


#endif //MATH_H
