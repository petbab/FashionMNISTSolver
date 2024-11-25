#include <iostream>

#include "network.h"
#include "matrix.h"

int main() {
#ifdef DEBUG
    std::cout << "DEBUG\n";
#endif
#ifdef RELEASE
    std::cout << "RELEASE\n";
#endif

    matrix<4, 2> xor_problem{{
        0, 0, 1, 1,
        0, 1, 0, 1,
    }};
    matrix<4, 1> labels{{0., 1., 1., 0.}};
    float learning_rate = 0.5;

    network net;
    for (int t = 0; t < 1000; ++t) {
        if (t % 100 == 0) {
            std::cout << "\nt = " << t << "\n";
            auto r = net.evaluate(xor_problem);
            for (unsigned int i = 0; i < 4; ++i)
                std::cout << '[' << xor_problem[i, 0] << ", " << xor_problem[i, 1] << "]: " << r[i, 0] << std::endl;
        }
        // net.backpropagation<4>(xor_problem, labels, learning_rate * 1. / static_cast<float>(t + 1));
        net.backpropagation(xor_problem, labels, learning_rate);
    }

    return 0;
}
