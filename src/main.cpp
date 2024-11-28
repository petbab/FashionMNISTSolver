#include <iostream>

#include "csv.h"
#include "network.h"
#include "matrix.h"

int main() {
#ifdef DEBUG
    std::cout << "DEBUG\n";
#endif
#ifdef RELEASE
    std::cout << "RELEASE\n";
#endif

    float learning_rate = 0.5;
    //
    // network net;
    // for (int t = 0; t < 1000; ++t) {
    //     if (t % 100 == 0) {
    //         std::cout << "\nt = " << t << "\n";
    //         auto r = net.evaluate(xor_problem);
    //         for (std::size_t int i = 0; i < 4; ++i)
    //             std::cout << '[' << xor_problem[i, 0] << ", " << xor_problem[i, 1] << "]: " << r[i, 0] << std::endl;
    //     }
    //     // net.backpropagation<4>(xor_problem, labels, learning_rate * 1. / static_cast<float>(t + 1));
    //     net.backpropagation(xor_problem, labels, learning_rate);
    // }

    csv inputs{"./data/fashion_mnist_train_vectors.csv", csv::mode_t::read};
    csv labels{"./data/fashion_mnist_train_labels.csv", csv::mode_t::read};

    network net;
    auto input = inputs.read_batch();
    auto label = labels.read_batch_labels();
    net.evaluate(input).print();
    net.backpropagation(input, label, learning_rate);
    net.evaluate(input).print();


    csv out{"my_predictions.csv", csv::mode_t::write};


    return 0;
}
