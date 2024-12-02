#include <iostream>

#include "csv.h"
#include "network.h"


int main() {
#ifdef DEBUG
    std::cout << "DEBUG\n";
#endif
#ifdef RELEASE
    std::cout << "RELEASE\n";
#endif

    csv train_inputs{"data/fashion_mnist_train_vectors.csv", csv::mode_t::read};
    csv labels{"data/fashion_mnist_train_labels.csv", csv::mode_t::read};
    csv train_out{"train_predictions.csv", csv::mode_t::write};

    network net;
    net.learn(train_inputs, labels, train_out);

    csv test_inputs{"data/fashion_mnist_test_vectors.csv", csv::mode_t::read};
    csv test_out{"test_predictions.csv", csv::mode_t::write};
    net.predict(test_inputs, test_out);

    return 0;
}
