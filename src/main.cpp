#include "csv.h"
#include "network.h"
#include "timer.h"

#include <iostream>


int main() {
    timer t{std::cout, "Total time"};

    csv train_inputs{"data/fashion_mnist_train_vectors.csv", csv::mode_t::read};
    csv labels{"data/fashion_mnist_train_labels.csv", csv::mode_t::read};

    network net;
    net.learn(train_inputs, labels);

    csv test_inputs{"data/fashion_mnist_test_vectors.csv", csv::mode_t::read};
    csv test_out{"test_predictions.csv", csv::mode_t::write};
    net.predict<network::test_set_size>(test_inputs, test_out);

    csv train_out{"train_predictions.csv", csv::mode_t::write};
    net.predict<network::training_set_size + network::validation_set_size>(train_inputs, train_out);

    return 0;
}
