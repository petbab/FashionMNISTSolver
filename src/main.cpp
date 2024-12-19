#include "csv.h"
#include "network.h"
#include "timer.h"

#include <iostream>


int main() {
    timer t{std::cout, "Total time"};

    csv_image_reader train_inputs = csv_image_reader::with_statistics<
            network::training_set_size, cfg::batch_size, network::input_size
        >("data/fashion_mnist_train_vectors.csv");
    csv_label_reader labels{"data/fashion_mnist_train_labels.csv"};

    network net;
    net.learn(train_inputs, labels);

    csv_image_reader test_inputs{"data/fashion_mnist_test_vectors.csv",
                                 train_inputs.get_mean(), train_inputs.get_sd()};
    csv_writer test_out{"test_predictions.csv"};
    net.predict<network::test_set_size>(test_inputs, test_out);

    csv_writer train_out{"train_predictions.csv"};
    net.predict<network::training_set_size + network::validation_set_size>(train_inputs, train_out);

    return 0;
}
