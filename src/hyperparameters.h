#ifndef PV021_HYPERPARAMETERS_H
#define PV021_HYPERPARAMETERS_H

namespace cfg {

static constexpr double learning_rate = .001;
static constexpr double momentum = .6;
static constexpr double weight_decay = 1. - 1e-8;

static constexpr unsigned hidden1_neurons = 64;
static constexpr unsigned hidden2_neurons = 32;

static constexpr unsigned batch_size = 25;

}

#endif //PV021_HYPERPARAMETERS_H
