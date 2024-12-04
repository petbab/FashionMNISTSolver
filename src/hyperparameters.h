#ifndef PV021_HYPERPARAMETERS_H
#define PV021_HYPERPARAMETERS_H

namespace cfg {

static constexpr float learning_rate = 0.001f;
static constexpr float momentum = 0.5f;
static constexpr float weight_decay = 0.9999999f;
static constexpr unsigned hidden1_neurons = 64;
static constexpr unsigned hidden2_neurons = 32;
static constexpr unsigned batch_size = 16;

}

#endif //PV021_HYPERPARAMETERS_H
