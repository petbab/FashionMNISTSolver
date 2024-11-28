#ifndef CSV_READER_H
#define CSV_READER_H
#include <fstream>

#include "matrix.h"
#include "network.h"


class csv {
public:
    enum class mode_t { read, write };

    csv(const std::string &file_path, mode_t mode) {
        switch (mode) {
        case mode_t::read:
            file = std::fstream{file_path, std::ios::in};
            break;
        case mode_t::write:
            file = std::fstream{file_path, std::ios::out | std::ios::trunc};
            break;
        }

        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + file_path);
    }

    network::labels_t read_batch_labels() {
        network::labels_t result;
        for (std::size_t batch = 0; batch < network::batch_size; ++batch)
            file >> result[batch];
        return result;
    }

    matrix<network::batch_size, network::input_size> read_batch() {
        matrix<network::batch_size, network::input_size> result;

        std::string line;
        for (std::size_t batch = 0; batch < network::batch_size; ++batch) {
            for (std::size_t i = 0; i < network::input_size; ++i) {
                float color;
                file >> color;
                result[batch, i] = color / 255.f;
                file.get();
            }
        }

        return result;
    }

    void write_batch(const network::labels_t& batch) {
        for (unsigned label : batch)
            file << label << '\n';
    }

private:
    std::fstream file;
};


#endif //CSV_READER_H
