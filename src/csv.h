#ifndef CSV_READER_H
#define CSV_READER_H
#include <fstream>

#include "matrix.h"


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

    template<class LABELS_T, unsigned BATCH_SIZE>
    LABELS_T read_batch_labels() {
        LABELS_T result;
        for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch)
            file >> result[batch];
        return result;
    }

    template<unsigned BATCH_SIZE, unsigned INPUT_SIZE>
    matrix<BATCH_SIZE, INPUT_SIZE> read_batch() {
        matrix<BATCH_SIZE, INPUT_SIZE> result;

        for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch) {
            for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
                float color;
                file >> color;
                result[batch, i] = color / 255.f;
                file.get();
            }
        }

        return result;
    }

    void write_batch(const auto& batch) {
        for (unsigned label : batch)
            file << label << '\n';
    }

    void seek_begin() {
        file.seekg(0);
    }

private:
    std::fstream file;
};


#endif //CSV_READER_H
