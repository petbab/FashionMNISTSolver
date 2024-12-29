#ifndef CSV_READER_H
#define CSV_READER_H
#include <fstream>

#include "matrix.h"


class csv_reader {
public:
    explicit csv_reader(const std::string &file_path) : file{file_path} {
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + file_path);
    }

    void seek_begin() {
        file.seekg(0);
    }

protected:
    std::ifstream file;
};

class csv_image_reader : public csv_reader {
    explicit csv_image_reader(const std::string &file_path)
            : csv_reader(file_path), mean{0.}, sd{0.} {}

public:
    csv_image_reader(const std::string &file_path, double mean, double sd)
        : csv_reader(file_path), mean{mean}, sd{sd} {}

    /**
     * @brief Creates a CSV image reader with pre-calculated mean and standard deviation statistics
     *
     * @tparam DATASET_SIZE Total number of images in the dataset
     * @tparam BATCH_SIZE Number of images to read in each batch
     * @tparam INPUT_SIZE Size of each input image (number of pixels)
     *
     * @param file_path Path to the CSV file containing image data
     *
     * @return csv_image_reader Instance initialized with calculated mean and standard deviation
     *
     * @details This function performs two passes over the dataset:
     *          1. First pass calculates the mean across all pixels/features
     *          2. Second pass calculates the standard deviation using the computed mean
     *
     * The statistics are stored in the reader's mean and sd members respectively.
     * After calculation, the reader is reset to the beginning of the file.
     */
    template<unsigned DATASET_SIZE, unsigned BATCH_SIZE, unsigned INPUT_SIZE>
    static csv_image_reader with_statistics(const std::string &file_path) {
        csv_image_reader reader{file_path};

        for (std::size_t i = 0; i < DATASET_SIZE / BATCH_SIZE; ++i) {
            matrix batch = reader.read_batch<BATCH_SIZE, INPUT_SIZE>(false);
            for (std::size_t j = 0; j < BATCH_SIZE * INPUT_SIZE; ++j)
                reader.mean += batch[j];
        }
        auto n = static_cast<double>(DATASET_SIZE * INPUT_SIZE);
        reader.mean /= n;
        reader.seek_begin();

        for (std::size_t i = 0; i < DATASET_SIZE / BATCH_SIZE; ++i) {
            matrix batch = reader.read_batch<BATCH_SIZE, INPUT_SIZE>(false);
            for (std::size_t j = 0; j < BATCH_SIZE * INPUT_SIZE; ++j)
                reader.sd += std::pow(batch[j] - reader.mean, 2.);
        }
        reader.sd = std::sqrt(reader.sd / n);
        reader.seek_begin();

        return reader;
    }

    template<unsigned BATCH_SIZE, unsigned INPUT_SIZE>
    matrix<BATCH_SIZE, INPUT_SIZE> read_batch(bool normalize = true) {
        matrix<BATCH_SIZE, INPUT_SIZE> result;

        for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch) {
            for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
                file >> result[batch, i];
                file.get();
            }
        }

        if (normalize)
            result.normalize(mean, sd);
        return result;
    }

    double get_mean() const { return mean; }
    double get_sd() const { return sd; }

private:
    double mean, sd;
};

class csv_label_reader : public csv_reader {
public:
    using csv_reader::csv_reader;

    template<class LABELS_T, unsigned BATCH_SIZE>
    LABELS_T read_batch() {
        LABELS_T result;
        for (std::size_t batch = 0; batch < BATCH_SIZE; ++batch)
            file >> result[batch];
        return result;
    }
};

class csv_writer {
public:
    explicit csv_writer(const std::string &file_path) : file{file_path, std::ios::out | std::ios::trunc} {
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + file_path);
    }

    void write_batch(const auto& batch) {
        for (unsigned label : batch)
            file << label << '\n';
    }

private:
    std::ofstream file;
};


#endif //CSV_READER_H
