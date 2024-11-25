#ifndef CSV_READER_H
#define CSV_READER_H
#include <fstream>
#include <sstream>

#include "matrix.h"


class csv_reader {
public:
    enum class csv_t { labels, images };
    static constexpr unsigned image_size = 28 * 28;

    explicit csv_reader(const std::string &file_path) : file{std::ifstream{file_path}} {
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + file_path);
    }

    int get_label() {
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);
        int label;
        iss >> label;
        return label;
    }

    matrix<1, image_size> get_image() {
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);
        matrix<1, image_size> result;
        for (auto &x : result.data) {
            iss >> x;
            iss.get();
        }
        return result;
    }

private:
    std::ifstream file;
};


#endif //CSV_READER_H
