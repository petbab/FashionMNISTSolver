#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <random>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>


namespace init {

static std::default_random_engine engine{42};

template<unsigned IN, unsigned N>
float normal_glorot() {
    static std::normal_distribution d(0., 2. / static_cast<float>(IN + N));
    return d(engine);
}

template<unsigned IN>
float normal_he() {
    static std::normal_distribution d(0., 2. / static_cast<float>(IN));
    return d(engine);
}

}

enum class random_t { normal_glorot, normal_he };

template<unsigned COLS, unsigned ROWS>
struct matrix {
    matrix() : data{std::make_unique<data_t>()} {}
    explicit matrix(std::array<float, COLS * ROWS> data)
            : data{std::make_unique<data_t>(std::move(data))} {}

    matrix(matrix&& other) noexcept : data{std::move(other.data)} {}
    matrix& operator=(matrix&& other) noexcept {
        data = std::move(other.data);
        return *this;
    }
    matrix(const matrix& other) : data{std::make_unique<data_t>(*other.data)} {}
    matrix& operator=(const matrix& other) {
        data = std::make_unique<data_t>(*other.data);
        return *this;
    }
    ~matrix() = default;

    void print() const {
        for (std::size_t row = 0; row < ROWS; ++row) {
            for (std::size_t col = 0; col < COLS; ++col) {
                std::cout << (*data)[row * COLS + col] << ", ";
            }
            std::cout << std::endl;
        }
    }

    matrix<ROWS, COLS> transpose() const {
        matrix<ROWS, COLS> transposed;
        for (std::size_t col = 0; col < COLS; ++col)
            for (std::size_t row = 0; row < ROWS; ++row)
                transposed[col * ROWS + row] = (*data)[row * COLS + col];
        return transposed;
    }

    template<unsigned NEW_COLS>
    friend matrix<NEW_COLS, ROWS> operator*(const matrix<COLS, ROWS>& l, const matrix<NEW_COLS, COLS>& r) {
        matrix<NEW_COLS, ROWS> result;
        for (std::size_t l_row = 0; l_row < ROWS; ++l_row) {
            for (std::size_t r_col = 0; r_col < NEW_COLS; ++r_col) {
                float sum = 0;
                for (std::size_t l_col = 0; l_col < COLS; ++l_col)
                    sum += l[l_col, l_row] * r[r_col, l_col];
                result[r_col, l_row] = sum;
            }
        }
        return result;
    }

    /**
     * Performs piecewise multiplication.
     * @param a left matrix
     * @param b right matrix
     * @return piecewise product of `a` and `b`
     */
    friend matrix<COLS, ROWS> pmult(matrix<COLS, ROWS> a, const matrix<COLS, ROWS>& b) {
        for (std::size_t i = 0; i < ROWS * COLS; ++i)
            a[i] *= b[i];
        return a;
    }

    matrix& operator*=(float x) {
        for (float& y : *data)
            y *= x;
        return *this;
    }
    friend matrix operator*(float x, matrix m) {
        m *= x;
        return m;
    }

    matrix& operator+=(const matrix& m) {
        for (std::size_t i = 0; i < ROWS * COLS; ++i)
            (*data)[i] += m[i];
        return *this;
    }
    friend matrix operator+(matrix m1, const matrix& m2) {
        m1 += m2;
        return m1;
    }

    matrix& operator-=(const matrix& m) {
        for (std::size_t i = 0; i < ROWS * COLS; ++i)
            (*data)[i] -= m[i];
        return *this;
    }
    friend matrix operator-(matrix m1, const matrix& m2) {
        m1 -= m2;
        return m1;
    }

    float operator[](std::size_t col, std::size_t row) const {
        return (*data)[row * COLS + col];
    }
    float& operator[](std::size_t col, std::size_t row) {
        return (*data)[row * COLS + col];
    }

    float operator[](std::size_t i) const {
        return (*data)[i];
    }
    float& operator[](std::size_t i) {
        return (*data)[i];
    }

    static matrix random(random_t rt) {
        matrix m{};
        for (std::size_t i = 0; i < COLS * ROWS; ++i) {
            switch (rt) {
                case random_t::normal_glorot:
                    m[i] = init::normal_glorot<COLS, ROWS>();
                    break;
                case random_t::normal_he:
                    m[i] = init::normal_he<COLS>();
                    break;
            }
        }

        return m;
    }

    static matrix ones() {
        matrix m{};
        std::ranges::fill(*m.data, 1.f);
        return m;
    }

    static constexpr unsigned cols = COLS, rows = ROWS;

private:
    using data_t = std::array<float, ROWS * COLS>;
    std::unique_ptr<data_t> data;
};

#endif //MATRIX_H
