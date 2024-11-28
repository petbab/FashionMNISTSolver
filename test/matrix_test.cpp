#include <gtest/gtest.h>
#include <cmath>
#include "../src/matrix.h"

class MatrixMultiplicationTest : public ::testing::Test {
protected:
    template<unsigned COLS, unsigned ROWS>
    static void expect_equal(const matrix<COLS, ROWS>& result, const matrix<COLS, ROWS>& expected) {
        for (std::size_t col = 0; col < COLS; ++col)
            for (std::size_t row = 0; row < ROWS; ++row)
                EXPECT_FLOAT_EQ((result[col, row]), (expected[col, row]))
                    << "Mismatch at position: [" << col << ", " << row << "].";
    }
};

TEST_F(MatrixMultiplicationTest, BasicMultiplication) {
    matrix<3, 2> m1{{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    }};

    matrix<3, 3> m2{{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    }};

    matrix<3, 2> expected{{
        30.0f, 36.0f, 42.0f,
        66.0f, 81.0f, 96.0f,
    }};

    auto result = m1 * m2;
    expect_equal(result, expected);
}

TEST_F(MatrixMultiplicationTest, IdentityMultiplication) {
    // Test multiplication with identity matrix
    matrix<2, 2> m1{{
        1.0f, 2.0f,
        3.0f, 4.0f
    }};

    matrix<2, 2> identity{{
        1.0f, 0.0f,
        0.0f, 1.0f
    }};

    auto result = m1 * identity;
    expect_equal(result, m1);
}

TEST_F(MatrixMultiplicationTest, ZeroMatrix) {
    // Test multiplication with zero matrix
    matrix<2, 3> m1{{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }};

    matrix<3, 2> zero{{
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f
    }};

    matrix<2, 2> expected{{
        0.0f, 0.0f,
        0.0f, 0.0f,
    }};

    auto result = zero * m1;
    expect_equal(result, expected);
}

TEST_F(MatrixMultiplicationTest, SingleElementMatrices) {
    // Test multiplication of 1x1 matrices
    matrix<1, 1> m1{{2.0f}};
    matrix<1, 1> m2{{3.0f}};
    matrix<1, 1> expected{{6.0f}};

    auto result = m1 * m2;
    expect_equal(result, expected);
}