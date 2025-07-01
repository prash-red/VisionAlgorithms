#pragma once

#include <array>
#include <iostream>

using namespace std;

/**
 * @class Linalg
 * @brief Provides static linear algebra utilities for solving systems of linear equations.
 *
 * This class implements Gaussian elimination with partial pivoting to solve
 * systems of linear equations represented in matrix form.
 */
class Linalg {
    public:
    /**
     * @brief Solves a system of linear equations using Gaussian elimination.
     *
     * @tparam rows Number of equations (rows in the matrix).
     * @tparam columns Number of unknowns (columns in the matrix).
     * @param equationMatrix The coefficient matrix, stored as a flat array of size rows * columns.
     * @param constantMatrix The constants on the right-hand side, as an array of size columns.
     * @return An array containing the solution vector. If the matrix is singular, returns an empty array.
     */
    template <size_t rows, size_t columns>
    static array<float, columns> linalgSolve (const array<float, rows * columns>& equationMatrix,
    const array<float, columns>& constantMatrix) {
        array<float, rows*(columns + 1)> augmentedMatrix = {};
        for (int i = 0; i < static_cast<int> (rows); ++i) {
            for (int j = 0; j < static_cast<int> (columns); ++j) {
                augmentedMatrix[i * (columns + 1) + j] = equationMatrix[i * columns + j];
            }
            augmentedMatrix[i * (columns + 1) + columns] = constantMatrix[i];
        }

        if (const int singularFlag = forwardElim<rows, columns + 1> (augmentedMatrix, rows);
        singularFlag != -1) {
            std::cerr << "Matrix is singular, cannot solve." << std::endl;
            return {};
        }

        array<float, rows> solution = {};
        backSubstitution<rows, columns + 1> (augmentedMatrix, solution);
        return solution;
    }

    /**
     * @brief Prints a matrix to the standard output.
     *
     * @tparam rows Number of rows in the matrix.
     * @tparam columns Number of columns in the matrix.
     * @param matrix The matrix to print, stored as a flat array.
     */
    template <size_t rows, size_t columns>
    static void printMatrix (const array<float, rows * columns>& matrix) {
        for (int i = 0; i < static_cast<int> (rows); ++i) {
            for (int j = 0; j < static_cast<int> (columns); ++j) {
                std::cout << matrix[i * columns + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    private:
    /**
     * @brief Swaps two rows in a matrix.
     *
     * @tparam rows Number of rows in the matrix.
     * @tparam columns Number of columns in the matrix.
     * @param matrix The matrix to operate on.
     * @param row_1 The first row to swap.
     * @param row_2 The second row to swap.
     */
    template <size_t rows, size_t columns>
    static void swapRow (array<float, rows * columns>& matrix, int row_1, int row_2) {
        for (int col = 0; col < static_cast<int> (columns); ++col) {
            std::swap (matrix[row_1 * columns + col], matrix[row_2 * columns + col]);
        }
    }

    /**
     * @brief Performs forward elimination to convert the matrix to upper triangular form.
     *
     * @tparam rows Number of rows in the matrix.
     * @tparam columns Number of columns in the matrix.
     * @param matrix The augmented matrix to operate on.
     * @param numUnknowns The number of unknowns (used for iteration).
     * @return -1 if successful, otherwise the index of the singular row.
     */
    template <size_t rows, size_t columns>
    static int forwardElim (array<float, rows * columns>& matrix, int numUnknowns) {
        for (int k = 0; k < numUnknowns; k++) {
            int maxIndex   = k;
            float maxValue = abs (matrix[k * columns + k]);
            for (int i = k + 1; i < numUnknowns; i++) {
                if (abs (matrix[i * columns + k]) > maxValue) {
                    maxValue = abs (matrix[i * columns + k]);
                    maxIndex = i;
                }
            }
            if (matrix[maxIndex * columns + k] == 0) {
                return k;
            }
            if (maxIndex != k) {
                swapRow<rows, columns> (matrix, k, maxIndex);
            }
            for (int i = k + 1; i < numUnknowns; i++) {
                float f = matrix[i * columns + k] / matrix[k * columns + k];
                for (int j = k; j < static_cast<int> (columns); j++) {
                    matrix[i * columns + j] -= matrix[k * columns + j] * f;
                }
            }
        }
        return -1;
    }

    /**
     * @brief Performs back substitution to solve for the unknowns after forward elimination.
     *
     * @tparam rows Number of rows in the matrix.
     * @tparam columns Number of columns in the matrix.
     * @param matrix The upper triangular augmented matrix.
     * @param solution The array to store the solution vector.
     */
    template <size_t rows, size_t columns>
    static void backSubstitution (array<float, rows * columns>& matrix,
    array<float, rows>& solution) {
        for (int i = rows - 1; i >= 0; --i) {
            solution[i] = matrix[i * columns + columns - 1];
            for (int j = i + 1; j < static_cast<int> (rows); ++j) {
                solution[i] -= matrix[i * columns + j] * solution[j];
            }
            solution[i] /= matrix[i * columns + i];
        }
    }
};