/* Copyright 2025 Philippe Felix Haupt, Eric-Ramon Kreyer, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>

void naive_matrix_mult(auto A, auto B, auto C, float const alpha, float const beta)
{
    auto A_dim = A.getExtents();
    auto C_dim = C.getExtents();
    for(uint32_t j = 0; j < C_dim.y(); ++j)
    {
        for(uint32_t i = 0; i < C_dim.x(); ++i)
        {
            float sum = 0.0;
            for(uint32_t k = 0; k < A_dim.x(); ++k)
            {
                sum += A[Vec2D{j, k}] * B[Vec2D{k, i}];
            }
            C[Vec2D{j, i}] = alpha * sum + beta * C[Vec2D{j, i}];
        }
    }
}
