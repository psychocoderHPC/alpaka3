/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "helper.hpp"
#include "misc.hpp"
#include "simd_access_sh.hpp"

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <experimental/simd>

namespace stdx = std::experimental;

namespace sa
{
    template<class T, class Op>
    requires(!simd_access::simd_arithmetic<T>)
    auto reduce(T const& x, Op op)
    {
        return stdx::reduce(x, op);
    }

    template<class T, class Op>
    requires(simd_access::simd_arithmetic<T>)
    auto reduce(T const& x, Op)
    {
        return x;
    }

    template<typename T>
    void haccmk(
        int const repeat,
        size_t const n, // global size
        int const ilp, // inner loop count
        T const fsrrmax,
        T const mp_rsm,
        T const fcoeff,
        T const* __restrict xx,
        T const* __restrict yy,
        T const* __restrict zz,
        T const* __restrict mass,
        T* __restrict vx2,
        T* __restrict vy2,
        T* __restrict vz2)
    {
        float total_time = 0.f;

        for(int r = 0; r < repeat; r++)
        {
            auto start = std::chrono::steady_clock::now();

#if defined(_OPENMP)
#    pragma omp target teams distribute parallel for
#endif
            for(int i = 0; i < n; i++)
            {
                constexpr float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313,
                                ma4 = 0.0000605491, ma5 = -0.00000147177;

                using SimdModel = stdx::native_simd<float>;
                SimdModel xi = 0.f;
                SimdModel yi = 0.f;
                SimdModel zi = 0.f;

                float xxi = xx[i];
                float yyi = yy[i];
                float zzi = zz[i];

                simd_access::loop<SimdModel>(
                    0,
                    ilp,
                    [&](auto j)
                    {
                        auto dxc = SIMD_ACCESS(xx, j) - xxi;
                        auto dyc = SIMD_ACCESS(yy, j) - yyi;
                        auto dzc = SIMD_ACCESS(zz, j) - zzi;


                        auto r2 = dxc * dxc + dyc * dyc + dzc * dzc;
                        using var_type = simd_access::simdized_by_index_t<float, decltype(j)>;
                        var_type m(.0f);
                        stdx::where(r2 < fsrrmax, m) = SIMD_ACCESS(mass, j);

                        auto tmp = r2 + mp_rsm;
                        using std::sqrt;
                        auto p = 1.f / (tmp * sqrt(tmp));

                        auto f = p - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5)))));
                        var_type fac(.0f);
                        stdx::where(r2 > 0.0f, fac) = m * f;

                        if constexpr(std::same_as<float, decltype(xi)>)
                        {
                            xi[0] += fac * dxc;
                            yi[0] += fac * dyc;
                            zi[0] += fac * dzc;
                        }
                        else
                        {
                            xi += fac * dxc;
                            yi += fac * dyc;
                            zi += fac * dzc;
                        }
                    });
                /* No `+=` is used.
                 * This differs to the original publication but is necessary due to the OpenMP bug.
                 */
                vx2[i] = sa::reduce(xi, std::plus{}) * fcoeff;
                vy2[i] = sa::reduce(yi, std::plus{}) * fcoeff;
                vz2[i] = sa::reduce(zi, std::plus{}) * fcoeff;
            }

            auto end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            total_time += time;
        }
        printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);
    }
} // namespace sa

int main(int argc, char* argv[])
{
    int n2 = 1;
    int n1 = 1;
    int repeat = 1;

    if(int const ret = hacc::parseCmd(argc, argv, n2, n1, repeat))
        return ret;

    float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2;
    int i;

    printf("Outer loop count is set %d\n", n1);
    printf("Inner loop count is set %d\n", n2);

    float* xx = hacc::allocAligned<float>(n2);
    float* yy = hacc::allocAligned<float>(n2);
    float* zz = hacc::allocAligned<float>(n2);
    float* mass = hacc::allocAligned<float>(n2);
    float* vx2 = hacc::allocAligned<float>(n2);
    float* vy2 = hacc::allocAligned<float>(n2);
    float* vz2 = hacc::allocAligned<float>(n2);
    float* vx2_hw = hacc::allocAligned<float>(n2);
    float* vy2_hw = hacc::allocAligned<float>(n2);
    float* vz2_hw = hacc::allocAligned<float>(n2);

    /* Initial data preparation */
    fcoeff = 0.23f;
    fsrrmax2 = 0.5f;
    mp_rsm2 = 0.03f;
    dx1 = 1.0f / (float) n2;
    dy1 = 2.0f / (float) n2;
    dz1 = 3.0f / (float) n2;
    xx[0] = 0.f;
    yy[0] = 0.f;
    zz[0] = 0.f;
    mass[0] = 2.f;

    for(i = 1; i < n2; i++)
    {
        xx[i] = xx[i - 1] + dx1;
        yy[i] = yy[i - 1] + dy1;
        zz[i] = zz[i - 1] + dz1;
        mass[i] = (float) i * 0.01f + xx[i];
    }

    for(i = 0; i < n2; i++)
    {
        vx2[i] = 0.f;
        vy2[i] = 0.f;
        vz2[i] = 0.f;
        vx2_hw[i] = 0.f;
        vy2_hw[i] = 0.f;
        vz2_hw[i] = 0.f;
    }

    for(i = 0; i < n1; ++i)
    {
        hacc::haccmk_gold(n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2);
        vx2[i] = vx2[i] + dx2 * fcoeff;
        vy2[i] = vy2[i] + dy2 * fcoeff;
        vz2[i] = vz2[i] + dz2 * fcoeff;
    }

    sa::haccmk<float>(repeat, n1, n2, fsrrmax2, mp_rsm2, fcoeff, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw);

    int error = hacc::verify(n2, vx2, vy2, vz2, vx2_hw, vy2_hw, vz2_hw);

    hacc::freeAligned(xx);
    hacc::freeAligned(yy);
    hacc::freeAligned(zz);
    hacc::freeAligned(mass);
    hacc::freeAligned(vx2);
    hacc::freeAligned(vy2);
    hacc::freeAligned(vz2);
    hacc::freeAligned(vx2_hw);
    hacc::freeAligned(vy2_hw);
    hacc::freeAligned(vz2_hw);

    return error;
}
