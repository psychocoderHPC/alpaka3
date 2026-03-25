#include "helper.hpp"
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

    void Step10_sa(
        int count1,
        float xxi,
        float yyi,
        float zzi,
        float fsrrmax2,
        float mp_rsm2,
        float* xx1,
        float* yy1,
        float* zz1,
        float* mass1,
        float* dxi,
        float* dyi,
        float* dzi)
    {
        float const ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491,
                    ma5 = -0.00000147177;

        using SimdModel = stdx::native_simd<float>;
        SimdModel xi = 0.f;
        SimdModel yi = 0.f;
        SimdModel zi = 0.f;

        simd_access::loop<SimdModel>(
            0,
            count1,
            [&](auto j)
            {
                auto dxc = SIMD_ACCESS(xx1, j) - xxi;
                auto dyc = SIMD_ACCESS(yy1, j) - yyi;
                auto dzc = SIMD_ACCESS(zz1, j) - zzi;


                auto r2 = dxc * dxc + dyc * dyc + dzc * dzc;
                using var_type = simd_access::simdized_by_index_t<float, decltype(j)>;
                var_type m(.0f);
                stdx::where(r2 < fsrrmax2, m) = SIMD_ACCESS(mass1, j);

                auto tmp = r2 + mp_rsm2;
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
        *dxi = sa::reduce(xi, std::plus{});
        *dyi = sa::reduce(yi, std::plus{});
        *dzi = sa::reduce(zi, std::plus{});
    }
} // namespace sa

#define NC 16'777'216
#define ETOL 1.e-4 /* Tolerance for correctness */

int main(int argc, char* argv[])
{
    size_t numElements = 1;
    size_t numberOfRuns = 1;

    if(int const ret = hacc::parseCmd(argc, argv, numElements, numberOfRuns))
        return ret;

    float* xx = hacc::allocAlligned<float>(numElements);
    float* yy = hacc::allocAlligned<float>(numElements);
    float* zz = hacc::allocAlligned<float>(numElements);
    float* mass = hacc::allocAlligned<float>(numElements);
    float* vx1 = hacc::allocAlligned<float>(numElements);
    float* vy1 = hacc::allocAlligned<float>(numElements);
    float* vz1 = hacc::allocAlligned<float>(numElements);

    float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1;

    [[maybe_unused]] char M1[NC], M2[NC];
    int n, count, i, rank, nprocs;
    double elapsed = 0.0, validation, final;

    rank = 0;
    nprocs = 1;

    count = 327;

    if(rank == 0)
    {
        printf("count is set %d\n", count);
        printf("Total MPI ranks %d\n", nprocs);
    }

#pragma omp parallel
    {
        if((rank == 0) && (omp_get_thread_num() == 0))
        {
            printf("Number of OMP threads %d\n\n", omp_get_num_threads());
            // printf( "      N         Time,us        Validation result\n" );
        }
    }

    auto tm3 = std::chrono::high_resolution_clock::now();

    final = 0.;

    for(n = 400; n < numElements; n = n + 20)
    {
        /* Initial data preparation */
        fcoeff = 0.23f;
        fsrrmax2 = 0.5f;
        mp_rsm2 = 0.03f;
        dx1 = 1.0f / (float) n;
        dy1 = 2.0f / (float) n;
        dz1 = 3.0f / (float) n;
        xx[0] = 0.f;
        yy[0] = 0.f;
        zz[0] = 0.f;
        mass[0] = 2.f;

        for(i = 1; i < n; i++)
        {
            xx[i] = xx[i - 1] + dx1;
            yy[i] = yy[i - 1] + dy1;
            zz[i] = zz[i - 1] + dz1;
            mass[i] = (float) i * 0.01f + xx[i];
        }

        for(i = 0; i < n; i++)
        {
            vx1[i] = 0.f;
            vy1[i] = 0.f;
            vz1[i] = 0.f;
        }

        /* Data preparation done */


        /* Clean L1 cache */
        for(i = 0; i < NC; i++)
            M1[i] = 4;
        for(i = 0; i < NC; i++)
            M2[i] = M1[i];

        auto t1 = std::chrono::high_resolution_clock::now();


#pragma omp parallel for private(dx1, dy1, dz1)
        for(i = 0; i < count; ++i)
        {
            sa::Step10_sa(n, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx1, &dy1, &dz1);

            vx1[i] = vx1[i] + dx1 * fcoeff;
            vy1[i] = vy1[i] + dy1 * fcoeff;
            vz1[i] = vz1[i] + dz1 * fcoeff;
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        validation = 0.;
        for(i = 0; i < n; i++)
        {
            validation = validation + (vx1[i] + vy1[i] + vz1[i]);
        }

        final = final + validation;


        double t3 = std::chrono::duration<double>(t2 - t1).count();

        elapsed = elapsed + t3;
    }


    auto tm4 = std::chrono::high_resolution_clock::now();
    if(rank == 0)
    {
        printf("\nKernel elapsed time, s: %18.8lf\n", elapsed);
        printf("Total  elapsed time, s: %18.8lf\n", std::chrono::duration<double>(tm4 - tm3).count());
        printf("Result validation: %18.8lf\n", final);
        printf("Result expected  : 6636045675.12190628\n");
    }

    return 0;
}
