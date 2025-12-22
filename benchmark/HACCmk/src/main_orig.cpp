#include <omp.h>

#include <chrono>
#include <cstdio>

extern void Step10_orig(
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
    float* dzi);

#define NC 16'777'216
#define N 15000 /* Vector length, must be divisible by 4  15000 */
#define ETOL 1.e-4 /* Tolerance for correctness */

int main(int argc, char* argv[])
{
    static float xx[N], yy[N], zz[N], mass[N], vx1[N], vy1[N], vz1[N];
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

    for(n = 400; n < N; n = n + 20)
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
            Step10_orig(n, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx1, &dy1, &dz1);

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
