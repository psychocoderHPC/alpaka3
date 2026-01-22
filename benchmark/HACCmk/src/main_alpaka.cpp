#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>

#include <omp.h>

#include <bit>
#include <chrono>
#include <experimental/simd>
#include <iostream>

namespace stdx = std::experimental;

#define NC 16'777'216
#define N 15000 /* Vector length, must be divisible by 4  15000 */
#define ETOL 1.e-4 /* Tolerance for correctness */

struct Kernel
{
    template<typename T_Type, uint32_t T_maxConcurrencyInByte, uint32_t T_cacheLineInByte>
    static constexpr auto calcSimdWidth()
    {
        constexpr uint32_t maxSimdBytes = std::min(T_cacheLineInByte, T_maxConcurrencyInByte);
        return alpaka::divExZero(maxSimdBytes, static_cast<uint32_t>(sizeof(T_Type)));
    }

    constexpr void step10(
        auto const& acc,
        int count1,
        float xxi,
        float yyi,
        float zzi,
        float fsrrmax2,
        float mp_rsm2,
        auto const& xx1,
        auto const& yy1,
        auto const& zz1,
        auto const& mass1,
        float* dxi,
        float* dyi,
        float* dzi) const
    {
        using namespace alpaka;
        float const ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491,
                    ma5 = -0.00000147177;

        constexpr uint32_t maxArchSimdWidth
            = getArchSimdWidth<float>(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});

        constexpr auto maxConcurrecy
            = alpaka::getNumElemPerThread<float>(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){})
              * sizeof(float);

        constexpr uint32_t cachelineBytes
            = getCachelineSize(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){});

        constexpr uint32_t width = std::min(maxArchSimdWidth, calcSimdWidth<float, maxConcurrecy, cachelineBytes>());

        auto force = Vec<Simd<float, width>, 3u>::fill(Simd<float, width>::fill(0.));

        auto simdGrid = onAcc::SimdAlgo{onAcc::WorkerGroup{Vec{0}, Vec{1}}};
        simdGrid.concurrent(
            acc,
            Vec{count1},
            [&](auto const&, auto&& simd_xx1, auto&& simd_yy1, auto&& simd_zz1, auto&& simd_mass1) constexpr
            {
                auto dxc = simd_xx1.load() - xxi;
                auto dyc = simd_yy1.load() - yyi;
                auto dzc = simd_zz1.load() - zzi;

                auto r2 = dxc * dxc + dyc * dyc + dzc * dzc;

                using SimdType = ALPAKA_TYPEOF(simd_mass1.load());
#if 0
                auto m = SimdType::fill(0.);
#else
                SimdType m(0.f);
#endif
                where(r2 < fsrrmax2, m) = simd_mass1.load();
                // stdx::where(r2 < fsrrmax2, m.asBaseType()) = simd_mass1.load().asBaseType();
                auto tmp = r2 + mp_rsm2;
#define FAST_POW 1
#define SQRT_IMPL 3

#if FAST_POW == 1
#    if SQRT_IMPL == 1
                auto bar = SimdType([&](uint32_t const idx) constexpr { return math::sqrt(tmp[idx]); });
                auto p = float{1.0} / (tmp * bar);
#    elif SQRT_IMPL == 2
                using std::sqrt;
                auto bar = sqrt(*reinterpret_cast<stdx::fixed_size_simd<float, SimdType::width()>*>(&tmp));
                auto p = float{1.0} / (tmp * *reinterpret_cast<SimdType*>(&bar));
#    elif SQRT_IMPL == 3
                using std::sqrt;
                auto bar = sqrt(tmp);
                auto p = float{1.0} / (tmp * SimdType(bar));
#    endif

#else
                auto p = SimdType([&](uint32_t const idx) constexpr { return math::pow(tmp[idx], float{-1.5}); });
#endif

                auto f = p - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5)))));
#if 0
                auto fac = SimdType::fill(0.);
#else
                SimdType fac(0.f);
#endif
                where(r2 > 0.0f, fac) = m * f;
                // stdx::where(r2 > 0.0f, fac.asBaseType()) = ( m * f).asBaseType();

                if constexpr(SimdType::width() == 1)
                {
                    force.x()[0] += (fac * dxc)[0];
                    force.y()[0] += (fac * dyc)[0];
                    force.z()[0] += (fac * dzc)[0];
                }
                else
                {
                    force.x() += (fac * dxc);
                    force.y() += (fac * dyc);
                    force.z() += (fac * dzc);
                }
            },
            xx1,
            yy1,
            zz1,
            mass1);

        *dxi = force.x().sum();
        *dyi = force.y().sum();
        *dzi = force.z().sum();
    }

    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        int count,
        int n,
        float fsrrmax2,
        float mp_rsm2,
        float fcoeff,
        float dx1_in,
        float dy1_in,
        float dz1_in,
        alpaka::concepts::IMdSpan auto mass,
        alpaka::concepts::IMdSpan auto xx,
        alpaka::concepts::IMdSpan auto yy,
        alpaka::concepts::IMdSpan auto zz,
        alpaka::concepts::IMdSpan auto vx1,
        alpaka::concepts::IMdSpan auto vy1,
        alpaka::concepts::IMdSpan auto vz1) const
    {
        using namespace alpaka;

        for(auto i :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearThreadsInGrid, alpaka::IdxRange{count}))
        {
            float dx1 = dx1_in;
            float dy1 = dy1_in;
            float dz1 = dz1_in;
            step10(acc, n, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx1, &dy1, &dz1);

            vx1[i] = vx1[i] + dx1 * fcoeff;
            vy1[i] = vy1[i] + dy1 * fcoeff;
            vz1[i] = vz1[i] + dz1 * fcoeff;
        }
    }
};

auto example(auto const deviceSpec, auto const exec, int numElements, size_t numberOfRuns) -> int
{
    using namespace alpaka;

    using IdxVec = Vec<int, 1u>;

    // Define problem size
    IdxVec const extent(numElements);

    // Define the buffer element type
    using Data = float;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << onHost::demangledName<Data>() << std::endl;
    std::cout << "Number of runs: " << numberOfRuns << std::endl;

    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue(queueKind::blocking);

    // Allocate 3 host memory buffers
    auto xx = onHost::allocUnified<Data>(devAcc, extent);
    auto yy = onHost::allocUnified<Data>(devAcc, extent);
    auto zz = onHost::allocUnified<Data>(devAcc, extent);
    auto mass = onHost::allocUnified<Data>(devAcc, extent);
    auto vx1 = onHost::allocUnified<Data>(devAcc, extent);
    auto vy1 = onHost::allocUnified<Data>(devAcc, extent);
    auto vz1 = onHost::allocUnified<Data>(devAcc, extent);

    float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1;

    [[maybe_unused]] char M1[NC], M2[NC];
    int n, count, i, rank;
    double elapsed = 0.0, validation, final;

    rank = 0;
    count = 327;

    auto tm3 = std::chrono::high_resolution_clock::now();

    int chunkSize = 1;
    auto frameSpec = onHost::FrameSpec{divExZero(count, chunkSize), chunkSize};
    std::cout << "FrameSpec: " << frameSpec << std::endl;

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


        auto const haccKernel = KernelBundle{
            Kernel{},
            count,
            n,
            fsrrmax2,
            mp_rsm2,
            fcoeff,
            dx1,
            dy1,
            dz1,
            mass,
            xx,
            yy,
            zz,
            vx1,
            vy1,
            vz1};


        auto t1 = std::chrono::high_resolution_clock::now();

        queue.enqueue(exec, frameSpec, haccKernel);

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

void help(char* argv[])
{
    std::cerr << argv[0] << " [-n  numElements] [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    size_t numElements = N;
    size_t numberOfRuns = 1;

    int opt;
    while((opt = getopt(argc, argv, "hn:r:")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'r':
            try
            {
                numberOfRuns = std::stoul(optarg, nullptr, 0);
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid number of runs '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: number of runs '" << optarg << "' out of range for size_t.\n";
                return EXIT_FAILURE;
            }
            if(numberOfRuns == 0)
            {
                std::cerr << "Error: number of runs must be greater than zero.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;

    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return example(deviceSpec, executor, numElements);
     *  @endcode
     *
     * Some examples for device specifications (depending on the active dependencies).
     *
     *   onHost::DeviceSpec{api::host, deviceKind::cpu}
     *   onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}
     *   onHost::DeviceSpec{api::hip, deviceKind::amdGpu}
     *   onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu}
     *
     * A list of api's and device kinds can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     * A list of executors can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return example(
                backend[alpaka::object::deviceSpec],
                backend[alpaka::object::exec],
                numElements,
                numberOfRuns);
        },
        onHost::allBackends(onHost::enabledApis, exec::enabledExecutors));
}
