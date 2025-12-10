/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_test_macros.hpp>

#if ALPAKA_LANG_SYCL

TEST_CASE("SYCL native handle usage", "")
{
    using namespace alpaka;

    auto devSelector = onHost::makeDeviceSelector(api::oneApi, deviceKind::cpu);

    try
    {
        sycl::platform platform = sycl::platform(sycl::cpu_selector_v);
        std::vector<sycl::device> devices = platform.get_devices();

        sycl::device myDevice;
        for(auto& device : devices)
        {
            sycl::context ctx{device};
            sycl::queue syclQueue{ctx, device, sycl::property::queue::in_order{}};
            auto nativeDeviceHandle = std::make_pair(device, ctx);
            onHost::Device aDevice = devSelector.linkDevice(nativeDeviceHandle, true);
            onHost::Queue aQueue = aDevice.linkQueue(queueKind::blocking, syclQueue, true);

            std::cout << "device name: " << aDevice.getName() << std::endl;
            std::cout << "queue name: " << aQueue.getName() << std::endl;

            int extent = 100;
            auto mem = onHost::allocUnified<int>(aDevice, extent);
            onHost::iota(aQueue, 0, mem);

            // validate iota starting with zero
            int refIotaCounter = 0;
            meta::ndLoopIncIdx(
                Vec{extent},
                [&](auto idx)
                {
                    CHECK(mem[idx] == refIotaCounter);
                    ++refIotaCounter;
                });

            aDevice.unlinkQueue(syclQueue);
            devSelector.unLinkDevice(nativeDeviceHandle);
        }
    }
    catch(sycl::exception const& e)
    {
        throw std::runtime_error(std::string("SYCL exception: ") + e.what());
    }
}

#endif

#if ALPAKA_LANG_CUDA

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            throw std::runtime_error(                                         \
                std::string("CUDA error: ") + cudaGetErrorString(err__) +     \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                     \
    } while (0)

TEST_CASE("CUDA/HIP native handle usage", "")
{
    using namespace alpaka;

    auto devSelector = onHost::makeDeviceSelector(api::cuda, deviceKind::nvidiaGpu);

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if(err != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +  cudaGetErrorString(err));
    }

    if(deviceCount == 0)
    {
        std::cerr << "No CUDA devices found.\n";
    }

    for(int devIdx = 0; devIdx <deviceCount;++devIdx)
    {
        CUDA_CHECK(cudaSetDevice(devIdx));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));


        onHost::Device aDevice = devSelector.linkDevice(devIdx, true);

        onHost::Queue aQueue = aDevice.linkQueue(queueKind::blocking, stream, false);

        std::cout << "device name: " << aDevice.getName() << std::endl;
        std::cout << "queue name: " << aQueue.getName() << std::endl;

        int extent = 100;
        auto mem = onHost::allocUnified<int>(aDevice, extent);
        onHost::iota(aQueue, 0, mem);

        // validate iota starting with zero
        int refIotaCounter = 0;
        meta::ndLoopIncIdx(
            Vec{extent},
            [&](auto idx)
            {
                CHECK(mem[idx] == refIotaCounter);
                ++refIotaCounter;
            });

        aDevice.unlinkQueue(stream);
        devSelector.unLinkDevice(devIdx);

        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}
#endif
