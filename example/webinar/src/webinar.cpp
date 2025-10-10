/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cstdio>
#include <functional>

// execute the code for the device kind with the given executor
auto exampleUnified(auto const deviceSpec, auto const exec) -> int
{
    unsigned n = 5;

    // Select a device
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device devAcc = devSelector.makeDevice(0);
    printf(
        "Using alpaka accelerator: %s for %s\n",
        alpaka::onHost::demangledName(exec).c_str(),
        devAcc.getName().c_str());

    // Create a blocking queue on the device (no synchronization after kernel or algorithm calls required)
    alpaka::onHost::Queue queue = devAcc.makeQueue(alpaka::queueKind::blocking);

    // allocate unified memory that is accessible on host and device
    auto a = alpaka::onHost::allocUnified<int>(devAcc, n);
    auto b = alpaka::onHost::allocUnified<int>(devAcc, n);
    auto c = alpaka::onHost::allocUnified<int>(devAcc, n);

    // Initialize values on host
    for(unsigned i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = 1;
    }

    // Run element-wise multiplication on device with the given executor
    alpaka::onHost::transform(queue, exec, c, std::multiplies{}, a, b);

    for(unsigned i = 0; i < n; i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}

auto exampleBuffers(auto const deviceSpec, auto const exec) -> int
{
    unsigned n = 5;

    // Select a device
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device devAcc = devSelector.makeDevice(0);
    printf(
        "Using alpaka accelerator: %s for %s\n",
        alpaka::onHost::demangledName(exec).c_str(),
        devAcc.getName().c_str());

    // Create a blocking queue on the device (no synchronization after kernel or algorithm calls required)
    alpaka::onHost::Queue queue = devAcc.makeQueue(alpaka::queueKind::blocking);

    // allocate memory that is accessible on host
    auto h_a = alpaka::onHost::allocHost<int>(n);
    auto h_b = alpaka::onHost::allocHostLike(h_a);
    auto h_c = alpaka::onHost::allocHostLike(h_a);

    // allocate memory on the device and inherit the extents from a
    auto a = alpaka::onHost::allocLike(devAcc, h_a);
    auto b = alpaka::onHost::allocLike(devAcc, h_a);
    auto c = alpaka::onHost::allocLike(devAcc, h_a);

    // Initialize values on host
    for(unsigned i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = 1;
    }

    // copy host memory element wise to the device memory instance
    alpaka::onHost::memcpy(queue, a, h_a);
    alpaka::onHost::memcpy(queue, b, h_b);

    // Run element-wise multiplication on device with the given executor
    alpaka::onHost::transform(queue, exec, c, std::multiplies{}, a, b);

    // copy the device result back to host memory
    alpaka::onHost::memcpy(queue, h_c, c);

    for(unsigned i = 0; i < n; i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}

struct IdxAsignKernel
{
    constexpr void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IMdSpan auto a,
        unsigned region,
        unsigned n) const
    {
        unsigned nPerRegion = a.getExtents() / n;
        unsigned regionOffset = nPerRegion * region;
        for(auto [idx] : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::threadsInGrid,
                alpaka::IdxRange{regionOffset, regionOffset + nPerRegion}))
        {
            a[idx] = idx;
        }
    }
};

// execute the code for the device kind with the given executor
auto exampleKernelUnifiedAsync(auto const deviceSpec, auto const exec) -> int
{
    unsigned n = 5;
    unsigned nx = 20;

    // Select a device
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device devAcc = devSelector.makeDevice(0);
    printf(
        "Using alpaka accelerator: %s for %s\n",
        alpaka::onHost::demangledName(exec).c_str(),
        devAcc.getName().c_str());

    // Create a non-blocking queue on the device
    using QueueType = alpaka::onHost::Queue<ALPAKA_TYPEOF(devAcc), alpaka::queueKind::NonBlocking>;
    std::vector<QueueType> queues;
    for(unsigned region = 0; region < n; region++)
    {
        queues.emplace_back(devAcc.makeQueue(alpaka::queueKind::nonBlocking));
    }

    // allocate unified memory that is accessible on host and device
    auto a = alpaka::onHost::allocUnified<int>(devAcc, nx);

    unsigned frameExtent = 32u;
    auto frameSpec = alpaka::onHost::FrameSpec{alpaka::divExZero(nx, frameExtent), frameExtent};

    // Run element-wise multiplication on device with the given executor
    for(unsigned region = 0; region < n; region++)
    {
        queues[region].enqueue(exec, frameSpec, IdxAsignKernel{}, a, region, n);
    }
    // we do not need to wait for all queues, we can also wait for the full device which implicits all queues
    alpaka::onHost::wait(devAcc);

    for(unsigned i = 0; i < nx; i++)
        printf("a[%d] = %d\n", i, a[i]);

    return 0;
}

struct MulKernel
{
    constexpr void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IMdSpan auto c,
        alpaka::concepts::IMdSpan auto const a,
        alpaka::concepts::IMdSpan auto const b) const
    {
        for(auto idx :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{c.getExtents()}))
        {
            c[idx] = a[idx] * b[idx];
        }
    }
};

// execute the code for the device kind with the given executor
auto exampleKernelUnified(auto const deviceSpec, auto const exec) -> int
{
    unsigned n = 5;

    // Select a device
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device devAcc = devSelector.makeDevice(0);
    printf(
        "Using alpaka accelerator: %s for %s\n",
        alpaka::onHost::demangledName(exec).c_str(),
        devAcc.getName().c_str());

    // Create a blocking queue on the device (no synchronization after kernel or algorithm calls required)
    alpaka::onHost::Queue queue = devAcc.makeQueue(alpaka::queueKind::blocking);

    // allocate unified memory that is accessible on host and device
    auto a = alpaka::onHost::allocUnified<int>(devAcc, n);
    auto b = alpaka::onHost::allocUnified<int>(devAcc, n);
    auto c = alpaka::onHost::allocUnified<int>(devAcc, n);

    // Initialize values on host
    for(unsigned i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = 1;
    }

    unsigned frameExtent = 32u;
    auto frameSpec = alpaka::onHost::FrameSpec{alpaka::divExZero(n, frameExtent), frameExtent};

    // Run element-wise multiplication on device with the given executor
    queue.enqueue(exec, frameSpec, MulKernel{}, c, a, b);

    for(unsigned i = 0; i < n; i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}

auto exampleKernelBuffers(auto const deviceSpec, auto const exec) -> int
{
    unsigned n = 5;

    // Select a device
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device devAcc = devSelector.makeDevice(0);
    printf(
        "Using alpaka accelerator: %s for %s\n",
        alpaka::onHost::demangledName(exec).c_str(),
        devAcc.getName().c_str());

    // Create a blocking queue on the device (no synchronization after kernel or algorithm calls required)
    alpaka::onHost::Queue queue = devAcc.makeQueue(alpaka::queueKind::blocking);

    // allocate memory that is accessible on host
    auto h_a = alpaka::onHost::allocHost<int>(n);
    auto h_b = alpaka::onHost::allocHostLike(h_a);
    auto h_c = alpaka::onHost::allocHostLike(h_a);

    // allocate memory on the device and inherit the extents from a
    auto a = alpaka::onHost::allocLike(devAcc, h_a);
    auto b = alpaka::onHost::allocLike(devAcc, h_a);
    auto c = alpaka::onHost::allocLike(devAcc, h_a);

    // Initialize values on host
    for(unsigned i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = 1;
    }

    // copy host memory element wise to the device memory instance
    alpaka::onHost::memcpy(queue, a, h_a);
    alpaka::onHost::memcpy(queue, b, h_b);

    unsigned frameExtent = 32u;
    auto frameSpec = alpaka::onHost::FrameSpec{alpaka::divExZero(n, frameExtent), frameExtent};

    // Run element-wise multiplication on device with the given executor
    queue.enqueue(exec, frameSpec, MulKernel{}, c, a, b);

    // copy the device result back to host memory
    alpaka::onHost::memcpy(queue, h_c, c);

    for(unsigned i = 0; i < n; i++)
    {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}

auto main() -> int
{
    using namespace alpaka;

    /* Execute the example once for each available backend (device specification + executor)
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
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html##available-apis
     * A list of executors can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return exampleUnified(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec])
                   | exampleBuffers(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec])
                   | exampleKernelUnified(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec])
                   | exampleKernelBuffers(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec])
                   | exampleKernelUnifiedAsync(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]);
        },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
