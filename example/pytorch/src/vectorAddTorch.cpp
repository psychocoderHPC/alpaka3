/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#if ALPAKA_LANG_CUDA
#    include <ATen/cuda/CUDAContext.h>
#endif

#if ALPAKA_LANG_HIP
#    include <ATen/hip/HIPContext.h>
#endif

#include <iostream>
#include <map>
#include <optional>
#include <tuple>

// #include <sys/mman.h>


// Declaration of CUDA function
torch::Tensor vector_add_impl(torch::Tensor& a, torch::Tensor& b, c10::optional<torch::Tensor>);

// Register schema
TORCH_LIBRARY(vectoradd, m)
{
    m.def("add(Tensor a, Tensor b, Tensor(a!)? out=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(vectoradd, CPU, m)
{
    m.impl("add", vector_add_impl);
}

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
TORCH_LIBRARY_IMPL(vectoradd, CUDA, m)
{
    m.impl("add", vector_add_impl);
}
#endif

template<uint32_t T_bytes>
inline constexpr bool isAligned(auto&&... tensors)
{
    return ((reinterpret_cast<uintptr_t>(tensors.data_ptr()) % T_bytes == 0) && ...);
}

template<typename T_Type>
void example(auto aQueue, auto const aExec, torch::Tensor& c, torch::Tensor& a, torch::Tensor& b)
{
    using namespace alpaka;

    if(isAligned<64u>(c, a, b))
    {
        concepts::MdSpan auto aView = makeMdSpan(a.data_ptr<T_Type>(), Vec{a.size(0)}, Alignment<64>{});
        concepts::MdSpan auto bView = makeMdSpan(b.data_ptr<T_Type>(), Vec{b.size(0)}, Alignment<64>{});
        concepts::MdSpan auto cView = makeMdSpan(c.data_ptr<T_Type>(), Vec{c.size(0)}, Alignment<64>{});

        onHost::transform(aQueue, aExec, cView, std::plus{}, aView, bView);
    }
    else
    {
        concepts::MdSpan auto aView = makeMdSpan(a.data_ptr<T_Type>(), Vec{a.size(0)});
        concepts::MdSpan auto bView = makeMdSpan(b.data_ptr<T_Type>(), Vec{b.size(0)});
        concepts::MdSpan auto cView = makeMdSpan(c.data_ptr<T_Type>(), Vec{c.size(0)});

        onHost::transform(aQueue, aExec, cView, std::plus{}, aView, bView);
    }
}

torch::Tensor vector_add_impl(torch::Tensor& a, torch::Tensor& b, c10::optional<torch::Tensor> c)
{
    torch::NoGradGuard no_grad;
    TORCH_CHECK(a.scalar_type() == at::kFloat, "Input a must be a float tensor");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "Input b must be a float tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same device");

    if(!c)
    {
        c = torch::empty_like(a, torch::MemoryFormat::Contiguous);
#if 0
        uint8_t* ptr = c->data_ptr<uint8_t>();
        size_t n = c->numel();
       // madvise(ptr, n*4, MADV_WILLNEED);
        size_t step = 4096 ; /// sizeof(float); // one float per 4 KiB
        for (size_t i = 0; i < n; i += step) {
             ptr[i] ={0}; // touch the page
        }
#endif
    }
    using namespace alpaka;
#if ALPAKA_LANG_CUDA
    if(a.is_cuda())
    {
        int deviceId = at::cuda::current_device();
        auto nativeStream = at::cuda::getCurrentCUDAStream().stream();

        auto devSelector = alpaka::onHost::makeDeviceSelector(api::cuda, deviceKind::nvidiaGpu);
        using DeviceType = ALPAKA_TYPEOF(devSelector.linkDevice(0));

        // keep it alife until the end of the usage in python
        static std::map<int, DeviceType> deviceMap;
        std::optional<DeviceType> computeDevice;

        auto deviceInMap = deviceMap.find(deviceId);
        if(deviceInMap == deviceMap.end())
        {
            computeDevice = devSelector.linkDevice(deviceId);
            deviceMap.emplace(deviceId, *computeDevice);
        }
        else
        {
            auto foo = deviceInMap->second;
            computeDevice = foo;
        }
        auto queue = computeDevice.value().linkQueue(queueKind::nonBlocking, nativeStream, false);

        AT_DISPATCH_ALL_TYPES(
            a.scalar_type(),
            "vector_add_impl",
            [&] { example<scalar_t>(queue, exec::gpuCuda, *c, a, b); });
        computeDevice.value().unlinkQueue(nativeStream);
        return c->contiguous();
    }
#endif

#if ALPAKA_LANG_HIP
    if(a.is_cuda())
    {
        int deviceId = at::hip::current_device();
        auto nativeStream = at::hip::getCurrentHIPStream().stream();

        auto devSelector = alpaka::onHost::makeDeviceSelector(api::hip, deviceKind::amdGpu);
        using DeviceType = ALPAKA_TYPEOF(devSelector.linkDevice(0));

        // keep it alife until the end of the usage in python
        static std::map<int, DeviceType> deviceMap;
        std::optional<DeviceType> computeDevice;

        auto deviceInMap = deviceMap.find(deviceId);
        if(deviceInMap == deviceMap.end())
        {
            computeDevice = devSelector.linkDevice(deviceId);
            deviceMap.emplace(deviceId, *computeDevice);
        }
        else
        {
            auto foo = deviceInMap->second;
            computeDevice = foo;
        }
        auto queue = computeDevice.value().linkQueue(queueKind::nonBlocking, nativeStream, false);

        AT_DISPATCH_ALL_TYPES(
            a.scalar_type(),
            "vector_add_impl",
            [&] { example<scalar_t>(queue, exec::gpuHip, *c, a, b); });
        computeDevice.value().unlinkQueue(nativeStream);
        return c->contiguous();
    }
#endif
    if(a.is_cpu())
    {
        static auto computeDevice = onHost::makeHostDevice();
        static auto queue = computeDevice.makeQueue(queueKind::blocking);

        auto executor = supportedMappings(queue.getDevice(), exec::allExecutors);
        AT_DISPATCH_ALL_TYPES(
            a.scalar_type(),
            "vector_add_impl",
            [&]
            {
                example<scalar_t>(queue, std::get<0>(executor), *c, a, b);
                // std::cout << "aptr: " << c->data_ptr<scalar_t>() << std::endl;
            });
        onHost::wait(queue);

        return c->contiguous();
    }
    else
    {
        TORCH_CHECK(false, "Unsupported device: ", a.device().type());
    }

    return c->contiguous();
}
