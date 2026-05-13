/* Copyright 2026 OpenAI
 * SPDX-License-Identifier: MPL-2.0
 */

#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <tuple>

namespace poisson
{
    struct SolverStats
    {
        uint32_t iterations = 0u;
        Real initialResidual = 0.0;
        Real finalResidual = 0.0;
        double solverSeconds = 0.0;
        double preconditionerSeconds = 0.0;
    };

    enum class BoundaryCondition
    {
        dirichlet,
        neumann
    };

    ALPAKA_FN_HOST_ACC auto isDirichletBoundaryCell(auto const& idx, Extent const extent) -> bool;

    struct Multiply
    {
        ALPAKA_FN_ACC constexpr auto operator()(auto const& lhs, auto const& rhs) const
        {
            return lhs * rhs;
        }
    };

    struct UpdateP
    {
        Real beta = 0.0;
        Real omega = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(auto const& r, auto const& p, auto const& v) const
        {
            return r + beta * (p - omega * v);
        }
    };

    struct CombineAlpha
    {
        Real alpha = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(auto const& lhs, auto const& rhs) const
        {
            return lhs - alpha * rhs;
        }
    };

    struct UpdateResidual
    {
        Real omega = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(auto const& s, auto const& t) const
        {
            return s - omega * t;
        }
    };

    struct UpdateSolution
    {
        Real alpha = 0.0;
        Real omega = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(auto const& x, auto const& y, auto const& z) const
        {
            return x + alpha * y + omega * z;
        }
    };

    struct JacobiRelax
    {
        template<typename TAcc>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto const& z,
            alpaka::concepts::IMdSpan auto const& rhs,
            alpaka::concepts::IMdSpan auto const& az,
            Extent const extent,
            RealVec const invH2) const -> void
        {
            using namespace alpaka;

            Real diagonal = 0.0;
            for(uint32_t dim = 0u; dim < dimensions; ++dim)
                diagonal += 2.0 * invH2[dim];

            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extent}))
            {
                Real const pointDiagonal = isDirichletBoundaryCell(idx, extent) ? 1.0 : diagonal;
                out[idx] = z[idx] + (rhs[idx] - az[idx]) / pointDiagonal;
            }
        }
    };

    struct Scale
    {
        Real factor = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(auto const& value) const
        {
            return factor * value;
        }
    };

    struct ChebyshevStep
    {
        Real alpha = 0.0;
        Real beta = 0.0;

        ALPAKA_FN_ACC constexpr auto operator()(auto const& residual, auto const& direction) const
        {
            return alpha * residual + beta * direction;
        }
    };

    ALPAKA_FN_HOST_ACC auto unitDirection(uint32_t const dim) -> Extent
    {
        auto direction = Extent::fill(0u);
        direction[dim] = 1u;
        return direction;
    }

    ALPAKA_FN_HOST_ACC auto boundaryCondition(uint32_t const userDim, bool const lowerSide) -> BoundaryCondition
    {
        if(userDim == 0u)
            return lowerSide ? BoundaryCondition::dirichlet : BoundaryCondition::neumann;
        return lowerSide ? BoundaryCondition::neumann : BoundaryCondition::dirichlet;
    }

    ALPAKA_FN_HOST_ACC auto isDirichletBoundaryCell(auto const& idx, Extent const extent) -> bool
    {
        for(uint32_t userDim = 0u; userDim < dimensions; ++userDim)
        {
            auto const dim = alpakaDimFromUserDim(userDim);
            if(idx[dim] == 0u && boundaryCondition(userDim, true) == BoundaryCondition::dirichlet)
                return true;
            if(idx[dim] + 1u == extent[dim] && boundaryCondition(userDim, false) == BoundaryCondition::dirichlet)
                return true;
        }
        return false;
    }

    ALPAKA_FN_HOST_ACC auto paperLowerBounds() -> RealVec
    {
        auto lowerBounds = RealVec::fill(0.0);
        if constexpr(dimensions >= 1u)
            lowerBounds[alpakaDimFromUserDim(0u)] = 3.0;
        if constexpr(dimensions >= 2u)
            lowerBounds[alpakaDimFromUserDim(1u)] = 2.5;
        if constexpr(dimensions >= 3u)
            lowerBounds[alpakaDimFromUserDim(2u)] = 10.0;
        if constexpr(dimensions >= 4u)
            lowerBounds[alpakaDimFromUserDim(3u)] = 1.5;
        return lowerBounds;
    }

    ALPAKA_FN_HOST_ACC auto makePosition(auto const& idx, RealVec const spacing) -> RealVec
    {
        auto position = paperLowerBounds();
        for(uint32_t dim = 0u; dim < dimensions; ++dim)
            position[dim] += static_cast<Real>(idx[dim]) * spacing[dim];
        return position;
    }

    ALPAKA_FN_HOST_ACC auto exactSolution(RealVec const& position) -> Real
    {
        Real const x = position[alpakaDimFromUserDim(0u)];
        Real value = 10.0 + alpaka::math::sin(x);
        if constexpr(dimensions >= 2u)
        {
            Real const y = position[alpakaDimFromUserDim(1u)];
            value += alpaka::math::cos(y) - y * y;

            Real productTerm = x * x;
            for(uint32_t userDim = 1u; userDim < dimensions; ++userDim)
                productTerm *= position[alpakaDimFromUserDim(userDim)];
            value += productTerm;
        }
        if constexpr(dimensions >= 3u)
        {
            Real const z = position[alpakaDimFromUserDim(2u)];
            value += 3.0 * alpaka::math::sin(z);
        }
        if constexpr(dimensions >= 4u)
        {
            Real const w = position[alpakaDimFromUserDim(3u)];
            value += 4.0 * alpaka::math::cos(w);
        }
        return value;
    }

    struct ApplyOperatorKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto const& in,
            Extent const extent,
            RealVec const invH2) const -> void
        {
            using namespace alpaka;

            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{extent}))
            {
                if(isDirichletBoundaryCell(idx, extent))
                {
                    out[idx] = in[idx];
                    continue;
                }

                Real value = 0.0;
                for(uint32_t dim = 0u; dim < dimensions; ++dim)
                {
                    auto const direction = unitDirection(dim);
                    if(idx[dim] == 0u)
                    {
                        value += invH2[dim] * (2.0 * in[idx] - 2.0 * in[idx + direction]);
                    }
                    else if(idx[dim] + 1u == extent[dim])
                    {
                        value += invH2[dim] * (2.0 * in[idx] - 2.0 * in[idx - direction]);
                    }
                    else
                    {
                        value += invH2[dim] * (2.0 * in[idx] - in[idx - direction] - in[idx + direction]);
                    }
                }
                out[idx] = value;
            }
        }
    };

    auto printDeviceInfo(auto const& deviceSpec, auto const& exec, auto const& device) -> void
    {
        auto const properties = device.getDeviceProperties();
        std::cout << "api: " << deviceSpec.getApi().getName() << '\n';
        std::cout << "device kind: " << deviceSpec.getDeviceKind().getName() << '\n';
        std::cout << "device name: " << properties.getName() << '\n';
        std::cout << "executor: " << exec.getName() << '\n';
        std::cout << "multiprocessors: " << properties.multiProcessorCount << '\n';
        std::cout << "warp size: " << properties.warpSize << '\n';
        std::cout << "max threads per block: " << properties.maxThreadsPerBlock << '\n';
        std::cout << "global memory bytes: " << properties.globalMemCapacityBytes << '\n';
    }

    auto extentProduct(Extent const extent) -> size_t
    {
        return static_cast<size_t>(extent.product());
    }

    auto formatExtent(Extent const extent) -> std::string
    {
        std::ostringstream stream;
        for(uint32_t userDim = 0u; userDim < dimensions; ++userDim)
        {
            if(userDim != 0u)
                stream << " x ";
            stream << extent[alpakaDimFromUserDim(userDim)];
        }
        return stream.str();
    }

    auto isJacobiIterationEnabled(std::string_view const preconditioner) -> bool
    {
        return preconditioner == "jacobi-iter";
    }

    auto isChebyshevEnabled(std::string_view const preconditioner) -> bool
    {
        return preconditioner == "chebyshev";
    }

    auto makeExtent(std::array<IdxType, dimensions> const& sizes) -> Extent
    {
        auto extent = Extent::fill(0u);
        for(uint32_t userDim = 0u; userDim < dimensions; ++userDim)
            extent[alpakaDimFromUserDim(userDim)] = sizes[userDim];
        return extent;
    }

    auto makeSpacing(std::array<IdxType, dimensions> const& sizes) -> RealVec
    {
        auto spacing = RealVec::fill(0.0);
        for(uint32_t userDim = 0u; userDim < dimensions; ++userDim)
            spacing[alpakaDimFromUserDim(userDim)] = 0.1;
        return spacing;
    }

    auto makeInvSpacingSquared(RealVec const spacing) -> RealVec
    {
        auto invH2 = RealVec::fill(0.0);
        for(uint32_t dim = 0u; dim < dimensions; ++dim)
            invH2[dim] = 1.0 / (spacing[dim] * spacing[dim]);
        return invH2;
    }

    auto preferredFrameExtent() -> Extent
    {
        if constexpr(dimensions == 1u)
            return Extent{256u};
        else if constexpr(dimensions == 2u)
            return Extent{16u, 16u};
        else if constexpr(dimensions == 3u)
            return Extent{8u, 8u, 4u};
        else
            return Extent{4u, 4u, 4u, 4u};
    }

    auto makeFrameExtent(Extent const extent) -> Extent
    {
        auto frameExtent = preferredFrameExtent();
        for(uint32_t dim = 0u; dim < dimensions; ++dim)
            frameExtent[dim] = std::min(frameExtent[dim], extent[dim]);
        return frameExtent;
    }

    auto applyOperator(auto const& queue, auto const exec, auto& out, auto const& in, Extent const extent, RealVec const invH2)
        -> void
    {
        using namespace alpaka;

        auto const frameExtent = makeFrameExtent(extent);
        queue.enqueue(
            exec,
            onHost::FrameSpec{divCeil(extent, frameExtent), frameExtent},
            KernelBundle{ApplyOperatorKernel{}, out, in, extent, invH2});
    }

    auto dotProduct(auto const& queue, auto const exec, auto& scalarBuffer, auto const& lhs, auto const& rhs) -> Real
    {
        using namespace alpaka;

        onHost::transformReduce(queue, exec, Real{0.0}, scalarBuffer, std::plus{}, Multiply{}, lhs, rhs);
        auto scalarHost = onHost::allocHostLike(scalarBuffer);
        onHost::memcpy(queue, scalarHost, scalarBuffer);
        onHost::wait(queue);
        return scalarHost[0u];
    }

    auto l2Norm(auto const& queue, auto const exec, auto& scalarBuffer, auto const& input) -> Real
    {
        return std::sqrt(std::max<Real>(dotProduct(queue, exec, scalarBuffer, input, input), 0.0));
    }

    auto maxAbsError(auto const& queue, auto const& solution, auto const& reference) -> Real
    {
        using namespace alpaka;

        auto solutionHost = onHost::allocHostLike(solution);
        auto referenceHost = onHost::allocHostLike(reference);
        onHost::memcpy(queue, solutionHost, solution);
        onHost::memcpy(queue, referenceHost, reference);
        onHost::wait(queue);

        Real maxError = 0.0;
        auto const extent = solutionHost.getExtents();
        meta::ndLoopIncIdx(
            extent,
            [&](auto const& idx)
            {
                maxError = std::max(maxError, std::abs(solutionHost[idx] - referenceHost[idx]));
            });
        return maxError;
    }

    auto initializeExactSolution(auto& hostBuffer, RealVec const spacing) -> void
    {
        auto const extent = hostBuffer.getExtents();
        alpaka::meta::ndLoopIncIdx(
            extent,
            [&](auto const& idx)
            {
                hostBuffer[idx] = exactSolution(makePosition(idx, spacing));
            });
    }

    auto mixedBoundaryEigenvalueBounds(Extent const extent, RealVec const invH2) -> std::pair<Real, Real>
    {
        auto eigenvalue1d = [](IdxType const interiorPoints, Real const invSpacingSquared, IdxType const mode) -> Real
        {
            Real const angle
                = static_cast<Real>(mode) * alpaka::math::constants::pi / static_cast<Real>(interiorPoints + 1u);
            return 2.0 * (1.0 - std::cos(angle)) * invSpacingSquared;
        };

        Real lambdaMin = std::numeric_limits<Real>::max();
        Real lambdaMax = 1.0;
        for(uint32_t dim = 0u; dim < dimensions; ++dim)
        {
            IdxType const points = extent[dim];
            lambdaMin = std::min(lambdaMin, eigenvalue1d(points, invH2[dim], 1u));
            lambdaMax += 4.0 * invH2[dim];
        }
        return {std::max(lambdaMin, Real{1.0e-6}), lambdaMax};
    }

    auto chebyshevEigenvalueBounds(Extent const extent, RealVec const invH2) -> std::pair<Real, Real>
    {
        auto [lambdaMin, lambdaMax] = mixedBoundaryEigenvalueBounds(extent, invH2);
        lambdaMin *= 1.0 - 1.0e-4;
        lambdaMax *= 100.0;
        return {lambdaMin, lambdaMax};
    }

    auto applyJacobiIterationPreconditioner(
        auto const& queue,
        auto const exec,
        auto& z,
        auto& workspace,
        auto const& rhs,
        Extent const extent,
        RealVec const invH2,
        uint32_t const steps,
        double& preconditionerSeconds) -> void
    {
        using namespace alpaka;

        auto const start = std::chrono::steady_clock::now();
        onHost::fill(queue, z, Real{0.0});
        auto const frameExtent = makeFrameExtent(extent);
        for(uint32_t iteration = 0u; iteration < steps; ++iteration)
        {
            applyOperator(queue, exec, workspace, z, extent, invH2);
            queue.enqueue(
                exec,
                onHost::FrameSpec{divCeil(extent, frameExtent), frameExtent},
                KernelBundle{JacobiRelax{}, z, z, rhs, workspace, extent, invH2});
        }
        onHost::wait(queue);
        auto const end = std::chrono::steady_clock::now();
        preconditionerSeconds += std::chrono::duration<double>(end - start).count();
    }

    auto applyChebyshevPreconditioner(
        auto const& queue,
        auto const exec,
        auto& z,
        auto& residual,
        auto& direction,
        auto const& rhs,
        Extent const extent,
        RealVec const invH2,
        uint32_t const steps,
        double& preconditionerSeconds) -> void
    {
        using namespace alpaka;

        auto const [lambdaMin, lambdaMax] = chebyshevEigenvalueBounds(extent, invH2);
        Real const d = 0.5 * (lambdaMax + lambdaMin);
        Real const c = 0.5 * (lambdaMax - lambdaMin);

        auto const start = std::chrono::steady_clock::now();
        onHost::fill(queue, z, Real{0.0});
        onHost::transform(queue, exec, direction, Scale{1.0 / d}, rhs);
        onHost::transform(queue, exec, z, std::plus{}, z, direction);

        Real alpha = 1.0 / d;
        for(uint32_t iteration = 1u; iteration < steps; ++iteration)
        {
            applyOperator(queue, exec, residual, z, extent, invH2);
            onHost::transform(queue, exec, residual, std::minus{}, rhs, residual);
            Real const beta = std::pow(0.5 * c * alpha, 2);
            alpha = 1.0 / (d - beta);
            onHost::transform(queue, exec, direction, ChebyshevStep{alpha, beta}, residual, direction);
            onHost::transform(queue, exec, z, std::plus{}, z, direction);
        }
        onHost::wait(queue);
        auto const end = std::chrono::steady_clock::now();
        preconditionerSeconds += std::chrono::duration<double>(end - start).count();
    }

    auto solve(auto const& queue, auto const exec, auto& solution, auto const& rhs, Extent const extent, Options const& options, RealVec const invH2)
        -> SolverStats
    {
        using namespace alpaka;

        auto r = onHost::allocLikeDeferred(queue, solution);
        auto rHat = onHost::allocLikeDeferred(queue, solution);
        auto p = onHost::allocLikeDeferred(queue, solution);
        auto v = onHost::allocLikeDeferred(queue, solution);
        auto s = onHost::allocLikeDeferred(queue, solution);
        auto t = onHost::allocLikeDeferred(queue, solution);
        auto y = onHost::allocLikeDeferred(queue, solution);
        auto z = onHost::allocLikeDeferred(queue, solution);
        auto tmp = onHost::allocLikeDeferred(queue, solution);
        auto preconditionerDirection = onHost::allocLikeDeferred(queue, solution);
        auto scalar = onHost::allocDeferred<Real>(queue, 1u);

        onHost::fill(queue, solution, Real{0.0});
        onHost::fill(queue, r, Real{0.0});
        onHost::fill(queue, p, Real{0.0});
        onHost::fill(queue, v, Real{0.0});

        applyOperator(queue, exec, tmp, solution, extent, invH2);
        onHost::transform(queue, exec, r, std::minus{}, rhs, tmp);
        onHost::memcpy(queue, rHat, r);
        onHost::wait(queue);

        SolverStats stats;
        Real const rhsNorm = std::max(l2Norm(queue, exec, scalar, rhs), std::numeric_limits<Real>::min());
        stats.initialResidual = l2Norm(queue, exec, scalar, r) / rhsNorm;
        stats.finalResidual = stats.initialResidual;
        if(stats.finalResidual <= options.epsilon)
            return stats;

        Real rhoPrevious = 1.0;
        Real alpha = 1.0;
        Real omega = 1.0;

        auto const solveStart = std::chrono::steady_clock::now();
        for(uint32_t iteration = 0u; iteration < options.maxSteps; ++iteration)
        {
            Real const rho = dotProduct(queue, exec, scalar, rHat, r);
            if(std::abs(rho) <= std::numeric_limits<Real>::epsilon())
                break;

            if(iteration != 0u)
            {
                Real const beta = (rho / rhoPrevious) * (alpha / omega);
                onHost::transform(queue, exec, p, UpdateP{beta, omega}, r, p, v);
            }
            else
            {
                onHost::memcpy(queue, p, r);
            }

            if(isJacobiIterationEnabled(options.preconditioner))
                applyJacobiIterationPreconditioner(
                    queue,
                    exec,
                    y,
                    tmp,
                    p,
                    extent,
                    invH2,
                    options.preconditionerMaxSteps,
                    stats.preconditionerSeconds);
            else if(isChebyshevEnabled(options.preconditioner))
                applyChebyshevPreconditioner(
                    queue,
                    exec,
                    y,
                    tmp,
                    preconditionerDirection,
                    p,
                    extent,
                    invH2,
                    options.preconditionerMaxSteps,
                    stats.preconditionerSeconds);
            else
                onHost::memcpy(queue, y, p);

            applyOperator(queue, exec, v, y, extent, invH2);
            Real const denominator = dotProduct(queue, exec, scalar, rHat, v);
            if(std::abs(denominator) <= std::numeric_limits<Real>::epsilon())
                break;

            alpha = rho / denominator;
            onHost::transform(queue, exec, s, CombineAlpha{alpha}, r, v);

            Real const sNorm = l2Norm(queue, exec, scalar, s) / rhsNorm;
            if(sNorm <= options.epsilon)
            {
                onHost::transform(queue, exec, solution, UpdateSolution{alpha, 0.0}, solution, y, y);
                stats.iterations = iteration + 1u;
                stats.finalResidual = sNorm;
                break;
            }

            if(isJacobiIterationEnabled(options.preconditioner))
                applyJacobiIterationPreconditioner(
                    queue,
                    exec,
                    z,
                    tmp,
                    s,
                    extent,
                    invH2,
                    options.preconditionerMaxSteps,
                    stats.preconditionerSeconds);
            else if(isChebyshevEnabled(options.preconditioner))
                applyChebyshevPreconditioner(
                    queue,
                    exec,
                    z,
                    tmp,
                    preconditionerDirection,
                    s,
                    extent,
                    invH2,
                    options.preconditionerMaxSteps,
                    stats.preconditionerSeconds);
            else
                onHost::memcpy(queue, z, s);

            applyOperator(queue, exec, t, z, extent, invH2);
            Real const tt = dotProduct(queue, exec, scalar, t, t);
            if(std::abs(tt) <= std::numeric_limits<Real>::epsilon())
                break;

            omega = dotProduct(queue, exec, scalar, t, s) / tt;
            onHost::transform(queue, exec, solution, UpdateSolution{alpha, omega}, solution, y, z);
            onHost::transform(queue, exec, r, UpdateResidual{omega}, s, t);

            stats.finalResidual = l2Norm(queue, exec, scalar, r) / rhsNorm;
            stats.iterations = iteration + 1u;
            if(stats.finalResidual <= options.epsilon || std::abs(omega) <= std::numeric_limits<Real>::epsilon())
                break;

            rhoPrevious = rho;
        }
        onHost::wait(queue);
        auto const solveEnd = std::chrono::steady_clock::now();
        stats.solverSeconds = std::chrono::duration<double>(solveEnd - solveStart).count();
        applyOperator(queue, exec, tmp, solution, extent, invH2);
        onHost::fill(queue, r, Real{0.0});
        onHost::transform(queue, exec, r, std::minus{}, rhs, tmp);
        stats.finalResidual = l2Norm(queue, exec, scalar, r) / rhsNorm;
        return stats;
    }

    auto runBackend(auto const& deviceSpec, auto const exec, Options const& options) -> int
    {
        using namespace alpaka;

        auto selector = onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable())
        {
            std::cout << "Skip backend without available device: " << deviceSpec.getName() << '\n';
            return EXIT_SUCCESS;
        }

        onHost::Device device = selector.makeDevice(0u);
#if ALPAKA_LANG_ONEAPI
        if(deviceSpec.getApi() == api::oneApi)
        {
            if(device.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
            {
                std::cout << "Skip oneAPI device without FP64 support: " << device.getName() << '\n';
                return EXIT_SUCCESS;
            }
        }
#endif

        onHost::Queue queue = device.makeQueue(queueKind::blocking);
        auto const extent = makeExtent(options.sizes);
        auto const spacing = makeSpacing(options.sizes);
        auto const invH2 = makeInvSpacingSquared(spacing);

        auto const initStart = std::chrono::steady_clock::now();
        auto exactHost = onHost::allocHost<Real>(extent);
        auto exact = onHost::allocLike(device, exactHost);
        auto rhs = onHost::allocLike(device, exactHost);
        auto solution = onHost::allocLike(device, exactHost);

        initializeExactSolution(exactHost, spacing);
        onHost::memcpy(queue, exact, exactHost);
        applyOperator(queue, exec, rhs, exact, extent, invH2);
        onHost::wait(queue);
        auto const initEnd = std::chrono::steady_clock::now();
        double const initSeconds = std::chrono::duration<double>(initEnd - initStart).count();

        auto const stats = solve(queue, exec, solution, rhs, extent, options, invH2);
        Real const error = maxAbsError(queue, solution, exact);
        Real const validationTolerance = std::max<Real>(1e-4, options.epsilon * 1.0e4);

        std::cout << "==============================\n";
        printDeviceInfo(deviceSpec, exec, device);
        std::cout << "dimensions: " << dimensions << '\n';
        std::cout << "matrix size: " << formatExtent(extent) << " (" << extentProduct(extent) << " cells)\n";
        std::cout << "preconditioner: " << options.preconditioner << '\n';
        std::cout << "max steps: " << options.maxSteps << '\n';
        std::cout << "preconditioner max steps: " << options.preconditionerMaxSteps << '\n';
        std::cout << "epsilon: " << options.epsilon << '\n';
        std::cout << "init time [s]: " << initSeconds << '\n';
        std::cout << "solver time [s]: " << stats.solverSeconds << '\n';
        std::cout << "preconditioner time [s]: " << stats.preconditionerSeconds << '\n';
        std::cout << "iterations: " << stats.iterations << '\n';
        std::cout << "initial residual: " << stats.initialResidual << '\n';
        std::cout << "final residual: " << stats.finalResidual << '\n';
        std::cout << "max abs error: " << error << '\n';

        if(!(stats.finalResidual <= options.epsilon))
        {
            std::cerr << "Solver did not converge to the requested epsilon.\n";
            return EXIT_FAILURE;
        }
        if(!(error <= validationTolerance))
        {
            std::cerr << "Validation failed: error exceeds tolerance " << validationTolerance << '\n';
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace poisson

auto main(int argc, char* argv[]) -> int
{
    poisson::Options options;
    if(int const ret = poisson::parseCmd(argc, argv, options))
        return ret;

    int result = EXIT_SUCCESS;
    auto const backends
        = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);
    std::apply(
        [&](auto const&... backend)
        {
            (
                [&, cfg = backend]()
                {
                    if(poisson::runBackend(cfg[alpaka::object::deviceSpec], cfg[alpaka::object::exec], options)
                       != EXIT_SUCCESS)
                    {
                        result = EXIT_FAILURE;
                    }
                }(),
                ...);
        },
        backends);

    return result;
}
