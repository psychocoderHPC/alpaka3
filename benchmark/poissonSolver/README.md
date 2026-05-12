# Poisson Solver Benchmark

This benchmark implements a matrix-free Poisson solver in `alpaka3` using a preconditioned BiCGStab iteration.
It targets structured Cartesian grids with homogeneous Dirichlet boundaries and supports compile-time selection
of `1`, `2`, `3`, or `4` dimensions.

The implementation is derived from the solver ideas described in:

Luca Pennati, Måns I. Andersson, Klaus Steiniger, Rene Widera, Tapish Narwal, Michael Bussmann, Stefano Markidis,
"A Parallel and Highly-Portable HPC Poisson Solver: Preconditioned Bi-CGSTAB with alpaka",
arXiv:2503.08935, 2025.
https://arxiv.org/abs/2503.08935

This benchmark is not a paper-equivalent reproduction. The current code runs on a single selected alpaka device,
uses a manufactured analytical solution on the unit box for validation, and builds the right-hand side from
`rhs = A * exact` with the same matrix-free operator used by the solver. It does not model the paper's distributed
block-Jacobi setup or its full benchmark problem.

## Solver Overview

- Solver: BiCGStab
- Preconditioners:
  - `none`: no preconditioning
  - `jacobi-iter` (alias: `jacobi`): repeated diagonal Jacobi relaxation steps applied as a preconditioner
  - `chebyshev`: Chebyshev-style polynomial preconditioner using heuristically widened Dirichlet Laplacian eigenvalue bounds
- Operator: matrix-free N-dimensional Laplacian / Poisson stencil
- Validation: manufactured analytical solution with homogeneous Dirichlet boundaries
- Runtime reporting:
  - device information
  - matrix size
  - init time
  - solver time
  - preconditioner time

The compile-time dimension is configured with the CMake cache variable `alpaka_BENCHMARK_POISSON_SOLVER_DIMS`.
The default checked-in value is `2`.

## Build

```bash
cmake -S /workspace -B build-poisson-2d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=2
cmake --build build-poisson-2d --target poissonSolver -j 4
```

## Usage

The executable is:

```bash
./build-poisson-2d/benchmark/poissonSolver/poissonSolver
```

Common runtime parameters:

- `--max-steps`
- `--epsilon`
- `--preconditioner`
- `--preconditioner-max-steps`

The default `--preconditioner-max-steps` is `24`. This aligns the default Chebyshev iteration count more closely with
the paper-inspired setup, and it also affects `jacobi-iter` because both preconditioners use the same CLI parameter.

Dimension-dependent size parameters:

- 1D: `--size-x`
- 2D: `--size-x --size-y`
- 3D: `--size-x --size-y --size-z`
- 4D: `--size-x --size-y --size-z --size-w`

## Examples

### 1D

```bash
cmake -S /workspace -B build-poisson-1d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=1
cmake --build build-poisson-1d --target poissonSolver -j 4
./build-poisson-1d/benchmark/poissonSolver/poissonSolver \
  --size-x 10001 \
  --max-steps 400 \
  --epsilon 1e-7 \
  --preconditioner chebyshev \
  --preconditioner-max-steps 24
```

### 2D

```bash
cmake -S /workspace -B build-poisson-2d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=2
cmake --build build-poisson-2d --target poissonSolver -j 4
./build-poisson-2d/benchmark/poissonSolver/poissonSolver \
  --size-x 257 \
  --size-y 193 \
  --max-steps 200 \
  --epsilon 1e-8 \
  --preconditioner jacobi \
  --preconditioner-max-steps 4
```

### 3D

```bash
cmake -S /workspace -B build-poisson-3d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=3
cmake --build build-poisson-3d --target poissonSolver -j 4
./build-poisson-3d/benchmark/poissonSolver/poissonSolver \
  --size-x 33 \
  --size-y 25 \
  --size-z 17 \
  --max-steps 200 \
  --epsilon 1e-8 \
  --preconditioner jacobi \
  --preconditioner-max-steps 4
```

### 4D

```bash
cmake -S /workspace -B build-poisson-4d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=4
cmake --build build-poisson-4d --target poissonSolver -j 4
./build-poisson-4d/benchmark/poissonSolver/poissonSolver \
  --size-x 17 \
  --size-y 13 \
  --size-z 11 \
  --size-w 9 \
  --max-steps 200 \
  --epsilon 1e-8 \
  --preconditioner jacobi \
  --preconditioner-max-steps 4
```
