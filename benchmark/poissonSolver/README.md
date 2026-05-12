# Poisson Solver Benchmark

This benchmark implements a matrix-free Poisson solver in `alpaka3` using a preconditioned BiCGStab iteration.
The current implementation targets structured Cartesian grids with Dirichlet boundary conditions and supports
compile-time selection of `1`, `2`, `3`, or `4` dimensions.

The implementation is based on the ideas described in:

Luca Pennati, Måns I. Andersson, Klaus Steiniger, Rene Widera, Tapish Narwal, Michael Bussmann, Stefano Markidis,
"A Parallel and Highly-Portable HPC Poisson Solver: Preconditioned Bi-CGSTAB with alpaka",
arXiv:2503.08935, 2025.
https://arxiv.org/abs/2503.08935

## Solver Overview

- Solver: BiCGStab
- Preconditioners: `none`, `jacobi`, `chebyshev`
- Operator: matrix-free N-dimensional Laplacian / Poisson stencil
- Validation: manufactured analytical solution
- Runtime reporting:
  - device information
  - matrix size
  - init time
  - solver time
  - preconditioner time

The compile-time dimension is configured with the CMake cache variable `POISSON_SOLVER_DIMENSIONS`.
The default checked-in value is `2`.

## Build

```bash
cmake -S /workspace -B build-poisson-2d -Dalpaka_BENCHMARKS=ON -DPOISSON_SOLVER_DIMENSIONS=2
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

Dimension-dependent size parameters:

- 1D: `--size-x`
- 2D: `--size-x --size-y`
- 3D: `--size-x --size-y --size-z`
- 4D: `--size-x --size-y --size-z --size-w`

## Examples

### 1D

```bash
cmake -S /workspace -B build-poisson-1d -Dalpaka_BENCHMARKS=ON -DPOISSON_SOLVER_DIMENSIONS=1
cmake --build build-poisson-1d --target poissonSolver -j 4
./build-poisson-1d/benchmark/poissonSolver/poissonSolver \
  --size-x 10001 \
  --max-steps 400 \
  --epsilon 1e-7 \
  --preconditioner jacobi \
  --preconditioner-max-steps 4
```

### 2D

```bash
cmake -S /workspace -B build-poisson-2d -Dalpaka_BENCHMARKS=ON -DPOISSON_SOLVER_DIMENSIONS=2
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
cmake -S /workspace -B build-poisson-3d -Dalpaka_BENCHMARKS=ON -DPOISSON_SOLVER_DIMENSIONS=3
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
cmake -S /workspace -B build-poisson-4d -Dalpaka_BENCHMARKS=ON -DPOISSON_SOLVER_DIMENSIONS=4
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
