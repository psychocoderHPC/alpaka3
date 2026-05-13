# Poisson Solver Benchmark

This benchmark implements a matrix-free Poisson solver in `alpaka3` using a preconditioned BiCGStab iteration.
It targets structured Cartesian grids with compile-time selection of `1`, `2`, `3`, or `4` dimensions.

The implementation is derived from the solver ideas described in:

Luca Pennati, Måns I. Andersson, Klaus Steiniger, Rene Widera, Tapish Narwal, Michael Bussmann, Stefano Markidis,
"A Parallel and Highly-Portable HPC Poisson Solver: Preconditioned Bi-CGSTAB with alpaka",
arXiv:2503.08935, 2025.
https://arxiv.org/abs/2503.08935

This version is paper-aligned for the single-device case, but deliberately excludes MPI and block-Jacobi domain
decomposition. It keeps the BiCGStab plus CI-based preconditioning path from the paper, uses the paper's mixed
Dirichlet/Neumann boundary layout in the first three dimensions, keeps the paper's `0.1` mesh spacing, and extends
the benchmark problem consistently to `1D`, `2D`, and `4D`.

## Solver Overview

- Solver: BiCGStab
- Preconditioners:
  - `none`: no preconditioning
  - `jacobi-iter` (alias: `jacobi`): repeated diagonal Jacobi relaxation steps applied as a preconditioner
  - `chebyshev` (aliases: `g-ci`, `g-nocomm-ci`): Chebyshev-iteration preconditioner used as the paper-style global CI path
- Operator: matrix-free N-dimensional Laplacian / Poisson stencil
- Boundary model:
  - dimension `x`: Dirichlet on the lower side, Neumann on the upper side
  - dimensions `y`, `z`, `w`: Neumann on the lower side, Dirichlet on the upper side
- Validation: analytical reference field sampled on the paper-style physical coordinates and solved with the same discrete operator
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

The default `--preconditioner-max-steps` is `24`, matching the CI iteration count used in the paper.
The default `--epsilon` is `1e-10`, matching the paper's relative stopping target.

Dimension-dependent size parameters:

- 1D: `--size-x`
- 2D: `--size-x --size-y`
- 3D: `--size-x --size-y --size-z`
- 4D: `--size-x --size-y --size-z --size-w`

## Examples

The benchmark uses a fixed physical spacing of `0.1` in every dimension. The first three coordinate origins are
the paper's `x=3.0`, `y=2.5`, and `z=10.0`; the `4D` extension uses `w=1.5`.

For `3D`, the reference field is:

```text
phi(x, y, z) = 10 + sin(x) + cos(y) + 3 sin(z) + x^2 y z - y^2
```

which yields the paper's forcing term:

```text
-Delta phi = sin(x) + cos(y) + 3 sin(z) - 2 y z + 2
```

The `1D`, `2D`, and `4D` cases use the same family with the unavailable coordinates removed or extended.

## Smoke Tests

These are the checked-in CTest-scale runs:

### 1D

```bash
cmake -S /workspace -B build-poisson-1d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=1
cmake --build build-poisson-1d --target poissonSolver -j 4
./build-poisson-1d/benchmark/poissonSolver/poissonSolver \
  --size-x 257 \
  --max-steps 300 \
  --epsilon 1e-8 \
  --preconditioner g-nocomm-ci \
  --preconditioner-max-steps 24
```

### 2D

```bash
cmake -S /workspace -B build-poisson-2d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=2
cmake --build build-poisson-2d --target poissonSolver -j 4
./build-poisson-2d/benchmark/poissonSolver/poissonSolver \
  --size-x 33 \
  --size-y 33 \
  --max-steps 250 \
  --epsilon 1e-8 \
  --preconditioner g-nocomm-ci \
  --preconditioner-max-steps 24
```

### 3D

```bash
cmake -S /workspace -B build-poisson-3d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=3
cmake --build build-poisson-3d --target poissonSolver -j 4
./build-poisson-3d/benchmark/poissonSolver/poissonSolver \
  --size-x 13 \
  --size-y 13 \
  --size-z 13 \
  --max-steps 250 \
  --epsilon 1e-8 \
  --preconditioner g-nocomm-ci \
  --preconditioner-max-steps 24
```

### 4D

```bash
cmake -S /workspace -B build-poisson-4d -Dalpaka_BENCHMARKS=ON -Dalpaka_BENCHMARK_POISSON_SOLVER_DIMS=4
cmake --build build-poisson-4d --target poissonSolver -j 4
./build-poisson-4d/benchmark/poissonSolver/poissonSolver \
  --size-x 7 \
  --size-y 7 \
  --size-z 7 \
  --size-w 7 \
  --max-steps 250 \
  --epsilon 1e-7 \
  --preconditioner g-nocomm-ci \
  --preconditioner-max-steps 24
```

## Runtime-Oriented Presets

These single-device runs are calibrated to land roughly in the `10s` to `60s` range on the verified CPU host,
with faster GPU backends expected:

- `1D`: `--size-x 1025 --max-steps 300 --epsilon 1e-8 --preconditioner g-nocomm-ci --preconditioner-max-steps 24`
- `2D`: `--size-x 33 --size-y 33 --max-steps 250 --epsilon 1e-8 --preconditioner g-nocomm-ci --preconditioner-max-steps 24`
- `3D`: `--size-x 13 --size-y 13 --size-z 13 --max-steps 250 --epsilon 1e-8 --preconditioner g-nocomm-ci --preconditioner-max-steps 24`
- `4D`: `--size-x 7 --size-y 7 --size-z 7 --size-w 7 --max-steps 250 --epsilon 1e-7 --preconditioner g-nocomm-ci --preconditioner-max-steps 24`

Actual runtime still depends on the selected CPU or GPU backend.
