# C port

A from-scratch **C11** port of every `numerical_recipes` module -- the same
verification-first tests and the same hand-rolled algorithms, with no dependency
beyond the C standard library and `-lm` (no NumPy, FFTW, or BLAS).

```
include/nr/*.h   public API (one header per module)
src/*.c          the routines (one per module) + common.c (matrix helpers, RNG)
tests/test_*.c   one test file per module
Makefile         builds build/libnr.a + each test
```

## Build and test

```bash
make test      # build the static lib + every test, run them, fail on first exit != 0
make           # build build/libnr.a only
make clean     # wipe build/
```

Flags: `-std=c11 -O2 -Wall -Wextra`. Complex data uses C99 `double complex`.

## Conventions

- Row-major `double` arrays; matrices `n*m` contiguous.
- The caller supplies output buffers; optional history/trajectory buffers are
  nullable; failing functions return `int` (`0` / `-1`).
- Callbacks carry a `void *ud` payload.
- RNG is `nr_rng` (xoshiro256** + Box-Muller) in `nr/common.h`, seeded per test.
- `common` holds the shared matrix/RNG helpers the Python package gets from NumPy.

See the [root README](../README.md) for the project's verification-first
philosophy and the Python implementation.
