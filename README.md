# Overview

This library implements a class similar to the [GNU Scientific Library](https://www.gnu.org/software/gsl/) `matrix_view`. It points to a section of a C++ `vector` and interprets that section as a matrix. Matrix manipulations can then be done as usual. The matrices are [column-major](https://en.wikipedia.org/wiki/Matrix_representation). See documentation for the available methods.

## Dependencies

### BLAS/LAPACK

Interfaces to a subset of [BLAS](https://www.netlib.org/blas/) and [LAPACK](https://www.netlib.org/lapack/) routines is available. People prefer a variety of implementations of these libraries. I have to possibilities built-in:

- Using the libraries included with [R](https://cran.r-project.org/), useful for writing R extensions.
- Using a stand-alone library, in particular Intel's [Math Kernel Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html). To use this option, include the `-DNO_REXT` flag while compiling and make sure MKL is reachable by the compiler.

Using other libraries is also possible, but you may need to modify the `#include` statement for BLAS and LAPACK. If you want to use CBLAS, you may also need to modify routine names.

### Utilities

The implementation depends on a set of numerical utilities that I collected in the [bayesicUtilities](https://github.com/tonymugen/bayesicUtilities) repository. I assume that the utilities are available in a `bayesicUtilities` directory at the same level as `bayesicMatrix`. This can be changed by modifying the `#include` path.
