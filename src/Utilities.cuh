#include <curand_kernel.h>

__global__ void _random(unsigned int* cellData, float density, int seed);
__global__ void _next(unsigned int* cellData, unsigned int* cellNext);
__global__ void _copy(unsigned int* cellData, unsigned int* cellNext);

////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}