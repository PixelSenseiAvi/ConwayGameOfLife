#include <curand_kernel.h>

__global__ void _random(unsigned int* cellData, float density, int seed);
__global__ void _next(unsigned int* cellData, unsigned int* cellNext);
__global__ void _copy(unsigned int* cellData, unsigned int* cellNext);