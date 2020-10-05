#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif
__kernel void simple_sum(__global unsigned int* sum, __global const unsigned int* array)
{
    const unsigned int localId = get_local_id(0);
    const unsigned int globalId = get_global_id(0);

    __local unsigned int local_array[WORK_GROUP_SIZE];
    local_array[localId] = array[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0)
    {
        unsigned int temp_sum = 0;
        for(unsigned int i = 0; i < WORK_GROUP_SIZE; ++i)
            temp_sum += local_array[i];
        atomic_add(sum, temp_sum);
    }
}

__kernel void tree_sum(__global unsigned int* sum, __global const unsigned int* array)
{
    const unsigned int localId = get_local_id(0);
    const unsigned int globalId = get_global_id(0);

    __local unsigned int local_array[WORK_GROUP_SIZE];
    local_array[localId] = array[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2)
    {
        if (2 * localId < nvalues)
        {
            unsigned a = local_array[localId];
            unsigned b = local_array[localId + nvalues/2];
            local_array[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        atomic_add(sum, local_array[0]);
    }
}