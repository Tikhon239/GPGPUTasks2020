#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif
__kernel void max_prefix_sum(__global const int* sum_array, __global const int* prefix_array, __global const int* result_array,
                             __global int* new_sum_array, __global int* new_prefix_array, __global int* new_result_array,
                             unsigned int step, unsigned int n)
{
    const unsigned int localId = get_local_id(0);
    const unsigned groupId = get_group_id(0);
    const unsigned int globalId = get_global_id(0);

    __local int sum_local[WORK_GROUP_SIZE];
    __local int prefix_local[WORK_GROUP_SIZE];
    __local int result_local[WORK_GROUP_SIZE];
    if (localId < n) {
        sum_local[localId] = sum_array[globalId];
        prefix_local[localId] = prefix_array[globalId];
        result_local[localId] = result_array[globalId];
    }
    else
    {
        sum_local[localId] = -1;
        prefix_local[localId] = -1;
        result_local[localId] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0)
    {
        int sum = 0;
        int max_sum = 0;
        int result = 0;
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; ++i)
        {
            if (sum + prefix_local[i] > max_sum)
            {
                max_sum = sum + prefix_local[i];
                result = step * i + result_local[i];
            }
            sum += sum_local[i];
        }
        new_sum_array[groupId] = sum;
        new_prefix_array[groupId] = max_sum;
        new_result_array[groupId] = result;
    }
}