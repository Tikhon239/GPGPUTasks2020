#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 4 //128
#endif

//как довести до рабочего состония
__kernel void smth(__global unsigned int* as, __global unsigned int* prefix_sum, const unsigned int n)
{
    __local unsigned int local_array[WORK_GROUP_SIZE];

    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    if (global_id < n)
        local_array[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int pow = 0; pow <= 2; ++pow) {
        if (((local_id + 1) >> pow) & 1) {
            prefix_sum[global_id] += local_array[(local_id + 1) / (1 << pow) - 1];
        }
        if (local_id % 2 == 0)
            local_array[local_id / 2] = local_array[local_id] + local_array[local_id + 1];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void get_inverse_bit(__global unsigned int* as, __global unsigned int* bit_array, const unsigned int n, const unsigned int bit) {
    const unsigned int global_id = get_global_id(0);
    if (global_id < n)
        bit_array[global_id] = 1 - (as[global_id] >> bit) & 1;
}

__kernel void prefix_sum(__global unsigned int* partial_sum, __global unsigned int* prefix_sum, const unsigned int n, const unsigned int pow) {
    const unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        if (((global_id + 1) >> pow) & 1)
            prefix_sum[global_id] += partial_sum[(global_id + 1) / (1 << pow) - 1];
    }
}

__kernel void partial_sum(__global unsigned int* cur_partial_sum, __global unsigned int* next_partial_sum, const unsigned int n) {
    const unsigned int global_id = get_global_id(0);
    if (global_id < n)
        if (global_id % 2 == 0)
            next_partial_sum[global_id/2] = cur_partial_sum[global_id] + cur_partial_sum[global_id + 1];
}

__kernel void radix(__global unsigned int* cur_as, __global unsigned int* next_as, __global unsigned int* prefix_sum, const unsigned int n, const unsigned int bit) {
    const unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        if (1 - (cur_as[global_id] >> bit) & 1)
            next_as[prefix_sum[global_id] - 1] = cur_as[global_id];
        else
            next_as[prefix_sum[n - 1] + global_id - prefix_sum[global_id]] = cur_as[global_id];
    }
}



