#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

__kernel void bitonic(__global float* as, const unsigned int n, unsigned int k, unsigned int temp_k)
{
    const unsigned int global_id = get_global_id(0);
    const bool flag = (global_id / k) % 2 == 0;

    //не нравится что такая тяжелая формула, может можно сделать битовые операции, так как temp_k - степень двойки?
    unsigned int cur_id = global_id % temp_k + (global_id / temp_k) * 2 * temp_k;
    if (cur_id + temp_k < n) {
        float a = as[cur_id];
        float b = as[cur_id + temp_k];
        if ((a > b) == flag) {
            as[cur_id] = b;
            as[cur_id + temp_k] = a;
            }
    }
}

__kernel void local_bitonic(__global float* as, const unsigned int n, unsigned int k, unsigned int temp_k)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local float local_array[2*WORK_GROUP_SIZE];
    //наверное так не очень круто делать?
    if (2*global_id < n)
        local_array[2*local_id] = as[2*global_id];
    if (2*global_id + 1 < n)
        local_array[2*local_id + 1] = as[2*global_id + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    const bool flag = (global_id / k) % 2 == 0;

    while (temp_k >= 1) {
        //не нравится что такая тяжелая формула, может можно сделать битовые операции, так как temp_k - степень двойки?
        unsigned int cur_id = local_id % temp_k + (local_id / temp_k) * 2 * temp_k;
        if (global_id  - local_id + cur_id + temp_k < n) {
            float a = local_array[cur_id];
            float b = local_array[cur_id + temp_k];
            if ((a > b) == flag) {
                local_array[cur_id] = b;
                local_array[cur_id + temp_k] = a;
            }
        }
            barrier(CLK_LOCAL_MEM_FENCE);
            temp_k >>= 1;
    }

    if (2*global_id < n)
        as[2*global_id] = local_array[2*local_id];
    if (2*global_id + 1 < n)
        as[2*global_id + 1] = local_array[2*local_id + 1];
}
