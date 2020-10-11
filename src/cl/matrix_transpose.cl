#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 16
#endif

__kernel void simple_matrix_transpose(__global const float* matrix, __global float* matrix_transpose, const unsigned int M, const unsigned int K)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if (i >= K || j >= M)
        return;

    matrix_transpose[i * M + j] = matrix[j * K + i];
}

__kernel void bank_conflict_matrix_transpose(__global const float* matrix, __global float* matrix_transpose, const unsigned int M, const unsigned int K)
{
    __local float local_matrix[WORK_GROUP_SIZE][WORK_GROUP_SIZE];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);
    if (i < K && j < M)
        local_matrix[local_j][local_i] = matrix[j * K + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (group_i * WORK_GROUP_SIZE + local_j < K && group_j * WORK_GROUP_SIZE + local_i < M)
        matrix_transpose[(group_i * WORK_GROUP_SIZE + local_j) * M + group_j * WORK_GROUP_SIZE + local_i] = local_matrix[local_i][local_j];
}

__kernel void with_out_bank_conflict_matrix_transpose(__global const float* matrix, __global float* matrix_transpose, const unsigned int M, const unsigned int K)
{
    __local float local_matrix[WORK_GROUP_SIZE][WORK_GROUP_SIZE+1];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);
    if (i < K && j < M)
        local_matrix[local_j][local_i] = matrix[j * K + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (group_i * WORK_GROUP_SIZE + local_j < K && group_j * WORK_GROUP_SIZE + local_i < M)
        matrix_transpose[(group_i * WORK_GROUP_SIZE + local_j) * M + group_j * WORK_GROUP_SIZE + local_i] = local_matrix[local_i][local_j];
}