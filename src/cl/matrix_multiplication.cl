#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 16
#endif

__kernel void matmull(__global const float* A, __global const float* B, __global float* C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if (i >= N || j >= M)
        return;

    float sum = 0;
    for (unsigned int k = 0; k < K; ++k) {
        sum += A[j * K + k] * B[k * N + i];
    }

    C[j * N + i] = sum;
}

__kernel void matmull_with_transpose(__global const float* A, __global const float* B, __global float* C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if (i >= N || j >= M)
        return;

    float sum = 0;
    for (unsigned int k = 0; k < K; ++k) {
        sum += A[j * K + k] * B[i * N + k];
    }

    C[j * N + i] = sum;
}
__kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* C, const unsigned int M, const unsigned int K, const unsigned int N)
{
    __local float a_k[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float b_k[WORK_GROUP_SIZE][WORK_GROUP_SIZE];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    if (i < N && j < M)
    {
        float sum = 0;
        for (unsigned int k = 0; k < K; k += WORK_GROUP_SIZE) {
            a_k[local_j][local_i] = A[j * K + k + local_i];
            b_k[local_j][local_i] = B[(k + local_j) * N + i];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned int t = 0; t < WORK_GROUP_SIZE; ++t) {
                sum += a_k[local_j][t] * b_k[t][local_i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        C[j * N + i] = sum;
    }
}