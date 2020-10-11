#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = 1024;
    unsigned int K = 1024;

    std::vector<float> as(M*K, 0);
    std::vector<float> as_t(M*K, 0);

    FastRandom r(M+K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

    unsigned int work_group_size_x = 16;
    unsigned int global_work_size_x = (K + work_group_size_x - 1) / work_group_size_x * work_group_size_x;
    unsigned int work_group_size_y = work_group_size_x;
    unsigned int global_work_size_y = (M + work_group_size_y - 1) / work_group_size_y * work_group_size_y;


    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);
    as_gpu.writeN(as.data(), M*K);

    {
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "simple_matrix_transpose");
        matrix_transpose_kernel.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_transpose_kernel.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x,  global_work_size_y), as_gpu, as_t_gpu, M, K);
            t.nextLap();
        }
        as_t_gpu.readN(as_t.data(), M*K);
        std::cout << "simple matrix transpose GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "simple matrix transpose GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Same matrix!" << std::endl;

    {
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "bank_conflict_matrix_transpose");
        matrix_transpose_kernel.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_transpose_kernel.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x,  global_work_size_y), as_gpu, as_t_gpu, M, K);
            t.nextLap();
        }
        as_t_gpu.readN(as_t.data(), M*K);
        std::cout << "bank conflict matrix transpose GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "bank conflict matrix transpose GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {

                std::cerr << "Not the same! " << j << " " << i << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Same matrix!" << std::endl;

    {
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "with_out_bank_conflict_matrix_transpose");
        matrix_transpose_kernel.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_transpose_kernel.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x,  global_work_size_y), as_gpu, as_t_gpu, M, K);
            t.nextLap();
        }
        as_t_gpu.readN(as_t.data(), M*K);
        std::cout << "with out bank conflict matrix transpose GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "with out bank conflict matrix transpose GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Same matrix!" << std::endl;

    return 0;
}
