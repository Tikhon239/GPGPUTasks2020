#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    //cpu
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    //cpu OpenMP
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    //gpu
    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        as.resize(global_work_size, 0);
        gpu::gpu_mem_32u buffer;
        buffer.resizeN(global_work_size);
        // Прогружаем данные из вектора as
        // (есть нетипизированный метод write для которого количество измеряется в байтах,
        // и типизированный writeN, для которого количество измеряется в количестве float-элементов, т.к. gpu::gpu_mem_32f - это shared_device_buffer_typed<float>)
        buffer.writeN(as.data(), global_work_size);
        gpu::gpu_mem_32u sum_buffer;
        sum_buffer.resizeN(1);
        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "simple_sum");
            bool printLog = false;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_buffer.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_buffer, buffer);
                sum_buffer.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU simple_sum:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU simple_sum:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "tree_sum");
            bool printLog = false;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_buffer.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_buffer, buffer);
                sum_buffer.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU tree_sum:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree_sum:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }

    return 0;
}
