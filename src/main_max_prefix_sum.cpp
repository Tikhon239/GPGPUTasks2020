#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

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
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;

        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")"
                  << std::endl;

        //cpu
        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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

            as.resize(global_work_size, -1);
            gpu::gpu_mem_32i sum_buffer;
            sum_buffer.resizeN(global_work_size);
            sum_buffer.writeN(as.data(), global_work_size);
            gpu::gpu_mem_32i prefix_buffer;
            prefix_buffer.resizeN(global_work_size);
            prefix_buffer.writeN(as.data(), global_work_size);
            std::vector<int> temp(global_work_size, 1);
            gpu::gpu_mem_32i result_buffer;
            result_buffer.resizeN(global_work_size);
            result_buffer.writeN(temp.data(), global_work_size);


            gpu::gpu_mem_32i cur_sum_buffer;
            cur_sum_buffer.resizeN(global_work_size);
            gpu::gpu_mem_32i cur_prefix_buffer;
            cur_prefix_buffer.resizeN(global_work_size);
            gpu::gpu_mem_32i cur_result_buffer;
            cur_result_buffer.resizeN(global_work_size);

            gpu::gpu_mem_32i new_sum_buffer;
            new_sum_buffer.resizeN(global_work_size);
            gpu::gpu_mem_32i new_prefix_buffer;
            new_prefix_buffer.resizeN(global_work_size);
            gpu::gpu_mem_32i new_result_buffer;
            new_result_buffer.resizeN(global_work_size);

            ocl::Kernel kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            bool printLog = false;
            kernel.compile(printLog);
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter)
            {
                int max_sum = 0;
                int result = 0;
                unsigned int step = 1;
                sum_buffer.copyToN(cur_sum_buffer, global_work_size);
                prefix_buffer.copyToN(cur_prefix_buffer, global_work_size);
                result_buffer.copyToN(cur_result_buffer, global_work_size);
                int cur_size = global_work_size;
                while (cur_size > 1)
                {
                    int next_size = (cur_size + workGroupSize - 1) / workGroupSize;
                    //new_sum_buffer.resizeN(next_size);
                    //new_prefix_buffer.resizeN(next_size);
                    //new_result_buffer.resizeN(next_size);
                    kernel.exec(gpu::WorkSize(workGroupSize, cur_size), cur_sum_buffer, cur_prefix_buffer,
                            cur_result_buffer, new_sum_buffer, new_prefix_buffer, new_result_buffer, step, cur_size);

                    new_sum_buffer.copyToN(cur_sum_buffer, next_size);
                    new_prefix_buffer.copyToN(cur_prefix_buffer, next_size);
                    new_result_buffer.copyToN(cur_result_buffer, next_size);

                    /*
                    cur_sum_buffer.clmem();
                    cur_sum_buffer.resizeN(next_size);
                    new_sum_buffer.copyToN(cur_sum_buffer, next_size);

                    cur_prefix_buffer.clmem();
                    cur_prefix_buffer.resizeN(next_size);
                    new_prefix_buffer.copyToN(cur_prefix_buffer, next_size);

                    cur_result_buffer.clmem();
                    cur_result_buffer.resizeN(next_size);
                    new_result_buffer.copyToN(cur_result_buffer, next_size);
                    */
                    cur_size = next_size;
                    step *= workGroupSize;
                }
                cur_prefix_buffer.readN(&max_sum, 1);
                cur_result_buffer.readN(&result, 1);

                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
    return 0;
}
