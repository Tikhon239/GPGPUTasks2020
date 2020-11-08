#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

__kernel void merge(__global float* cur_as, __global float* next_as, const unsigned int n, const unsigned int group_size)
{
    const unsigned int global_id = get_global_id(0);
    if (global_id >= n)
        return;

    const unsigned int group_id = global_id / (2 * group_size);
    // work item соответсвует диагональ с номером local_id
    const unsigned int local_id = global_id % (2 * group_size);

    int step_b = max((int)0, (int)(local_id - group_size));
    int step_a = local_id - step_b - 1; //min((int)local_id, (int)group_size) - 1;
    // позиции массива b: 2 * group_size * group_id : 2 * group_size * group_id + group_size
    const unsigned int b_start_pos = 2 * group_size * group_id + step_b;
    // позиции массива a: 2 * group_size * group_id + group_size : 2 * group_size * (group_id + 1)
    const unsigned int a_start_pos = 2 * group_size * group_id + group_size + step_a;

    int L = -1;  // L + 1 + step_b сколько раз сдвинулись вправо
    int R = min((int)local_id, (int)(2 * group_size - local_id));

    while (L < R - 1) {
        int m = L + (R - L) / 2;
        if (cur_as[b_start_pos + m] <= cur_as[a_start_pos - m]) {
            // двигаемся вправо
            L = m;
        } else {
            // двигаемся влево
            R = m;
        }
    }

    if (local_id >= group_size && L + 1 + step_b == group_size) {
        next_as[global_id] = cur_as[a_start_pos - step_a + step_b];
    }
    else {
        if (local_id >= group_size && L + 1 == 0) {
            next_as[global_id] = cur_as[b_start_pos];
        }
        else {
            next_as[global_id] = min((float) cur_as[b_start_pos + L + 1], (float) cur_as[a_start_pos - L]);
        }
    }
}
