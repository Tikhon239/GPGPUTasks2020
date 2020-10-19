#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 256
#endif

__kernel do_some_work()
{
    assert(get_group_id == [256, 1, 1]);

    __local disjoint_set = ...;
    __local bool queue[WORK_GROUP_SIZE];

    const unsigned int local_id = get_local_id(0);

    for (int iters = 0; iters < 100; ++iters) {      // потоки делают сто итераций
        queue[local_id] = false; //очищаем queue от результатов на прошлых итерациях
        if (some_random_predicat(get_local_id(0))) { // предикат срабатывает очень редко (например шанс - 0.1%)
                                                     // на каждой итерации некоторые потоки
                                                     // могут захотеть обновить нашу структурку
            queue[local_id] = true;                   // добавляем потоки которые хотят обновлять в структуру

        }
        barrier(CLK_LOCAL_MEM_FENCE); // ждем пока все потоки из work group заполнят queue
        if (local_id == 0) //все операции union() будет делать первый поток
            for(int pseudo_local_id = 0; pseudo_local_id < WORK_GROUP_SIZE; ++pseudo_local_id)
                if queue[pseudo_local_id]
                    union(disjoint_set, ...);
                    //тут не нужен барьер, так как все делает первый поток
                    //(как понимаю если поставить сюда барьер то все сломается, так как сюда зайдет только первый поток)
        barrier(CLK_LOCAL_MEM_FENCE); //все потоки ждут пока первый поток выполнит все union()
        ...
        tmp = get(disjoint_set, ...); // потоки постоянно хотят читать из структурки
        ...
        //как понимаю это барьер не обязательный, так как потоки будут ждать друг друга не меняю структуру на первом барьере
        //barrier(CLK_LOCAL_MEM_FENCE); //ждем пока все потоки из work group выполнят get()
    }
}