import pyarrow
import pandas as pd


# впринципе заменяемо на словарь но тогда код некрасивый

class Task:
    def __init__(self, owner: int, time: int):
        self.owner = owner
        self.time = time
        self.remaining_time = time
        self.waiting_time = 0
        self.is_waiting = True

    def get_is_waiting(self):
        return self.is_waiting

    def lock(self):
        self.is_waiting = False

    def work(self):
        self.waiting_time = 0
        if self.remaining_time > 0:
            self.remaining_time -= 1

    def get_time(self):
        return self.time

    def get_remaining_time(self):
        return self.remaining_time

    def is_done(self):
        return self.remaining_time <= 0

    def wait(self):
        self.is_waiting = True
        self.waiting_time += 1

    def get_waiting_time(self):
        return self.waiting_time

    def __str__(self):
        return "P{}({})".format(self.owner, self.remaining_time)


# любой пул такой типа:

class AnyPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.running_tasks = [None for i in range(self.capacity)]
        self.tasks_queue = []

    def get_running_tasks(self):
        return self.running_tasks

    def add_task_to_queue(self, process_owner: int, task_time: int):
        default_task = Task(process_owner, task_time)
        self.tasks_queue.append(default_task)

    # разные реализации
    def get_and_lock_task_from_queue(self):
        pass

    def displace_running_tasks(self):
        finished_tasks = []

        for index, task in enumerate(self.running_tasks):
            if task is not None:
                if task.is_done():
                    finished_tasks.append(task)
                    self.running_tasks[index] = None

        # удаляем их референсы в очереди
        for task in finished_tasks:
            try:
                self.tasks_queue.remove(task)
            except ValueError:
                pass

        return finished_tasks

    def tick_get_part(self):
        # ждущим рабочим назначаем задачи
        for index, task in enumerate(self.running_tasks):
            if task is None:
                self.running_tasks[index] = self.get_and_lock_task_from_queue()

    def tick_work_part(self):
        # рабочие работают если у них есть задачи
        for task in self.running_tasks:
            if task is not None:
                task.work()

    def tick_wait_part(self):
        # завершенные работы (когда всё время обслуживания выполненно) отходят от дел
        finished_tasks = self.displace_running_tasks()

        # ждуны ждут
        for task in self.tasks_queue:
            if task not in self.running_tasks:
                task.wait()

        # возвращаем что завершили
        return finished_tasks


# сами пулы рабочих

class FCFSPool(AnyPool):
    def get_and_lock_task_from_queue(self):
        if len(self.tasks_queue) > 0:
            self.tasks_queue[0].lock()
            return self.tasks_queue.pop(0)
        return None


class SPNPool(AnyPool):
    def get_and_lock_task_from_queue(self):
        next_process = -1
        min_time = 2 ** 15

        for index, task in enumerate(self.tasks_queue):
            if task.get_is_waiting() and min_time > task.get_time():
                min_time = task.get_time()
                next_process = index

        if next_process == -1:
            return None

        self.tasks_queue[next_process].lock()
        return self.tasks_queue.pop(next_process)


class SRTPool(AnyPool):
    def get_and_lock_task_from_queue(self):
        next_process = -1
        min_remaining_time = 100

        for index, task in enumerate(self.tasks_queue):
            if task.get_is_waiting() and min_remaining_time > task.get_remaining_time():
                min_remaining_time = task.get_remaining_time()
                next_process = index

        if next_process == -1:
            return None

        self.tasks_queue[next_process].lock()
        return self.tasks_queue[next_process]

    def displace_running_tasks(self):
        finished_tasks = []

        for index, task in enumerate(self.running_tasks):
            if task is not None:
                if task.is_done():
                    finished_tasks.append(task)
                    self.running_tasks[index] = None

        # удаляем их референсы в очереди
        for task in finished_tasks:
            try:
                self.tasks_queue.remove(task)
            except ValueError:
                pass

        for index, task in enumerate(self.running_tasks):
            self.running_tasks[index] = None

        return finished_tasks


class HRRNPool(AnyPool):
    def get_and_lock_task_from_queue(self):
        next_process = -1
        max_response_ratio = -1

        for index, task in enumerate(self.tasks_queue):
            response_ratio = (task.get_waiting_time() + task.get_time()) / task.get_time()
            if task.get_is_waiting() and max_response_ratio < response_ratio:
                max_response_ratio = response_ratio
                next_process = index

        if next_process == -1:
            return None

        self.tasks_queue[next_process].lock()
        return self.tasks_queue[next_process]


class RRPool(AnyPool):
    def __init__(self, capacity: int, time_slicing: int):
        super().__init__(capacity)
        self.time_slicing = time_slicing
        self.workers_timers = [0 for i in range(capacity)]

    def displace_running_tasks(self):
        finished_tasks = []

        for index, task in enumerate(self.running_tasks):
            if task is None:
                continue

            self.workers_timers[index] += 1

            if task.is_done():
                self.workers_timers[index] = 0
                finished_tasks.append(task)
                self.running_tasks[index] = None

            elif self.workers_timers[index] >= self.time_slicing:
                self.workers_timers[index] = 0
                self.running_tasks[index] = None

        for task in finished_tasks:
            try:
                self.tasks_queue.remove(task)
            except ValueError:
                pass

        return finished_tasks

    def get_and_lock_task_from_queue(self):
        next_process = -1
        max_waiting_time = -1

        for index, task in enumerate(self.tasks_queue):
            print(task.owner, len(self.tasks_queue), max_waiting_time, task.get_waiting_time())
            if task.get_is_waiting() and max_waiting_time <= task.get_waiting_time():
                max_waiting_time = task.get_waiting_time()
                next_process = index

        if next_process == -1:
            return None

        self.tasks_queue[next_process].lock()
        return self.tasks_queue[next_process]

    def tick_get_part(self):
        # ждущим рабочим назначаем задачи
        for index, task in enumerate(self.running_tasks):
            if task is None:
                self.running_tasks[index] = self.get_and_lock_task_from_queue()
                self.workers_timers[index] = 0


# для удобной записи входных значений

class TaskType:
    def __init__(self, time: int):
        self.time = time
        self.task_type = None


CPU_task_type = 0


class CPU(TaskType):
    def __init__(self, time: int):
        super().__init__(time)
        self.task_type = CPU_task_type


IO1_task_type = 1


class IO1(TaskType):
    def __init__(self, time: int):
        super().__init__(time)
        self.task_type = IO1_task_type


IO2_task_type = 2


class IO2(TaskType):
    def __init__(self, time: int):
        super().__init__(time)
        self.task_type = IO2_task_type


class Process:
    def __init__(self, pid: int, tasks_queue: list):
        self.pid = pid
        self.tasks_queue = tasks_queue
        self.tasks_count = len(self.tasks_queue)
        self.current_task = 0
        self.waiting_result = False

    def get_is_done(self):
        return self.current_task >= self.tasks_count

    def get_current_task(self):
        if not self.waiting_result and self.current_task < self.tasks_count:
            self.waiting_result = True
            return self.tasks_queue[self.current_task]

        return None

    def task_is_done(self):
        self.waiting_result = False
        self.current_task += 1


class Lab1:
    def __init__(self, cpu_pool: AnyPool):
        self.cpu_count = cpu_pool.capacity
        self.cpu_pool = cpu_pool
        self.io1_pool = FCFSPool(1)
        self.io2_pool = FCFSPool(1)

        '''
        self.processes = [
            Process(0, [CPU(6), IO1(14), CPU(6), IO1(20), CPU(2), IO1(14)]),
            Process(1, [CPU(2), IO2(20), CPU(6), IO1(18), CPU(6), IO2(20), CPU(10), IO1(12), CPU(6), IO1(20), CPU(4), IO1(10), CPU(4), IO1(14)]),
            Process(2, [CPU(8), IO2(10), CPU(4), IO1(20), CPU(6), IO1(12), CPU(10), IO2(20), CPU(6), IO1(14)]),
            Process(3, [CPU(48), IO2(14), CPU(12), IO2(18), CPU(12), IO1(10), CPU(60), IO2(12), CPU(24), IO2(14)]),
            Process(4, [CPU(60), IO1(12), CPU(60), IO1(20), CPU(24), IO1(18), CPU(24), IO2(10), CPU(24), IO2(10)]),
            Process(5, [CPU(24), IO2(18), CPU(36), IO2(20), CPU(60), IO2(14), CPU(12), IO2(10), CPU(48), IO1(12)])
        ]
        '''

        self.processes = [
            Process(0, [CPU(3)]),
            Process(1, [CPU(6)]),
            Process(2, [CPU(4)]),
            Process(3, [CPU(5)]),
            Process(4, [CPU(2)])
        ]

        self.processes_in_system = []

        self.df_header = ["CPU {}".format(i) for i in range(1, self.cpu_count + 1)] + ["IO 1", "IO 2"]
        self.output_data = []

    def manage_processes_and_do_tick(self):
        # проверяем все ли процессы закончили работу
        all_are_dead = True
        for process in self.processes_in_system:
            all_are_dead = all_are_dead and process.get_is_done()

        # система доработала все процессы
        if all_are_dead:
            return False

        # каждый процесс который не ждет выполнение задачи на CPU, IO1, или IO2, должен передать свою задачу в пул и ждать
        for process in self.processes_in_system:
            if not process.waiting_result and not process.get_is_done():
                current_task = process.get_current_task()
                if current_task.task_type == CPU_task_type:
                    self.cpu_pool.add_task_to_queue(process.pid, current_task.time)
                elif current_task.task_type == IO1_task_type:
                    self.io1_pool.add_task_to_queue(process.pid, current_task.time)
                else:
                    self.io2_pool.add_task_to_queue(process.pid, current_task.time)

        # все пулы работают (типа параллельно) и возвращают решенные задачки
        self.cpu_pool.tick_get_part()
        self.io1_pool.tick_get_part()
        self.io2_pool.tick_get_part()

        # состояние на данный момент
        cpus = []
        for i in range(self.cpu_count):
            if self.cpu_pool.running_tasks[i] is None:
                cpus.append(" ")
            else:
                cpus.append("P{}({})".format(self.cpu_pool.running_tasks[i].owner + 1,
                                             self.cpu_pool.running_tasks[i].remaining_time))

        ios1 = [" " if self.io1_pool.running_tasks[0] is None else "P{}({})".format(self.io1_pool.running_tasks[0].owner + 1, self.io1_pool.running_tasks[0].remaining_time)]
        ios2 = [" " if self.io2_pool.running_tasks[0] is None else "P{}({})".format(self.io2_pool.running_tasks[0].owner + 1, self.io2_pool.running_tasks[0].remaining_time)]

        output_line = cpus + ios1 + ios2
        self.output_data.append(output_line)

        # остальные части тика
        self.cpu_pool.tick_work_part()
        self.io1_pool.tick_work_part()
        self.io2_pool.tick_work_part()

        cpu_results = self.cpu_pool.tick_wait_part()
        io1_results = self.io1_pool.tick_wait_part()
        io2_results = self.io2_pool.tick_wait_part()
        results = cpu_results + io1_results + io2_results

        # процессы понимают что их текущие задачи выполнены и готовят указатель на следующие
        for task in results:
            self.processes_in_system[task.owner].task_is_done()

        # система наконец сделала полный тик

        # система продолжает работать над процессами
        return True

    def run(self):
        # то самое про новый процесс каждые 2 мс
        for i in range(len(self.processes)):
            self.processes_in_system.append(self.processes[i])

            self.manage_processes_and_do_tick()
            self.manage_processes_and_do_tick()

        # работаем пока процессы не закончили своё выполнение
        while self.manage_processes_and_do_tick():
            pass

        df = pd.DataFrame(self.output_data, columns=self.df_header)
        return df


cpu_count = 1
out = Lab1(FCFSPool(cpu_count)).run()
out.to_csv("./results/FCFS.csv")

out = Lab1(SPNPool(cpu_count)).run()
out.to_csv("./results/SPN.csv")

out = Lab1(SRTPool(cpu_count)).run()
out.to_csv("./results/SRT.csv")

out = Lab1(RRPool(cpu_count, 1)).run()
out.to_csv("./results/RR1.csv")

out = Lab1(RRPool(cpu_count, 4)).run()
out.to_csv("./results/RR4.csv")

out = Lab1(HRRNPool(cpu_count)).run()
out.to_csv("./results/HRRN.csv")

