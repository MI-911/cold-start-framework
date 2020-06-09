from multiprocessing.context import Process
from time import sleep

from loguru import logger


class Consumer(Process):
    def __init__(self, task_queue, result_queue):
        Process.__init__(self)

        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        i = 0
        while True:
            next_task = self.task_queue.get()
            i += 1
            if not next_task:
                while i < 100000:
                    i += 1
                logger.warning(f'Received kill signal on {self.name}')

                break

            logger.debug('%s: %s' % (self.name, next_task))

            answer = next_task()

            self.task_queue.task_done()
            self.result_queue.put(answer)


class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        i = 0
        while i < 10:
            i += 1
            # time.sleep(0.5)
        # time.sleep(5) # pretend to take some time to do the work
        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)

    def __str__(self):
        return 'users %s * %s' % (self.a, self.b)
