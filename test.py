from multiprocessing import Process, Queue


def proc(q: Queue):
    while True:
        if not q.empty():
            val = q.get()
            print(val)


if __name__ == '__main__':
    q = Queue()
    p = Process(target=proc, args=(q,))
    p.daemon = True
    p.start()
    for i in range(100):
        q.put(i)
        