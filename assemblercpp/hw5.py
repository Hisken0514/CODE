def trmm_worker(A, B, alpha, result_queue):
    result = alpha * np.dot(A, B)
    result_queue.put(result)

def parallel_trmm(A, B, alpha, num_workers):
    result_queue = mp.Queue()

    workers = []
    for i in range(num_workers):
        p = mp.Process(target=trmm_worker, args=(A, B, alpha, result_queue))
        workers.append(p)
        p.start()

    B_final = result_queue.get()
    for p in workers:
        p.join()

    return B_final

# Example matrices
A = np.tril(np.random.rand(3, 3))  # Lower triangular matrix
B = np.random.rand(3, 3)

result = parallel_trmm(A, B, alpha, 2)
print("TRMM result:\n", result)
