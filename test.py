import multiprocessing

# L = list(range(0, 10000000))

# def per_proc(i):
#     return i + 1

# pool = multiprocessing.Pool(4)

# result = pool.map(per_proc, L)
# pool.close()
# pool.join()

# print(result)

a = []
for i in range(0, 10000000):
    a.append(i + 1)