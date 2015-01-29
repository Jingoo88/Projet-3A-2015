__author__ = 'gilles.drigout'

__version__ = '0.1'
__maintainer__ = ['gilles.drigout', 'thomas.clavier']
__status__ = 'Development'

from numbapro.cudalib import curand
import numpy as np
import time as t
import matplotlib.pyplot as plt

def get_randoms(x, y):
    rand = np.random.standard_normal((x, y))
    return rand

def get_cuda_randoms(x, y):
    rand = np.empty((x * y), np.float64)
    prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
    prng.normal(rand, 0, 1) # filling the container
    rand = rand.reshape((x, y))
    return rand

def time_comparison(factor):
    cuda_times = list()
    cpu_times = list()
    for j in range(1, 10000, step):
        i = j * factor
        t0 = t.time()
        a = get_randoms(i, 1)
        t1 = t.time()
        cpu_times.append(t1 - t0)
        t2 = t.time()
        a = get_cuda_randoms(i, 1)
        t3 = t.time()
        cuda_times.append(t3 - t2)

    print("Bytes of largest array %i" % a.nbytes)
    return cuda_times, cpu_times

def plot_results(cpu_times, cuda_times, factor):
    plt.plot(x * factor, cpu_times,'b', label='NUMPY')
    plt.plot(x * factor, cuda_times, 'r', label='CUDA')
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('size of random number array')
    plt.ylabel('time')
    plt.axis('tight')
    plt.show()

step = 1000

if __name__ == '__main__':

    factor = 5000
    cuda_times, cpu_times = time_comparison(factor)
    x = np.arange(1, 10000, step)
    plot_results(cpu_times, cuda_times, factor)


