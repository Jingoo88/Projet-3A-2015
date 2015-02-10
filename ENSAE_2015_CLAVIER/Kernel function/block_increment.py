__author__ = 'thomas'

import time
import numpy as np
import math

from numbapro import cuda
from pylab import imshow, show
from timeit import default_timer as timer
from numbapro.cudalib import curand
from numba import *
from imh_block import BlockIMH



"""
main device function to be used
"""

@cuda.autojit
def cuda_div(a, b, c,n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] / b[i]




def index(p,x):
    """ function meant to keep the index within range
    will create a circular permutation of the vectors"""
    if x >= p:
        return x-p
    else:
        return x


def likelihood(x,y,z):
    """ will compare the likelihood ratio of the new point
    compared to the actual one and choose based on this
    whether to change or keep the current state"""

    p = ((1 + x*x)*math.exp(-0.5*x*x))/((1 + y*y)*math.exp(-0.5*y*y))
    if p > z:
        return x
    else:
        return y


#We define our functions to be used on the device
index_gpu = cuda.jit(restype=uint16, argtypes= [uint16,uint16 ], device=True)(index)
likelihood_gpu = cuda.jit(restype=f4, argtypes=[f4,f4,f4], device=True)(likelihood)

@cuda.jit(argtypes=[f4,uint32,f4[:],f4[:],f4[:,:]])
def block_kernel(start, p, a_dev, b_dev, c_dev):
    """
    kernel will perform incrementation of the p chains
    at the same time
    """

    x = cuda.grid(1) # equals to threadIdx.x + blockIdx.x * blockDim.x



    if x <= p:
        for i in range(p):
            c_dev[i,0] = start #starting point is the same for all chains
        for i in range(p-1):
            c_dev[x,i+1] = likelihood_gpu(a_dev[index_gpu(p,x+i)], c_dev[x,i], b_dev[i*x + i])

def block_increment(start, n):

    cuda.select_device(0)
    stream = cuda.stream()
    blockdim = 256
    griddim = n//256 + 1
    c_host = np.zeros((n,n), dtype = np.float32)
    m_dev = curand.normal(0,1,n,dtype=np.float32, device=True)
    n_dev = curand.normal(0,1,n,dtype=np.float32, device=True)
    a_host =np.zeros(n,dtype=np.float32)
    a_dev = cuda.device_array_like(a_host)
    cuda_div[griddim, blockdim,stream](m_dev,n_dev,a_dev,n)
    #keeps a_dev on the device for the kernel ==> no access at this point to the device memory
    # so i cant know what appends to m_dev and n_dev best guess is python GC is
    # translated into desallocation on the device
    b_dev = curand.uniform((n*n),dtype=np.float32, device=True)
    c_dev = cuda.device_array_like(c_host, stream)
    block_kernel[griddim, blockdim,stream](start,n, a_dev, b_dev, c_dev)
    c_dev.copy_to_host(c_host,stream)
    stream.synchronize()

    return c_host






if __name__ == '__main__':

    t0 = time.time()
    n=8000
    stream = cuda.stream()
    blockdim = 256
    griddim = n//256 + 1
    c_host =  block_increment(0,n)
    stream.synchronize()
    cuda.close()
    print(c_host)



    print(time.time()-t0)
