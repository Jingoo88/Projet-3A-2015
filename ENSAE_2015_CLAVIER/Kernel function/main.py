__author__ = 'thomas'

import time
import numpy as np
import matplotlib.pyplot as plt
import math

from imh_block import BlockIMH
from proposal import Cauchy
from block_increment import block_increment



from numbapro import cuda
from pylab import imshow, show

from numbapro.cudalib import curand
from numba import *



if __name__ == '__main__':
    #enter the parameters
    N = 1000000
    start = 12
    block_size = 3000
    num_blocks = N//block_size +1
    #create the chain object
    chain = BlockIMH(start, block_size, num_blocks)
    print("Generating a",N,"x",block_size, "chain")

    total_time = time.time()
    t = []
    tk =[]
    for i in range(num_blocks):
        t0 = time.time()
        start = chain.next_start()
        #booooooooooom got 0.05s to spend for 3000*3000 incrementation?
        block = block_increment(start,block_size)

        tk.append(time.time()-t0)
        chain.save_chain(block)
        t.append(time.time()-t0)
    print(time.time()-total_time)


    """
    we basically plot the time of the kernel and the time of the full chain
    time of execution can go wild if the total chain becomes too large and is sent to the swap
    """


    plt.plot(t)
    plt.plot(tk)
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('number of blocks')
    plt.ylabel('time')
    plt.show()
