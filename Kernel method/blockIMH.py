__author__ = 'gilles.drigout'

from proposal import *
from numpy import zeros, random
from numbapro import cuda
from numbapro.cudalib import curand
import math
import time

@cuda.autojit
def cu_one_block(x_start, y, omega, uniforms, result, size, chain_length):
    i = cuda.grid(1)

    if i < size:

        result[i,0] = x_start

        for t in range(1, chain_length):
            x_prev = result[i, t-1]
            acceptance_ratio = min(1, omega[t]*(1+x_prev**2)*math.exp(-0.5*x_prev**2))

            if i*size + t*chain_length < N*N:
                u = uniforms[i*size + t*chain_length]

                if u < acceptance_ratio:
                    result[i, t] = y[t]
                else:
                    result[i, t] = x_prev


class BlockIMH:

    def __init__(self, chain_start, block_size, num_blocks, proposals, omegas):
        self.chain_start = chain_start
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.chain_length = block_size * num_blocks
        self.proposals = proposals
        self.omegas = omegas
        self.block_count = 0
        self.block_start = chain_start

    def iterate_block(self):

        if self.block_count*self.block_size < self.chain_length:

            proposals_block = self.proposals[self.block_count*self.block_size:(self.block_count+1)*self.block_size]
            omegas_block = self.omegas[self.block_count*self.block_size:(self.block_count+1)*self.block_size]
            block = Block(size=self.block_size, proposals=proposals_block, omegas=omegas_block, start=self.block_start)
            block_values = block.compute_block()

            # drawing random integer for next block start
            rand_chain = random.random_integers(self.block_size-1)


            self.block_start = block_values[-1][rand_chain]
            self.block_count +=1

    def iterate_all_chain(self):

        while self.block_count*self.block_size < self.chain_length:
            t0 = time.time()
            self.iterate_block()
            print(time.time()-t0)


class Block:

    def __init__(self, size, proposals, omegas, start):
        self.size = size
        self.host_proposals = proposals
        self.host_omegas = omegas
        self.start = start
        self.threads_per_block = 512
        self.grid_dim = (self.size // self.threads_per_block)+ 1

    def compute_block(self):

        device_uniforms = curand.uniform(size=N*N, device=True)
        host_results = zeros((self.size, self.size))

        stream = cuda.stream()
        device_proposals = cuda.to_device(self.host_proposals, stream=stream)
        device_omegas = cuda.to_device(self.host_omegas, stream=stream)
        device_results = cuda.device_array_like(host_results, stream=stream)
        cu_one_block[self.grid_dim, self.threads_per_block, stream](self.start, device_proposals,
                                                                    device_omegas, device_uniforms,
                                                                    device_results, self.size, self.size)
        device_results.copy_to_host(host_results, stream=stream)

        stream.synchronize()

        return host_results

    @staticmethod
    @cuda.autojit
    def cu_block(x_start, y, omega, uniforms, result, size, chain_length):
        i = cuda.grid(1)

        if i < size:

            result[i,0] = x_start

            for t in range(1, chain_length):
                x_prev = result[i, t-1]
                acceptance_ratio = min(1, omega[t]*(1+x_prev**2)*math.exp(-0.5*x_prev**2))

                if i*size + t*chain_length < N*N:
                    u = uniforms[i*size + t*chain_length]

                    if u < acceptance_ratio:
                        result[i, t] = y[t]
                    else:
                        result[i, t] = x_prev


if __name__ == '__main__':

    N = 10000
    t = ToyExample(N)
    host_y = t.host_values
    host_omega = t.host_omegas

    imh = BlockIMH(chain_start=0, block_size=100, num_blocks=10, proposals=host_y, omegas=host_omega)
    imh.iterate_all_chain()



