__author__ = 'thomas'

import numpy as np


class BlockIMH:

    def __init__(self, chain_start, block_size, num_blocks):
        self.chain_start = chain_start
        assert(block_size <= 8000), "BLOCK SIZE SHOULD NOT EXCEED 8000"
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.chain_length = block_size * num_blocks


        self.chain = {}
        self.chain[0] = np.ones((block_size,1), dtype= np.float32, order = 'F')*chain_start
        self.index = 0
        self.path = []

        self.long_chain = np.zeros(1, dtype = np.float32)





    def next_start(self):
        """ chooses the starting point for the next block
            randomly from the last state of the chain"""
        x = self.chain[self.index].shape[0]
        y = self.chain[self.index].shape[1]
        i = np.random.randint(x)
        self.path.append(i)
        next_start = self.chain[self.index][i,y-1]
        self.index +=1

        return next_start

    def concatenate_chain(self):

        """extremely brutal and time consuming method to concatenate
           the full markov chain, last minute add on with very little
           interest :) """
        j=1
        for i in self.path:

            np.concatenate((self.long_chain,self.chain[j][i,:]))
            j +=1

    def save_chain(self, block):
        """ the object will increment the chain
        by a block at each step. however, the function
        generating the next block is outside the class as
        of now, so we will use this method to concatenate
        and retain the complete chain"""


        assert(block.shape[0] == self.chain[self.index-1].shape[0]), "WRONG BLOCK SIZE"
        self.chain[self.index] = block







