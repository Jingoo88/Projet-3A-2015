__author__ = 'thomas.clavier'

__version__ = '0.1'
__maintainer__ = ['gilles.drigout', 'thomas.clavier']
__status__ = 'Development'


# Uses device generated random normal simulations to generate cauchy simulation
# Methods may be better if normal simulation are reused ==> to check

from numbapro import cuda
from numbapro import vectorize
from numbapro.cudalib import curand
import numpy as np
import matplotlib.pyplot as plt


cuda.select_device(0)

class Cauchy:
	
	def __init__(self, size):
		
		self.container = np.empty(size, np.float64)
		
	def __get_cuda_randoms(self):
	    
	    prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
	    prng.normal(self.container,0,1)
	    
	
	    #self.container = rand.reshape((x, y)) a completer

	@vectorize(['float64(float64, float64)'], target='gpu')
	def __Div(a, b):
		return a / b	    
		
		
	def get_cauchy_randoms(self):
		
	
		self.__get_cuda_randoms()
		np.random.shuffle(self.container)
		self.container = np.reshape(self.container, (2,self.container.shape[0]/2))
		self.sample = np.empty_like(self.container[0], dtype = self.container.dtype)
		self.sample = self.__Div(self.container[0], self.container[1])
		




if __name__ == '__main__':
	
	x = Cauchy(100000)
	x.get_cauchy_randoms()
	sample = x.sample[x.sample >-10]
	sample = sample[sample < 10]
	excluded = x.sample.shape[0] - sample.shape[0]
	t= "Number of values excluded = %i"%excluded


	n, bins, patches = plt.hist(sample,200, normed=1, facecolor='g', alpha=0.75)
	plt.xlabel("Samples")
	plt.ylabel('Probability')
	plt.text(-7, .4, t )
	plt.title('Histogram of Cauchy generation')
	plt.axis([-10, 10, 0, 0.5])
	plt.grid(True)
	plt.show()
