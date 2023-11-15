from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian,quantum_LinearOperator
import scipy.sparse as sp
import numexpr,cProfile
import numpy as np
import matplotlib.pyplot as plt

class LHS(sp.linalg.LinearOperator):
    '''
    Left-hand side of computing spectral function.
    '''
	#
    def __init__(self,H,omega,eta,E0,kwargs={}):
        self._H = H # Hamiltonian
        self._z = omega +1j*eta + E0 # complex energy
        self._kwargs = kwargs # arguments
	#
    @property
    def shape(self):
        return (self._H.Ns,self._H.Ns)
	#
    @property
    def dtype(self):
        return np.dtype(self._H.dtype)
	#
    def _matvec(self,v):
		# left multiplication
        return self._z * v - self._H.dot(v,**self._kwargs)
	#
    def _rmatvec(self,v):
		# right multiplication
        return self._z.conj() * v - self._H.dot(v,**self._kwargs)

class Spectrum():
    '''
    Wrapper for testing time scaling of example26 from quspin documentation.
    https://quspin.github.io/QuSpin/examples/example26.html#example26-label
    '''