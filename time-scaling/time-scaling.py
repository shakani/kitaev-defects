from __future__ import print_function, division
#
import sys,os
import numpy.typing as npt 
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
    
    def __init__(self, on_the_fly: bool, L: int):
        '''
        Class constructor for wrapper (all initializations)
        '''
        self._on_the_fly = on_the_fly   # toggles between using `hamiltonian` or `quantum_LinearOperator`
        self._L = L     # chain length
        
        # Error catching for Heisenberg chain from example
        if (self._L//2)%2 != 0:
            raise ValueError("Example requires modifications for Heisenberg chains with L=4*n+2.")
        if self._L%2 != 0:
            raise ValueError("Example requires modifications for Heisenberg chains with odd number of sites.")
        
        self._S = "1/2" # on-site spin size
        self._T = (np.arange(self._L)+1) % self._L # translation transformation on sites [0, ..., L-1]
        
        # construct basis
        self._basis0 = spin_basis_general(self._L, S = self._S, m=0, pauli=False, kblock=(T,0))
        
        # construct static list for Heisenberg chain
        self._Jzz_list = [[1.0,i,(i+1)%L] for i in range(self._L)]
        self._Jxy_list = [[0.5,i,(i+1)%L] for i in range(self._L)]
        self._static = [["zz", self._Jzz_list],["+-", self._Jxy_list],["-+", self._Jxy_list]]
        
        # construct operator for Hamiltonian in the ground state sector
        if self._on_the_fly:
            self._H0 = quantum_LinearOperator(self._static, basis = self._basis0, dtype = np.float64)
        else:
            self._H0 = hamiltonian(self._static, [], basis = self._basis0, dtype = np.float64)
            
    def compute_ground_state(self, k: int = 1, which: str = "SA"):
        """
        Computes and returns `k` lowest energy states and energies of Hamiltonian.
        `which = "SA"` specifies smallest algebraic eigenvalue.
        """
        [E0], psi0 = self._H0.eigsh(k = k, which = which)
        psi0.ravel()
        return [E0], psi0 
    
    def compute_spectral_function(self, w_min: float = 0, w_max: float = 1, dw: float = 0.01, eta: float = 0.1):
        qs = np.arange(-self._L//2 + 1, self._L//2, 1)
        omegas = np.arange()