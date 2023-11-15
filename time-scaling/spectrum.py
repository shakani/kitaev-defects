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
    
    def __init__(self, on_the_fly: bool, L: int) -> None:
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
        self._basis0 = spin_basis_general(self._L, S = self._S, m=0, pauli=False, kblock=(self._T,0))
        
        # construct static list for Heisenberg chain
        self._Jzz_list = [[1.0,i,(i+1)%L] for i in range(self._L)]
        self._Jxy_list = [[0.5,i,(i+1)%L] for i in range(self._L)]
        self._static = [["zz", self._Jzz_list],["+-", self._Jxy_list],["-+", self._Jxy_list]]
        
        # construct operator for Hamiltonian in the ground state sector
        if self._on_the_fly:
            self._H0 = quantum_LinearOperator(self._static, basis = self._basis0, dtype = np.float64, check_symm=False)
        else:
            self._H0 = hamiltonian(self._static, [], basis = self._basis0, dtype = np.float64, check_symm=False)
            
        self._E0, self._psi0 = 0., None # default values for ground state & ground state energy
            
    def compute_ground_state(self, k: int = 1, which: str = "SA") -> None:
        """
        Computes `k` lowest energy states and energies of Hamiltonian.
        `which = "SA"` specifies smallest algebraic eigenvalue.
        """
        [E0], psi0 = self._H0.eigsh(k = k, which = which)
        psi0.ravel()
        
        self._E0, self._psi0 = E0, psi0
    
    def compute_spectral_function(self, w_min: float = 0, w_max: float = 1, dw: float = 0.01, eta: float = 0.1, verbose = False):
        """
        Computes SzSz spectral function.
        """
        qs = np.arange(-self._L//2 + 1, self._L//2, 1)
        omegas = np.arange(w_min, w_max, dw)
        
        # Allocate array to store data
        G = np.zeros(omegas.shape + qs.shape, dtype = np.complex128)
        
        # loop over momentum sectors
        for j, q in enumerate(qs):
            if verbose:
                print(f"Computing momentum block q = {q}")
            
            # define block
            block = dict(qblock = (self._T, q))
            
            # define operator list for Op_shift_sector
            f = lambda i: np.exp(-2j * np.pi * q * i / self._L) / np.sqrt(self._L)
            Op_list = [["z", [i], f(i)] for i in range(self._L)]
            
            # define basis
            basisq = spin_basis_general(self._L, S = self._S, m = 0, pauli = False, **block)
            
            # define operators in the q-momentum sector
            if self._on_the_fly:
                Hq = quantum_LinearOperator(self._static, basis = basisq, dtype = np.complex128,
                                            check_symm = False, check_pcon = False, check_herm = False)
                
            else:
                Hq = hamiltonian(self._static, [], basis = basisq, dtype = np.complex128,
                                 check_symm = False, check_pcon = False, check_herm = False)
                
            # shift sectors
            
            # raise exception if ground state hasn't been comptued
            if self._psi0 is None:
                raise ValueError("Ground state has not been computed")
            
            psiA = basisq.Op_shift_sector(self._basis0, Op_list, self._psi0)
            
            # solve (z-H)|x> = |A> solve for |x> using iterative solver for each omega
            for i, omega in enumerate(omegas):
                lhs = LHS(Hq, omega, eta, self._E0)
                x, *_ = sp.linalg.bicg(lhs, psiA)
                G[i,j] = -np.vdot(psiA, x)/np.pi 
                

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    mySpectrum = Spectrum(on_the_fly=False, L=12)