import numba
import numpy as np
from tqdm import tqdm
from scipy import linalg
from ase.units import Hartree, Bohr
from itertools import product
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.utilities import unpack
from numpy.linalg import solve
from gpaw.mixer import DummyMixer

def smooth(t,dt,steps):
    return 1.-3.*(t/(steps*dt))**2+2.*(t/(steps*dt))**3

@numba.njit(parallel=True)
def iteration(nq,dt,H,I,wfn):
    num_proc=6
    for i in numba.prange(num_proc):
        for q in range(i,nq,num_proc):
            H_left =I+0.5j*dt*H[q]
            H_right=I-0.5j*dt*H[q]
            wfn[q]=solve(H_left, np.dot(H_right,wfn[q]))
    return wfn

class TDSE(object):
    def __init__(self,calc):
        self.calc=calc
        self.wfs=self.calc.wfs
        self.nq=len(calc.wfs.kpt_u)
        self.nbands=calc.get_number_of_bands()
        self.E=np.zeros((self.nq,self.nbands,self.nbands),dtype=np.complex)
        self.norm=calc.wfs.gd.dv
        self.volume=np.abs(np.linalg.det(calc.wfs.gd.cell_cv)) 
        self.fqn=np.zeros((self.nq,self.nbands))
        for q in range(self.nq):
            kpt=calc.wfs.kpt_u[q]
            self.fqn[q]=kpt.f_n
            self.E[q]=np.diag(kpt.eps_n)
        self.get_momentum_matrix()
        
    def get_momentum_matrix(self):
        self.momentum=np.zeros((3,self.nq,self.nbands,self.nbands),dtype=np.complex)
        for q in range(self.nq):
            kpt = self.calc.wfs.kpt_u[q]
            G=self.wfs.pd.get_reciprocal_vectors(q=kpt.q,add_q=False)
            for n in range(self.nbands):
                for m in range(self.nbands):
                    for i in range(3):
                        self.momentum[i,q,n,m]=self.wfs.pd.integrate(kpt.psit_nG[m],G[:,i]*kpt.psit_nG[n])
        return self.momentum
                        
    def current(self):
        return np.einsum('qn,qin,qjn,dqij->d',self.fqn,self.wfn.conj(),self.wfn,self.momentum)
    
    def linear_response(self,dt,steps,A0=[0,0,1e-5]):
        self.occupation=np.zeros((steps,self.nq,self.nbands))
        self.A0=A0
        self.J=np.zeros((steps,3),dtype=np.complex)
        
        I=np.eye(self.nbands)
        H=self.E+np.einsum('iqnm,i->qnm',self.momentum,A0)+I*np.linalg.norm(A0)**2
        self.wfn=np.zeros((self.nq,self.nbands,self.nbands),dtype=np.complex)
        
        for q in range(self.nq):
            E,D=np.linalg.eigh(H[q])
            self.wfn[q]=D
            
        for t in tqdm(range(steps)):
            self.J[t]=self.current()
            self.occupation[t]=np.einsum('qn,qin->qn',self.fqn,np.abs(self.wfn)**2)
            self.wfn=iteration(self.nq,dt,self.E,I,self.wfn)
            
            
        self.J=self.J/self.volume
        time=np.arange(steps)*dt
        freq = np.fft.fftfreq(steps, d=dt)
        freq=np.sort(freq)
        sigma=np.zeros((3,3,freq.size),dtype=complex)
        for i in range(3):
            sigma_=np.zeros(freq.size,dtype=complex)
            for w in range(freq.size):
                sigma_[w]=np.trapz(self.J[:,i]*np.exp(1j*freq[w]*time)*smooth(time,steps,dt),time)
            for j in range(3):
                if A0[j]!=0:
                    sigma[i,j]=-sigma_/A0[j]
        epsilon=1+4*np.pi*1j*sigma/freq[None,None,:]
        return epsilon,freq
        
        
        
#     def propagate(self,dt,steps,A):
#         self.wfn=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
#         for q in range(self.NK):
#             self.wfn[q]=np.eye(self.nbands)        
#         self.J=np.zeros((steps,3),dtype=np.complex)
#         I=np.eye(self.nbands)
#         for t in tqdm(range(steps)):
#             self.J[t]=self.current()
#             H=self.E+np.einsum('iqnm,i->qnm',self.momentum,A(t*dt))+I*np.linalg.norm(A(t*dt))**2
#             for q in range(self.NK):
#                 H_left = I+0.5j*dt*H[q]            
#                 H_right= I-0.5j*dt*H[q]
#                 self.wfn[q]=linalg.solve(H_left, np.dot(H_right,self.wfn[q]))
                
                
                
                
                
                
                
                
                
                