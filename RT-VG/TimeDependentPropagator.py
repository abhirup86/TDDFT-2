import numpy as np
from numpy.linalg import solve
from tqdm import tqdm
import numba

@numba.jit(nopython=True,parallel=True,fastmath=True)
def iteration(nq,dt,H,I,wfn,wfn_next):
    for q in numba.prange(nq):
        H_left =I+0.5j*dt*H[q]
        H_right=I-0.5j*dt*H[q]
        wfn_next[q]=solve(H_left, np.dot(H_right,wfn[q]))
    return wfn_next
def smooth(t,dt,steps):
    return 1.-3.*(t/(steps*dt))**2+2.*(t/(steps*dt))**3


class TimeDependentPropagator():
    def __init__(self,TDH):
        self.volume=np.abs(np.linalg.det(TDH.wfs.gd.cell_cv)) 
        self.TDH=TDH
        self.nq=TDH.nq
        self.nbands=TDH.nbands
        
    def linear_response(self,dt,steps,A0=[1e-5,0,0],NSCsteps=3):
        
        self.A0=A0
        self.J=np.zeros((steps,3),dtype=np.complex)
        
        I=np.eye(self.nbands)
        H=self.TDH.hamiltonian(A0)
        
        wfn=np.zeros((self.nq,self.nbands,self.nbands),dtype=np.complex)
        
        for q in range(self.nq):
            E,D=np.linalg.eigh(H[q])
            wfn[q]=D
         
        for t in tqdm(range(steps)):
            self.J[t]=self.TDH.calculate_current(wfn)
            wfn_next=np.copy(wfn)
            for i in range(NSCsteps):
                self.TDH.update_density(wfn_next)
                H_next = self.TDH.hamiltonian(0)
                H_mid = 0.5*(H + H_next)
                wfn_next=iteration(self.nq,dt,H_mid,I,wfn,wfn_next)
            wfn=np.copy(wfn_next)            
            H = np.copy(H_next)
            
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
        
    def propagate(self,A,steps,dt,NSCsteps=3):
        
        self.J=np.zeros((steps,3),dtype=complex)
        wfn=np.zeros((self.nq,self.nbands,self.nbands),dtype=complex)
        for q in range(self.nq):
            wfn[q]=np.eye(self.nbands)
            
        self.TDH.update_density(wfn)
        H = self.TDH.hamiltonian()
        I=np.eye(self.nbands)
        
        for t in tqdm(range(steps)):
            wfn_next=np.copy(wfn)
            for i in range(NSCsteps):
                
                self.TDH.update_density(wfn_next)
                H_next = self.TDH.hamiltonian(A(t*dt))
                H_mid = 0.5*(H + H_next)
                wfn_next=iteration(self.nq,dt,H_mid,I,wfn,wfn_next)
                
            wfn=np.copy(wfn_next)            
            H = np.copy(H_next)
            self.J[t]=self.TDH.calculate_current(wfn)
            
        self.wfn=wfn
        
                                             
            
        