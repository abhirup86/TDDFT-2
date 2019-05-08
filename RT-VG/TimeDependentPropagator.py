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


class TimeDependentPropagator():
    def __init__(self,TDH):
        self.volume=np.abs(np.linalg.det(TDH.wfs.gd.cell_cv)) 
        self.TDH=TDH
        self.nq=TDH.nq
        self.nbands=TDH.nbands
        
    def linear_response(self,dt,steps,A0=1e-5,NSCsteps=3):
        
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
            
        self.J/=self.volume
        J=self.J[:,2]
        time=np.arange(steps)*dt
        freq = np.fft.fftfreq(steps, d=dt)
        freq=np.sort(freq);freq=freq[np.abs(freq)<10]
        sigma=np.zeros(freq.size,dtype=complex)
        for w in range(freq.size):
            sigma[w]=np.trapz(J*np.exp(1j*freq[w]*time),time)
        sigma=-sigma/A0
        epsilon=1+4*np.pi*1j*sigma/freq
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
        
                                             
            
        