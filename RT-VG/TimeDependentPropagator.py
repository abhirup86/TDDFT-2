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
        self.TDH=TDH
        self.nq=TDH.nq
        self.nbands=TDH.nbands
        self.ne=int(TDH.calc.get_number_of_electrons()/2)
        
    def linear_response(self,dt,steps,A0=[1e-5,0,0],NSCsteps=3,NSC=10):
        
        self.J=np.zeros((steps,3),dtype=np.complex)
        self.occ=np.zeros((steps,self.nq,self.nbands))
        I=np.eye(self.nbands)

        #calculation start wavefunctions
        wfn=np.zeros((self.nq,self.nbands,self.ne),dtype=np.complex)
        
        self.TDH.update_gauge(A0)
        for i in range(NSC):
            H=self.TDH.hamiltonian()
            for q in range(self.nq):
                E,D=np.linalg.eigh(H[q])
                wfn[q]=D[:,:self.ne]
            self.TDH.update_density(wfn)
   
        self.TDH.update_gauge([0,0,0])
        H=self.TDH.hamiltonian()
        
        #time propagation
        for t in tqdm(range(steps)):
            
            self.J[t]=self.TDH.calculate_current(wfn)
            self.occ[t]=self.TDH.occupation
            
            wfn_next=np.copy(wfn)
            for i in range(NSCsteps):
                self.TDH.update_density(wfn_next)
                H_next = self.TDH.hamiltonian()
                H_mid = 0.5*(H + H_next)
                wfn_next=iteration(self.nq,dt,H_mid,I,wfn,wfn_next)
            wfn=np.copy(wfn_next)            
            H = np.copy(H_next)
        
        #linear response calculation
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
        #time-dependent propagation
        # A - function A(t)
        # steps - number of time steps
        # dt - step in a.u. time
        # NSCsteps - number of corrections
        
        self.J=np.zeros((steps,3),dtype=complex)
        wfn=np.zeros((self.nq,self.nbands,self.nbands),dtype=complex)
        for q in range(self.nq): wfn[q]=np.eye(self.nbands)
            
        self.TDH.update_density(wfn)
        H = self.TDH.hamiltonian()
        I = np.eye(self.nbands)
        
        for t in tqdm(range(steps)):
            self.J[t]=self.TDH.calculate_current(wfn)
            self.TDH.update_gauge(A(t*dt))
            wfn_next=np.copy(wfn)
            for i in range(NSCsteps):
                self.TDH.update_density(wfn_next)
                H_next = self.TDH.hamiltonian()
                H_mid = 0.5*(H + H_next)
                wfn_next=iteration(self.nq,dt,H_mid,I,wfn,wfn_next)
            wfn=np.copy(wfn_next)            
            H = np.copy(H_next)
        
                                             
            
        