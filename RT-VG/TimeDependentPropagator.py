import numpy as np
from scipy.linalg import solve
from tqdm import tqdm

class TimeDependentPropagator():
    def __init__(self,TDH):
        self.TDH=TDH
        self.nq=TDH.nq
        self.nbands=TDH.nbands
        
        
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
                for q in range(self.nq):
                    Hq_left = I+0.5j*dt*H_mid[q]
                    Hq_right =I-0.5j*dt*H_mid[q]
                    wfn_next[q]=solve(Hq_left, np.dot(Hq_right,wfn[q]))
                    
            wfn=np.copy(wfn_next)            
            H = np.copy(H_next)
            self.J[t]=self.TDH.calculate_current(wfn)
        self.wfn=wfn
                                             
            
        