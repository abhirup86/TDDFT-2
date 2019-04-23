import numpy as np
from scipy.linalg import solve
from tqdm import tqdm

class TimeDependentPropagator():
    def __init__(self,Hamiltonian):
        self.TDH=Hamiltonian
        self.wfs=self.TDH.wfs
        self.nq=self.TDH.nq
        self.nbands=self.TDH.nbands
        
    def propagate(self,A,steps,dt):
        pass
                
            
        