import numpy as np
from gpaw.wavefunctions.pw import PWLFC
from gpaw.utilities import unpack
from gpaw.transformers import Transformer
from itertools import product
from fast_functions import *

class TimeDependentHamiltonian():
    
    def __init__(self,calc):
        
        #initialization 
        self.calc=calc
        self.wfs=calc.wfs
        self.ham=calc.hamiltonian
        self.den=calc.density
        self.occ=calc.occupations
        
        #initialization plane and grid descriptors from GPAW calculation
        self.pd=calc.wfs.pd
        self.gd=calc.wfs.gd
        self.volume=np.abs(np.linalg.det(self.gd.cell_cv)) 
        
        #number of k-points
        self.nq=len(calc.wfs.kpt_u)
        #number of bands
        self.nbands=calc.get_number_of_bands()
        #number of electrons
        self.nelectrons=calc.get_number_of_electrons()
        
        #kinetic operator
        self.kinetic=np.zeros((self.nq,self.nbands,self.nbands),dtype=complex)
        #overlap operator
        self.overlap=np.zeros((self.nq,self.nbands,self.nbands),dtype=complex)
        #local momentum operator
        self.local_moment=np.zeros((3,self.nq,self.nbands,self.nbands),dtype=complex)
        #nonlocal momentum operator
        self.nonlocal_moment=np.zeros((3,self.nq,self.nbands,self.nbands),dtype=complex)
        #Fermi-Dirac occupation
        self.f_n=np.zeros((self.nq,self.nbands),dtype=float)
        
        #ground state Kohn-Sham orbitals wavefunctions
        psi_gs=[]
        #ground state Kohn-Sham orbitals density
        den_gs=[]
                
        for kpt in self.wfs.kpt_u:
            self.overlap[kpt.q]=np.eye(self.nbands)
            self.f_n[kpt.q]=kpt.f_n
            kinetic=0.5 * self.pd.G2_qG[kpt.q] # |G+q|^2/2
            gradient=self.pd.get_reciprocal_vectors(kpt.q)
            psi=[];den=[]
            for n in range(self.nbands):
                psi.append(self.pd.ifft(kpt.psit_nG[n],kpt.q))
                den.append(np.abs(psi[-1])**2)
                for m in range(self.nbands):
                    self.kinetic[kpt.q,n,m]=self.pd.integrate(kpt.psit_nG[n],kinetic*kpt.psit_nG[m])
                    #calculation local momentum
                    #<psi_qn|\nabla|psi_qm>
                    for i in range(3):
                        self.local_moment[i,kpt.q,n,m]=self.pd.integrate(kpt.psit_nG[n],gradient[:,i]*kpt.psit_nG[m])
            psi_gs.append(psi)
            den_gs.append(den)    
            
        self.psi_gs=np.array(psi_gs)
        self.den_gs=np.array(den_gs,dtype=float)
        
        #real space grid points 
        self.r=self.gd.get_grid_point_coordinates()
        
        #initialization local and nonlocal part of pseudopotential
        self.init_potential()
        
        self.proj=np.zeros((self.nq,self.nbands,self.norb),dtype=complex)
        self.proj_r=np.zeros((3,self.nq,self.nbands,self.norb),dtype=complex)
        
        self.density=self.den.nt_sG.copy()
        
        #initialization charge density (ion+electrons) for Hartree potential
        self.ion_density=calc.hamiltonian.poisson.pd.ifft(calc.density.rhot_q)-calc.density.nt_sg[0]
        #plane wave descriptor for Hartree potential
        self.pd0=self.ham.poisson.pd
        #reciprocal |G|^2 vectors for Hartree potential V(G)=4pi/|G|^2
        self.G2=self.ham.poisson.G2_q
        self.G=self.pd0.get_reciprocal_vectors()
        
        #fine to coarse and coarse to fine grids transformers (for correct calculation local potential)
        self.fine_to_coarse=Transformer(calc.density.finegd, calc.density.gd, 3) 
        self.coarse_to_fine=Transformer(calc.density.gd, calc.density.finegd, 3)
        
        #initialization local potenital from ground state density
        self.update_local_potential()
        self.update_gauge([0,0,0])
#----------------------------------------------------------------------------------------------------          
    def init_potential(self):
        #initialization of nonlocal part of pseudopotential
        # V_NL=|chi_i> V_i <chi_i|
        spline_aj = []
        for setup in self.wfs.setups:
            spline_aj.append(setup.pt_j)
        self.lfc=PWLFC(spline_aj,self.pd)
        self.lfc.set_positions(self.calc.spos_ac)  #set position of atoms
        proj_G=[];proj_r=[] #collect chi_i in real space using FFT
        
        for kpt in self.wfs.kpt_u:
            proj_G.append(self.lfc.expand(kpt.q))
            proj=[];n_i=proj_G[-1].shape[1]
            for i in range(n_i):
                proj.append(self.pd.ifft(proj_G[-1][:,i].copy(),kpt.q))
            proj_r.append(proj)
        self.chi=np.array(proj_r)/self.gd.dv
        
        s=0 # s=0 because we perform spin-paired calculation
        V=[] #collect V 
        for a in range(len(self.wfs.setups)):
            dH_ii = unpack(self.ham.dH_asp[a][s])
            V.append(dH_ii.diagonal())
            
        V=np.array(V);
        self.V=V.ravel()
        
        self.norb=self.V.size # number of orbitals in nonlocal potential
        
        #initialization of local part of pseudopotential
        V=self.ham.vbar.pd.zeros()
        self.ham.vbar.add(V)
        self.Vloc=self.ham.vbar.pd.ifft(V)
        
#----------------------------------------------------------------------------------------------------          
    def update_density(self,wfn):
        self.density[0]=np.zeros_like(self.density[0])
        fast_density(self.density[0],self.f_n,wfn,self.den_gs)
        self.occupation=np.einsum('qnm,qn->qm',np.abs(wfn)**2,self.f_n)                
        
#---------------------------------------------------------------------------------------------------- 
    def update_gauge(self,A):
        #update projections p_qno=<chi_qo|psi_qn>
        phase=np.exp(1j*np.einsum('ixyz,i->xyz',self.r,A))
        fast_projections(self.proj,self.chi,phase,self.psi_gs,self.gd.dv)
        
        #update interaction hamiltonian
        #H_I=\nabla*A+0.5*A^2
        self.interaction=np.einsum('i,iqnm->qnm',A,self.local_moment)+0.5*np.linalg.norm(A)**2*self.overlap
        
        #calculation nonlocal momentum operator
        # I_NL=[r,V_NL]=sum_o V_o (r|chi_o><chi_o - |chi_o><chi_o|r)
        for i in range(3):
            fast_projections(self.proj_r[0],self.chi,self.r[i]*phase,self.psi_gs,self.gd.dv)
        self.nonlocal_moment =-1j*(np.einsum('o,iqno,qmo->iqnm',self.V,self.proj_r.conj(),self.proj)-
                                   np.einsum('o,qno,iqmo->iqnm',self.V,self.proj.conj(),self.proj_r))
        self.moment=self.local_moment+self.nonlocal_moment
        
#----------------------------------------------------------------------------------------------------          
    def update_local_potential(self):
        
        #tranform density from coarse to fine grids
        density=self.coarse_to_fine.apply(self.density.copy())
        
        #calculate XC potential
        VXC=np.zeros_like(density)
        self.ham.xc.calculate(self.pd0.gd,density,VXC)
          
#         calculate Hartree potential
        charge_density=density+self.ion_density
        VH=4*np.pi*self.pd0.fft(charge_density)/self.G2
        VH=self.pd0.ifft(VH)
        
        #transform Hartree and XC potential from fine to coarse grids
        self.VXC=self.fine_to_coarse.apply(VXC[0])
        self.VH=self.fine_to_coarse.apply(VH)
        
#----------------------------------------------------------------------------------------------------            
    def calculate_nonlocal(self):      
        #calculation nonlocal part of Hamiltonian
        return np.einsum('o,qno,qmo->qnm',self.V,self.proj.conj(),self.proj)
    
#----------------------------------------------------------------------------------------------------      
    def calculate_local(self):
        #calculation local part of Hamiltonian
        local=np.zeros((self.nq,self.nbands,self.nbands),dtype=complex)
        return fast_local(local,self.Vloc+self.VXC+self.VH,self.psi_gs,self.gd.dv)
    
#----------------------------------------------------------------------------------------------------      
    def calculate_kinetic(self):
        #calculation kinetic part of Hamiltonian
        return self.kinetic
    
#----------------------------------------------------------------------------------------------------      
    def hamiltonian(self):
        #calculation hamiltonian
        return self.calculate_kinetic()+self.calculate_local()+self.calculate_nonlocal()+self.interaction
    
#----------------------------------------------------------------------------------------------------    
    def calculate_current(self,wfn):
        return np.einsum('iqnm,qb,qnb,qmb->i',self.moment,self.f_n,wfn.conj(),wfn)/self.volume
        
        
       
        
    
    
        
        
        
        