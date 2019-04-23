import numpy as np
from gpaw.wavefunctions.pw import PWLFC
from gpaw.utilities import unpack
from gpaw.transformers import Transformer
import pylibxc

class TimeDependentHamiltonian():
    
    def __init__(self,calc,direction=[0,0,1]):
        
        #initialization 
        self.calc=calc
        self.wfs=calc.wfs
        self.ham=calc.hamiltonian
        self.den=calc.density
        self.occ=calc.occupations
        
        #initialization plane and grid descriptors from GPAW calculation
        self.pd=calc.wfs.pd
        self.gd=calc.wfs.gd
        
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
        #momentum operator
        self.momentum=np.zeros((3,self.nq,self.nbands,self.nbands),dtype=complex)
        #dipole operator
        self.dipole=np.zeros((3,self.nq,self.nbands,self.nbands),dtype=complex)
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
                    
                    
                    for i in range(3):
                        self.momentum[i,kpt.q,n,m]=self.pd.integrate(kpt.psit_nG[n],gradient[:,i]*kpt.psit_nG[m])
                        if n!=m:
                            self.dipole[i,kpt.q,n,m]=self.momentum[i,kpt.q,n,m]/(kpt.eps_n[m]-kpt.eps_n[n])
            psi_gs.append(psi)
            den_gs.append(den)
         
        self.psi_gs=np.array(psi_gs)
        self.den_gs=np.array(den_gs,dtype=float)
        
        #scalar product with vector potential direction
        self.interaction_momentum=np.einsum('iqnm,i->qnm',self.momentum,direction)
        
        #real space grid points 
        self.r=np.einsum('ixyz,i->xyz',self.gd.get_grid_point_coordinates(),direction)
        
        #initialization local and nonlocal part of pseudopotential
        self.init_potential()
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
        
        
    
    def update_local_potential(self):
        
        #tranform density from coarse to fine grids
        density=self.coarse_to_fine.apply(self.density.copy())
        
        #calculate XC potential
        VXC=np.zeros_like(density)
        self.ham.xc.calculate(self.pd0.gd,density,VXC)
          
        #calculate Hartree potential
        charge_density=density+self.ion_density
        VH=4*np.pi*self.pd0.fft(charge_density)/self.G2
        VH=self.pd0.ifft(VH)
        
        #transform Hartree and XC potential from fine to coarse grids
        self.VXC=self.fine_to_coarse.apply(VXC[0])
        self.VH=self.fine_to_coarse.apply(VH)
            
        
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
        
    def update_density(self,wfn):
        self.density[0]=np.einsum('qn,qmn,qmxyz->xyz',self.f_n,np.abs(wfn)**2,self.den_gs)
    
    def calculate_nonlocal(self,A):
        #calculation nonlocal part of Hamiltonian
        #with gauge transform according to vector potential A
        phase=np.exp(1j*self.r*A)
        proj=np.zeros((self.nq,self.nbands,self.norb),dtype=complex)
        for q in range(self.nq):
            for n in range(self.nbands):
                for o in range(self.norb):
                    proj[q,n,o]=self.gd.integrate(self.chi[q,o],self.psi_gs[q,n]*phase)
                    
        return np.einsum('i,qni,qmi->qnm',self.V,proj.conj(),proj)
    
    def calculate_local(self):
        #calculation local part of Hamiltonian
        VL=np.einsum('xyz,qnxyz,qmxyz->qnm',self.Vloc+self.VXC+self.VH,self.psi_gs.conj(),self.psi_gs)*self.gd.dv
        return VL
    
    def calculate_kinetic(self):
        #calculation kinetic part of Hamiltonian
        return self.kinetic
    
    def calculate_interaction(self,A):
        #calculation interaction with field part
        return A*self.interaction_momentum+0.5*A**2*self.overlap
    
    def hamiltonian(self,A=0):
        #Hamiltonian calculation 
        K=self.calculate_kinetic()
        VL=self.calculate_local()
        VNL=self.calculate_nonlocal(A)
        I=self.calculate_interaction(A)
        return K+VL+VNL+I
    
    def calculate_current(self,A,wfn):
#         VNL=self.calculate_nonlocal(A)
#         J=np.zeros((3,self.nq,self.nbands,self.nbands),dtype=complex)
#         for i in range(3):
#             for q in range(self.nq):
#                 J[i,q]=np.dot(VNL[q],self.dipole[i,q])-np.dot(self.dipole[i,q],VNL[q])
        J=self.momentum
        current=np.einsum('iqnm,qb,qnb,qmb->i',J,self.f_n,wfn.conj(),wfn)
#         current+=self.nelectrons*A
        return current
        
        
       
        
    
    
        
        
        
        