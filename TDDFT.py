import numba
import numpy as np
import xc
from tqdm import tqdm
from scipy import linalg
from ase.units import Hartree, Bohr
from itertools import product
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor

@numba.jit(nopython=True,parallel=True,fastmath=True)
def operator_matrix_periodic(matrix,operator,wf_conj,wf):
    """perform integration of periodic part of Kohn Sham wavefunction"""
    NK=matrix.shape[0]
    nbands=matrix.shape[1]
    for k in numba.prange(NK):
        for n1 in range(nbands):
            for n2 in range(nbands):
                matrix[k,n1,n2]=np.sum(operator*wf_conj[k,n1]*wf[k,n2])
    return matrix

@numba.jit(nopython=True,parallel=True,fastmath=True)
def Fock_matrix(matrix,V,M_conj,M,ibz_map,occ_bands):
    """
    perform integration for Fock matrix
    M - pair-density matrices <m|exp(1j*(q+G))|n>
    V - coloumb potential V(q+G)
    """
    nbands=matrix.shape[1];NK=V.shape[0];NKF=V.shape[1]
    for q in numba.prange(NKF):
        for m in range(occ_bands):
                for k in range(NK):
                    for n1 in range(nbands):
                        for n2 in range(nbands):
                            matrix[k,n1,n2]+=2*np.sum(V[k,q]*M_conj[m,n1,ibz_map[q],k]*M[m,n2,ibz_map[q],k])
    matrix/=NKF
    return matrix

class TDDFT(object):
    """
    Time-dependent DFT+Hartree-Fock in Kohn-Sham orbitals basis:
    
        calc: GPAW calculator (setups='sg15')
        nbands (int): number of bands in calculation
        
    """
    
    def __init__(self,calc,nbands=None,Fock=False):
        self.Fock=Fock
        self.K=calc.get_ibz_k_points() # reduced Brillioun zone
        self.NK=self.K.shape[0] 
        
        self.wk=calc.get_k_point_weights() # weight of reduced Brillioun zone
        if nbands is None:
            self.nbands=calc.get_number_of_bands()
        else:
            self.nbands=nbands
        self.nvalence=int(calc.get_number_of_electrons()/2)
        
        self.EK=[calc.get_eigenvalues(k)[:self.nbands] for k in range(self.NK)] # bands energy
        self.EK=np.array(self.EK)/Hartree
        self.shape=tuple(calc.get_number_of_grid_points()) # shape of real space grid
        self.density=calc.get_pseudo_density()*Bohr**3 # density at zero time
        
        
        # array of u_nk (periodic part of Kohn-Sham orbitals,only reduced Brillion zone)
        self.ukn=np.zeros((self.NK,self.nbands,)+self.shape,dtype=np.complex) 
        for k in range(self.NK):
            kpt = calc.wfs.kpt_u[k]
            for n in range(self.nbands):
                psit_G = kpt.psit_nG[n]
                psit_R = calc.wfs.pd.ifft(psit_G, kpt.q)
                self.ukn[k,n]=psit_R 
                
        self.icell=2.0 * np.pi * calc.wfs.gd.icell_cv # inverse cell 
        self.cell = calc.wfs.gd.cell_cv # cell
        self.r=calc.wfs.gd.get_grid_point_coordinates()
        self.r[2]-=self.cell[2,2]/2.
        self.volume = np.abs(np.linalg.det(calc.wfs.gd.cell_cv)) # volume of cell
        self.norm=calc.wfs.gd.dv # 
        self.Fermi=calc.get_fermi_level()/Hartree #Fermi level
        
        #desriptors at q=gamma for Hartree
        self.kdH=KPointDescriptor([[0,0,0]]) 
        self.pdH=PWDescriptor(ecut=calc.wfs.pd.ecut,gd=calc.wfs.gd,kd=self.kdH,dtype=complex)
        
        #desriptors at q=gamma for Fock
        self.kdF=KPointDescriptor([[0,0,0]]) 
        self.pdF=PWDescriptor(ecut=calc.wfs.pd.ecut/4.,gd=calc.wfs.gd,kd=self.kdF,dtype=complex)
        
        #Fermi-Dirac temperature
        self.temperature=calc.occupations.width
        
        #calculate pair-density matrices
        if Fock:
            self.M=np.zeros((self.nbands,self.nbands,
                             self.NK,self.NK,
                             self.pdF.get_reciprocal_vectors().shape[0]),dtype=np.complex)
            indexes=[(n,k) for n,k in product(range(self.nbands),range(self.NK))]
            for i1 in range(len(indexes)):
                n1,k1=indexes[i1]
                for i2 in range(i1,len(indexes)):
                    n2,k2=indexes[i1]
                    self.M[n1,n2,k1,k2]=self.pdF.fft(self.ukn[k1,n1].conj()*self.ukn[k2,n2])
                    self.M[n2,n1,k2,k1]=self.M[n1,n2,k1,k2].conj()
            self.M*=calc.wfs.gd.dv
        
        #Fermi-Dirac distribution
        self.f=1/(1+np.exp((self.EK-self.Fermi)/self.temperature))
        
        self.Hartree_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAx_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAc_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        G=self.pdH.get_reciprocal_vectors()
        G2=np.linalg.norm(G,axis=1)**2;G2[G2==0]=np.inf
        matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        for k in tqdm(range(self.NK)):
            for n in range(self.nbands):
                density=2*np.abs(self.ukn[k,n])**2
                operator=xc.VLDAx(density)
                self.LDAx_elements[k,n]=operator_matrix_periodic(matrix,operator,self.ukn.conj(),self.ukn)*self.norm
                operator=xc.VLDAc(density)
                self.LDAc_elements[k,n]=operator_matrix_periodic(matrix,operator,self.ukn.conj(),self.ukn)*self.norm
                
                density=self.pdH.fft(density)
                operator=4*np.pi*self.pdH.ifft(density/G2)  
                self.Hartree_elements[k,n]=operator_matrix_periodic(matrix,operator,self.ukn.conj(),self.ukn)*self.norm
        
        self.wavefunction=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        self.Kinetic=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        self.dipole=self.get_dipole_matrix()
        for k in range(self.NK):
            self.wavefunction[k]=np.eye(self.nbands)
            self.Kinetic[k]=np.diag(self.EK[k])
        self.VH0=self.fast_Hartree_matrix(self.wavefunction)
        self.VLDAc0=self.fast_LDA_correlation_matrix(self.wavefunction)
        self.VLDAx0=self.fast_LDA_exchange_matrix(self.wavefunction)
        
        self.Full_BZ=calc.get_bz_k_points()
        self.IBZ_map=calc.get_bz_to_ibz_map()
    
    
    def get_dipole_matrix(self,direction=[0,0,1]):
        """ 
        return two-dimensional numpy complex array of dipole matrix elements(
        """ 
        direction/=np.linalg.norm(direction)
        dipole=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        r=np.array(sum([self.r[i]*direction[i] for i in range(3)]))
        dipole=operator_matrix_periodic(dipole,r,self.ukn.conj(),self.ukn)*self.norm
        return dipole
    
    def get_density(self,wavefunction):
        """ 
        return numpy array of electron density in real space at each k-point of full Brillioun zone
        wavefunction: numpy array [N_kpoint X N_band X N_band] of wavefunction in basis of Kohn-Sham orbital
        """ 
        if wavefunction is None:
            return self.density
        
        density=np.zeros(self.shape,dtype=np.float)
        for k in range(self.NK):
            for n in range(self.nbands):
                for m in range(self.nbands):
                    density+=2*self.wk[k]*self.f[k,n]*np.abs(wavefunction[k,m,n]*self.ukn[k,m])**2
        return density
    
    def get_Hartree_potential(self,wavefunction):
        """ 
        return numpy array of Hartree potential in real space at each k-point of full Brillioun zone
        wavefunction: numpy array [N_kpoint X N_band X N_band] of wavefunction in basis of Kohn-Sham orbital
        """ 
        density=self.get_density(wavefunction)
        VH=np.zeros(self.shape)
        G=self.pdH.get_reciprocal_vectors()
        G2=np.linalg.norm(G,axis=1)**2;G2[G2==0]=np.inf
        nG=self.pdH.fft(density)
        return -4*np.pi*self.pdH.ifft(nG/G2)
    
    def get_coloumb_potential(self,q):
        """
        return coloumb potential in plane wave space V= 4 pi /(|q+G|**2)
        q: [qx,qy,qz] vector in units of reciprocal space
        """
        G=self.pdF.get_reciprocal_vectors()+np.dot(q,self.icell)
        G2=np.linalg.norm(G,axis=1)**2;
        G2[G2==0]=np.inf
        return 4*np.pi/G2  
    
    def get_Hartree_matrix(self,wavefunction=None):
        """
        return numpy array [N_kpoint X N_band X N_band] of Hartree potential matrix elements
        wavefunction: numpy array [N_kpoint X N_band X N_band] of wavefunction in basis of Kohn-Sham orbital
        """
        VH=self.get_Hartree_potential(wavefunction)
        VH_matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        VH_matrix=operator_matrix_periodic(VH_matrix,VH,self.ukn.conj(),self.ukn)*self.norm
        return VH_matrix
    
    def get_Fock_matrix(self,wavefunction=None):
        """
        return numpy array [N_kpoint X N_band X N_band] of Fock potential matrix elements
        wavefunction: numpy array [N_kpoint X N_band X N_band] of wavefunction in basis of Kohn-Sham orbital
        """
        VF_matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)     
        if self.Fock:
            if wavefunction is None:
                wavefunction=np.zeros((self.NK,self.nbands,self.nbands))
                for k in range(self.NK):
                    wavefunction[k]=np.eye(self.nbands)
            K=self.Full_BZ;NK=K.shape[0]  
            NG=self.pdF.get_reciprocal_vectors().shape[0]
            V=np.zeros((self.NK,NK,NG))
            for k in range(self.NK):
                for q in range(NK):
                    kq=K[q]-self.K[k]
                    V[k,q]=self.get_coloumb_potential(kq)

            VF_matrix=Fock_matrix(VF_matrix,V,self.M.conj(),self.M,
                                  self.IBZ_map,self.nvalence)
        return VF_matrix/self.volume
    
    def get_LDA_exchange_matrix(self,wavefunction=None):
        """
        return numpy array [N_kpoint X N_band X N_band] of LDA exchange potential matrix elements
        wavefunction: numpy array [N_kpoint X N_band X N_band] of wavefunction in basis of Kohn-Sham orbital
        """
        density=self.get_density(wavefunction)
        exchange=xc.VLDAx(density)
        LDAx_matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        LDAx_matrix=operator_matrix_periodic(LDAx_matrix,exchange,self.ukn.conj(),self.ukn)*self.norm
        return LDAx_matrix
    
    def get_LDA_correlation_matrix(self,wavefunction=None):
        """
        return numpy array [N_kpoint X N_band X N_band] of LDA correlation potential matrix elements
        wavefunction: numpy array [N_kpoint X N_band X N_band] of wavefunction in basis of Kohn-Sham orbital
        """
        density=self.get_density(wavefunction)
        correlation=xc.VLDAc(density)
        LDAc_matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        LDAc_matrix=operator_matrix_periodic(LDAc_matrix,correlation,self.ukn.conj(),self.ukn)*self.norm
        return LDAc_matrix
    
    def occupation(self,wavefunction):
        return 2*self.wk[:,None]*np.sum(self.f[:,None,:]*np.abs(wavefunction)**2,axis=2)
    
    def fast_Hartree_matrix(self,wavefunction):
        return np.einsum('kn,knqij->qij',self.occupation(wavefunction),self.Hartree_elements)
    
    def fast_LDA_correlation_matrix(self,wavefunction):
        return np.einsum('kn,knqij->qij',self.occupation(wavefunction),self.LDAc_elements)
    
    def fast_LDA_exchange_matrix(self,wavefunction):
        return np.einsum('kn,knqij->qij',self.occupation(wavefunction),self.LDAx_elements)
    
    def propagate(self,dt,steps,E,operator,corrections=10):
        self.wavefunction=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        for k in range(self.NK):
            self.wavefunction[k]=np.eye(self.nbands)
        H=np.copy(self.Kinetic)
        operator_macro=np.array([operator[k].diagonal() for k in range(self.NK)])
        result=np.zeros((steps,self.nbands),dtype=np.complex)
        self.macro_dipole=np.zeros(steps,dtype=np.complex)
        for t in tqdm(range(steps)):
            wavefunction_next=np.copy(self.wavefunction)
            for i in range(corrections):
                H_next =self.Kinetic+E[t]*self.dipole
                H_next+=self.fast_Hartree_matrix(wavefunction_next)-self.VH0
                H_next+=self.fast_LDA_correlation_matrix(wavefunction_next)-self.VLDAc0
                H_next+=self.fast_LDA_exchange_matrix(wavefunction_next)-self.VLDAx0
                
                H_mid = 0.5*(H + H_next) 
                for k in range(self.NK):
                    H_left = np.eye(self.nbands)+0.5j*dt*H_mid[k]            
                    H_right= np.eye(self.nbands)-0.5j*dt*H_mid[k]
                    wavefunction_next[k]=linalg.solve(H_left, H_right@wavefunction_next[k])  
                self.wavefunction=np.copy(wavefunction_next)      
                H = np.copy(H_next)
            result[t]=np.sum(self.occupation(self.wavefunction),axis=0)
            self.macro_dipole[t]=np.sum(self.occupation(self.wavefunction)*operator_macro)
        return result
