import numba
import numpy as np
import xc
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
def Fock_matrix(matrix,V,M_conj,M,occupation,ibz_map):
    """
    perform integration for Fock matrix
    M - pair-density matrices <m|exp(1j*(q+G))|n>
    V - coloumb potential V(q+G)
    """
    nbands=matrix.shape[1];NK=V.shape[0];NKF=V.shape[1]
    for q in numba.prange(NKF):
        for m in range(nbands):
            if occupation[ibz_map[q],m]>1e-8:
                for k in range(NK):
                    for n1 in range(nbands):
                        for n2 in range(nbands):
                            matrix[k,n1,n2]+=2*np.sum(occupation[ibz_map[q],m]*V[k,q]*M_conj[m,n1,ibz_map[q],k]*M[m,n2,ibz_map[q],k])
    matrix/=NKF
    return matrix

class TDDFT(object):
    """
    Time-dependent DFT+Hartree-Fock in Kohn-Sham orbitals basis:
    
        calc: GPAW calculator (setups='sg15')
        nbands (int): number of bands in calculation
        
    """
    
    def __init__(self,calc):
        self.calc=calc # GPAW calculator object
        self.K=calc.get_ibz_k_points() # reduced Brillioun zone
        self.NK=self.K.shape[0] 
        
        self.wk=calc.get_k_point_weights() # weight of reduced Brillioun zone
    
        self.nvalence=int(calc.occupations.nvalence/2) #number of valence bands
        self.nbands=calc.get_number_of_bands()
        
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
        self.z=calc.wfs.gd.get_grid_point_coordinates()[2]-self.cell[2,2]/2.
        self.volume = np.abs(np.linalg.det(calc.wfs.gd.cell_cv)) # volume of cell
        self.norm=calc.wfs.gd.dv # 
        self.Fermi=self.calc.get_fermi_level()/Hartree #Fermi level
        
        #desriptors at q=gamma for Hartree
        self.kdH=KPointDescriptor([[0,0,0]]) 
        self.pdH=PWDescriptor(ecut=calc.wfs.pd.ecut,gd=calc.wfs.gd,kd=self.kdH,dtype=complex)
        
        #desriptors at q=gamma for Fock
        self.kdF=KPointDescriptor([[0,0,0]]) 
        self.pdF=PWDescriptor(ecut=calc.wfs.pd.ecut/4.,gd=calc.wfs.gd,kd=self.kdF,dtype=complex)
        
        #Fermi-Dirac temperature
        self.temperature=calc.occupations.width
        
        #calculate pair-density matrices
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
        self.f=1/(1+np.exp(self.EK/self.temperature))
        
    def plane_wave(self,k):
        """ 
        return plane wave defined on real space grid:
        if k is integer wave vector defined as k-th k-point of reduced Brillioun zone
        if k is array wave vector defined as np.dot(k,self.icell)
        """ 
        if type(k)==int:
            return self.calc.wfs.gd.plane_wave(self.K[k])
        return self.calc.wfs.gd.plane_wave(k)
    
    
    def get_dipole_matrix(self):
        """ 
        return two-dimensional numpy complex array of dipole matrix elements(
        """ 
        
        dipole=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        dipole=operator_matrix_periodic(dipole,self.z,self.ukn.conj(),self.ukn)*self.norm
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
        return 4*np.pi*self.pdH.ifft(nG/G2)
    
    def get_coloumb_potential(self,q):
        """
        return coloumb potential in plane wave space V= 4 pi /(|q+G|**2)
        q: [qx,qy,qz] vector in units of reciprocal space
        """
        G=self.pdF.get_reciprocal_vectors()+np.dot(q,self.icell)
        G2=np.linalg.norm(G,axis=1)**2;G2[G2==0]=np.inf
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
        if wavefunction is None:
            wavefunction=np.zeros((self.NK,self.nbands,self.nbands))
            for k in range(self.NK):
                wavefunction[k]=np.eye(self.nbands)
        K=self.calc.get_bz_k_points();NK=K.shape[0]  
        NG=self.pdF.get_reciprocal_vectors().shape[0]
        V=np.zeros((self.NK,NK,NG))
        for k in range(self.NK):
            for q in range(NK):
                kq=K[q]-self.K[k]
                kq[kq>0.5]-=1;kq[kq<=-0.5]+=1
                V[k,q]=self.get_coloumb_potential(kq)
        VF_matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)        
        VF_matrix=Fock_matrix(VF_matrix,V,self.M.conj(),self.M,
                              np.sum(np.abs(wavefunction)**2,axis=2)*self.f,
                              self.calc.get_bz_to_ibz_map())
        return VF_matrix
    
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
    
    