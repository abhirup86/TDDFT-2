import numpy as np
import sympy # module for symbolic calculus

n=sympy.symbols('n') # symbolic variable for number density

### Exchange energy per particle
# Carsten A. Ullrich and Zeng-hui Yang, 
# A Brief Compendium of Time-Dependent Density Functional Theory 
# Braz J Phys (2014) 44:154–188, Eq.(34)
axfactor=-(3/4.)*(3/np.pi)**(1/3.)
def ex(n):
    return axfactor*n**(1/3.) # per particle

### Correlation: Chachiyo-Karasiev parametrization
acfactor=(np.log(2.)-1.)/(2.*np.pi**2)
n2invrs=(4.*np.pi/3)**(1/3.)
def ec(n):
    invrs = n2invrs*n**(1/3.) 
    return acfactor*sympy.log( 1. + 21.7392245*invrs + 20.4562557*invrs**2 )

def exc(n):
    return ex(n)+ec(n)

np_exc=sympy.lambdify(n, exc(n), 'numpy')

### Check: calculate xc energy for a range of densities
# np_exc(np.array([0.1,0.2,0.3,0.4,0.5]))

### Below is the result for exc using the XC_LDA_XC_TETER93 functional from the libxc library:
# 0.100000 -0.395669
# 0.200000 -0.489903
# 0.300000 -0.555547
# 0.400000 -0.607600
# 0.500000 -0.651438

sym_VLDA=sympy.diff(n*exc(n),n)
VLDA=sympy.lambdify(n, sym_VLDA, 'numpy') # returns numpy values

### Check: calculate xc energy for a range of densities
# VLDA(np.array([0.1,0.2,0.3,0.4,0.5]))

### Below is the result for vxc using the XC_LDA_XC_TETER93 functional from the libxc library:
# 0.100000 -0.517133
# 0.200000 -0.641488
# 0.300000 -0.728231
# 0.400000 -0.797062
# 0.500000 -0.855057

sym_VLDAx=sympy.diff(n*ex(n),n)
VLDAx=sympy.lambdify(n, sym_VLDAx, 'numpy') # returns numpy values

sym_VLDAc=sympy.diff(n*ec(n),n)
VLDAc=sympy.lambdify(n, sym_VLDAc, 'numpy') # returns numpy values