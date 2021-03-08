import numpy as np  
import math 
from numpy import linalg as LA
import sys
#----------------------------------------
# Matrix Diagonalization

def Diag(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#----------------------------------------

# Potential
def Ve(param):
    L0 = param.N * param.d/2  # -L0 to +L0
    re = param.re 

    Vm = param.v *  (1+ np.cos(2.0 * np.pi  * re/ param.d)) * (re >= -L0) * (re <= L0)
    Vm += 2 * param.v * ((re < -L0) + (re > L0))

    return Vm  

def VHO(param):
    re = param.re 
    wc = 1.0
    return 0.5 * wc**2.0 * re **2.0 


# Kinetic energy for electron 
def Te(param):
 re = param.re
 N = float(len(re))
 mass = 1.0
 Tij = np.zeros((int(N),int(N)))
 Rmin = float(re[0])
 Rmax = float(re[-1])
 step = float((Rmax-Rmin)/N)
 K = np.pi/step

 for ri in range(int(N)):
  for rj in range(int(N)):
    if ri == rj:  
     Tij[ri,ri] = (0.5/mass)*K**2.0/3.0*(1+(2.0/N**2)) 
    else:    
     Tij[ri,rj] = (0.5/mass)*(2*K**2.0/(N**2.0))*((-1)**(rj-ri)/(np.sin(np.pi*(rj-ri)/N)**2)) 
 return Tij
#---------------------------------





def Hel(param):
    N = len(param.re)
    V = np.zeros((N,N))
    np.fill_diagonal( V , Ve(param) ) 
    T = Te(param) 
    He = T + V 
    E, Ψ = Diag(He)
    return  E[:param.states], Ψ[:param.states,:], µ(Ψ, param)



 
def µ(Ψ, param):

    R = np.zeros((param.N))
    R[len(R)/2:] = (2 * np.arange(0,len(R)/2) + 1) * ( param.d/ 2.0 )
    R[:len(R)/2] = - R[len(R)/2:][::-1]
    print (R)
    z = np.ones((param.N))/len(R)

    zR = z * R

    dm = np.zeros((param.states,param.states)) 
    for i in range(param.states):
        for j in range(i, param.states):
            dm[i,j] = - np.sum(param.re  * Ψ[:,i]  * Ψ[:,j] ) 
            dm[i,j] += zR * (i==j)
            dm[j,i] =  dm[i,j]
    return dm 


au = 0.529177249 # A to a.u.
# Parameters 2b
class param:
    states = 20 # number of states to print out
    N  = 20
    v  = 3.0 # potential depth
    d  = 2.0 # distance | Lattice Constant
    padding = d
    re = np.linspace(-N*d/2 - padding,  N*d/2 + padding, 1500) # electronic grid



if __name__=="__main__":
    par = param()
    #E, V =  Hel(R,param)
    np.savetxt("Ve.txt", np.c_[param.re,Ve(param)])

    E, Ψ, µij =  Hel(param)

    np.savetxt("E.txt", E)
    np.savetxt("Ψ.txt", Ψ)
    np.savetxt("µ.txt", µij)
