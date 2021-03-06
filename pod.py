"""
written by Shady, https://github.com/Shady-Ahmed
edited by Mehrdad, https://github.com/zomorodiyan
"""
import numpy as np
import os
from tools import jacobian, laplacian, poisson_fst, \
                  import_data, pod_svd, podproj_svd

# Load inputs
from inputs import lx, ly, nx, ny, nt, ns, freq, nr

#%% grid
dx = lx/nx; dy = ly/ny
x = np.linspace(0.0,lx,nx+1); y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% pod basis generation
nstart= 0; nend = nt; nstep = freq
#compute  mean field and basis functions for potential voriticity
wm,Phiw,Lw,RICw , tm,Phit,Lt,RICt  = pod_svd(nx,ny,dx,dy,nstart,nend,nstep,nr)


#%% Compute Streamfunction mean and basis functions
# from those of potential vorticity using Poisson equation

# compute psi_mean using the equation 28
tmp = wm.reshape([nx+1,ny+1])
tmp = poisson_fst(nx,ny,dx,dy,tmp)
sm = tmp.reshape([-1,])

# compute phi_psi_k, k in range(nr), using the equation 29
Phis = np.zeros([(nx+1)*(ny+1),nr])
for k in range(nr):
    print('loop 1/2 ',"{:.0f}".format(k/(nr-1)*100), '%   ', end='\r')
    tmp = np.copy(Phiw[:,k]).reshape([nx+1,ny+1])
    tmp = poisson_fst(nx,ny,dx,dy,tmp)
    Phis[:,k] = tmp.reshape([-1,])
print('')

#%% compute true modal coefficients
nstart= 0
nend = nt
nstep = freq

ns = int((nend-nstart)/nstep)

aTrue = np.zeros([ns+1,nr])
bTrue = np.zeros([ns+1,nr])

ii = 0
for i in range(nstart,nend+1,nstep):
    w,s,t = import_data(nx,ny,i)
    tmp = w.reshape([-1,])-wm
    aTrue[ii,:] = podproj_svd(tmp,Phiw)
    tmp = t.reshape([-1,])-tm
    bTrue[ii,:] = podproj_svd(tmp,Phit)
    ii = ii + 1
    print('loop 2/2 ',"{:.0f}".format((i-nstart+1)/(nend+1-nstart)*100), '%   ', end='\r')
print('')


#%% Save data
filename = './results/pod_' + str(nx) + 'x' + str(ny) + '.npz'
np.savez(filename, wm = wm, Phiw = Phiw,  sm = sm, Phis = Phis,  \
                   tm = tm, Phit = Phit, \
                   aTrue = aTrue, bTrue = bTrue,\
                   Lw = Lw, Lt = Lt)
