"""
written by Shady, https://github.com/Shady-Ahmed
edited by Mehrdad, https://github.com/zomorodiyan
"""
import numpy as np
import os
from tools import jacobian, laplacian, poisson_fst, \
                  import_data, pod_svd, podproj_svd

# Inputs
lx = 8; ly = 1
nx = 128; ny = int(nx/8)
Re = 1e4; Ri = 4; Pr = 1
Tm = 8; dt = 5e-4; nt = int(np.round(Tm/dt))
ns = 800; freq = int(nt/ns)

#%% grid
dx = lx/nx; dy = ly/ny
x = np.linspace(0.0,lx,nx+1); y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% pod basis generation
nstart= 0; nend = nt; nstep = freq
nr = 50 #number of basis to store [we might not need to *use* all of them]
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
    tmp = np.copy(Phiw[:,k]).reshape([nx+1,ny+1])
    tmp = poisson_fst(nx,ny,dx,dy,tmp)
    Phis[:,k] = tmp.reshape([-1,])

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

    if ii%100 == 0:
        print(ii)


#%% Save data
filename = './results/pod_' + str(nx) + 'x' + str(ny) + '.npz'
np.savez(filename, wm = wm, Phiw = Phiw,  sm = sm, Phis = Phis,  \
                   tm = tm, Phit = Phit, \
                   aTrue = aTrue, bTrue = bTrue,\
                   Lw = Lw, Lt = Lt)
