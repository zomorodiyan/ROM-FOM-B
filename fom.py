"""
written by Shady, https://github.com/Shady-Ahmed
edited by Mehrdad, https://github.com/zomorodiyan
"""
#%% Import libraries
import numpy as np

#%% Define functions
from tools import jacobian, laplacian, initial, RK3, tbc, \
                  poisson_fst, BoussRHS, velocity, export_data

#%% Main program
# Inputs
lx = 8
ly = 1
#nx = 4096
nx = 128
ny = int(nx/8)

Re = 1e4
Ri = 4
Pr = 1

Tm = 8
dt = 5e-4
nt = int(np.round(Tm/dt))

ns = 800
freq = int(nt/ns)

#%% grid
dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# initialize
n= 0
time=0
w,s,t = initial(nx,ny)
export_data(nx,ny,n,w,s,t)

#%% time integration
for n in range(1,nt+1):
    time = time+dt

    w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)

    u,v = velocity(nx,ny,dx,dy,s)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))

    cfl = np.max([umax*dt/dx, vmax*dt/dy])
    if cfl >= 0.8:
        print('CFL exceeds maximum value')
        break

    if n%500==0:
        print(n, " ", time, " ", np.max(w), " ", cfl)

    if n%freq==0:
        export_data(nx,ny,n,w,s,t)
