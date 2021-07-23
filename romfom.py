import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS

def RK3t(rhs,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt):
# time integration using third-order Runge Kutta method
    aa = 1.0/3.0
    bb = 2.0/3.0

    tt = np.zeros([nx+1,ny+1])
    tt = np.copy(t)

    #stage-1
    rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,t)
    tt[1:nx,1:ny] = t[1:nx,1:ny] + dt*rt
    tt = tbc(tt)

    #stage-2
    rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,tt)
    tt[1:nx,1:ny] = 0.75*t[1:nx,1:ny] + 0.25*tt[1:nx,1:ny] + 0.25*dt*rt
    tt = tbc(tt)

    #stage-3
    rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,tt)
    t[1:nx,1:ny] = aa*t[1:nx,1:ny] + bb*tt[1:nx,1:ny] + bb*dt*rt
    t = tbc(t)
    return t

def BoussRHS_t(nx,ny,dx,dy,Re,Pr,Ri,w,s,t):
    rt = np.zeros([nx-1,ny-1]) #define
    Lt = laplacian(nx,ny,dx,dy,t) #laplacian terms
    Jt = jacobian(nx,ny,dx,dy,t,s) #Jacobian terms
    rt = -Jt + (1/(Re*Pr))*Lt # t-equation
    return rt

lx = 8 #length in x direction
ly = 1 #length in y direction
nx = 128 #number of meshes in x direction
ny = int(nx/8) #number of meshes in y direction

Re = 1e4 #Reynolds Number: inertial/viscous
Ri = 4 #Richardson Number: Buoyancy/flow_shear
Pr = 1 #Prandtl Number: momentum_diffusivity/thermal_diffusivity

Tm = 8 #maximum time
dt = 5e-4 #time_step_size
nt = np.int(np.round(Tm/dt)) #number of time_steps

ns = 800 #number of snapshots
freq = np.int(nt/ns) #every freq time_stap we export data

#%% grid
dx = lx/nx
dy = ly/ny
x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# run a coarse fom, get psi omega theta for each step and each mesh
# (run fom.py get fom_nx'x'ny)

# run svd on psi phi theta, get phi_psi phi_omega phi_theta for each mesh
#   and alpha beta for number of modes and each time step
# (run pod.py)

# run lstm on alpha beta, get model
# (run lstm.py)

# Inputs (move to yml)
window_size = 3

lx = 8; ly = 1
nx = 128; ny = int(nx/8)
Re = 1e4; Ri = 4; Pr = 1
Tm = 8; dt = 5e-4; nt = int(np.round(Tm/dt))
ns = 800; freq = int(nt/ns)
nr = 50

alpha = np.zeros([ns+1,nr])
beta = np.zeros([ns+1,nr])
xtest = np.zeros((ns-window_size, window_size, 2*nr))
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
w_mean = data['wm']; Phiw = data['Phiw']
t_mean = data['tm']; Phit = data['Phit']

n=0; time=0;
w,s,t = initial(nx,ny)
w_1d = w.reshape([-1,])
w_spread_1d = w_1d - w_mean
alpha[0,:] = podproj_svd(w_spread_1d,Phiw)
t_1d = t.reshape([-1,])
t_spread_1d = t_1d - t_mean
beta[0,:] = podproj_svd(t_spread_1d,Phit)

# for the first window_size steps
#   1-run fom, get psi omega theta for each time step
#   2-run phi_psi phi_omega phi_theta projection on psi omega theta,
#     get alpha beta for each time step

#calculate tm
for i in range(1, window_size):
    time = time+dt
    w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    w_1d = w.reshape([-1,])
    w_spread_1d = w_1d - w_mean
    alpha[i,:] = podproj_svd(w_spread_1d,Phiw)
    t_1d = t.reshape([-1,])
    t_spread_1d = t_1d - t_mean
    beta[i,:] = podproj_svd(t_spread_1d,Phit)
    xtest[0,:,:] = np.concatenate((alpha, beta), axis=1)[0:window_size,:] # axes 0:snapshots 1:states

#for i in range(window_size, nt):

# for the rest of the steps
#   3-run model on a window of alpha beta, get alpha_new (rom/ml)
#   4-run reconstruct on phi_psi phi_omega alpha_new, get psi_new omega_new
#   5-run fom_energy on psi_new omega_new theta, get theta_new
#   6-run project on phi_psi phi_omega theta_new, get beta_new


