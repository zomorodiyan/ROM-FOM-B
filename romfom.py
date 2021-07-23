import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS, podrec_svd, RK3t, BoussRHS_t
from sklearn.preprocessing import MinMaxScaler

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

alpha_window = np.zeros([window_size,nr])
beta_window = np.zeros([window_size,nr])
xtest = np.empty((ns-window_size, window_size, 2*nr))
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
w_mean = data['wm']; Phiw = data['Phiw']
t_mean = data['tm']; Phit = data['Phit']
s_mean = data['sm']; Phis = data['Phis']

n=0; time=0;
w,s,t = initial(nx,ny)
w_1d = w.reshape([-1,])
w_spread_1d = w_1d - w_mean
alpha_window[0,:] = podproj_svd(w_spread_1d,Phiw)
t_1d = t.reshape([-1,])
t_spread_1d = t_1d - t_mean
beta_window[0,:] = podproj_svd(t_spread_1d,Phit)

# for the first window_size steps
#   1-run fom, get psi omega theta for each time step
#   2-run phi_psi phi_omega phi_theta projection on psi omega theta,
#     get alpha beta for each time step

for i in range(1, window_size):
    time = time+dt
    w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    w_1d = w.reshape([-1,])
    w_spread_1d = w_1d - w_mean
    alpha_window[i,:] = podproj_svd(w_spread_1d,Phiw)
    t_1d = t.reshape([-1,])
    t_spread_1d = t_1d - t_mean
    beta_window[i,:] = podproj_svd(t_spread_1d,Phit)
    alphabeta_window = np.concatenate((alpha_window, beta_window), axis=1)


filename = './results/scaler_'+ str(nx) + 'x' + str(ny) + '.npz'
data2 = np.load(filename)
scalermin = data2['scalermin']; scalermax = data2['scalermax']
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit([scalermin,scalermax])
alphabeta_window = scaler.fit_transform(alphabeta_window)
# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')
#model.summary()

#for i in range(window_size, nt):
for i in range(window_size, 2*window_size): # just for test: replace with the upper line
    alphalstmbeta_new = model.predict(np.expand_dims(alphabeta_window, axis=0))
    alpha_new = alphalstmbeta_new[:,0:nr]
    # update w s using alpha_new and phiw phis
    alphalstmbeta_new_invscaled = scaler.inverse_transform(alphalstmbeta_new)
    alpha_new_invscaled = alphalstmbeta_new_invscaled[:,0:nr]
    w_1d = podrec_svd(alpha_new_invscaled, Phiw) + w_mean.reshape([-1,1]) # maybe no need to reshape, (whats the difference between [-1,] and [-1,1]
    w = w_1d.reshape([nx+1,-1])
    s_1d = podrec_svd(alpha_new, Phis) + s_mean.reshape([-1,1])
    s = s_1d.reshape([nx+1,-1])
    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    t_1d = t.reshape([-1,])
    t_spread_1d = t_1d - t_mean
    beta_new = np.expand_dims(podproj_svd(t_spread_1d,Phit), axis=0)
    alphabeta_new = np.concatenate((alpha_new, beta_new), axis=1)
    for i in range(1, window_size):
        alphabeta_window[i-1,:] = alphabeta_window[i,:]
    alphabeta_window[window_size-1,:] = alphabeta_new


# for the rest of the steps
#   3-run model on a window of alpha beta, get alpha_new (rom/ml)
#   4-run reconstruct on phi_psi phi_omega alpha_new, get psi_new omega_new
#   5-run fom_energy on psi_new omega_new theta, get theta_new
#   6-run project on phi_psi phi_omega theta_new, get beta_new
