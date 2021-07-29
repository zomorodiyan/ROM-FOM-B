import numpy as np
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS, podrec_svd, RK3t,\
        BoussRHS_t, export_data_test
from sklearn.preprocessing import MinMaxScaler

lx = 8 #length in x direction
ly = 1 #length in y direction
nx = 256 #number of meshes in x direction
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
window_size = 5
lx = 8; ly = 1
nx = 256; ny = int(nx/8)
Re = 1e4; Ri = 4; Pr = 1
Tm = 8; dt = 5e-4; nt = int(np.round(Tm/dt))
ns = 800; freq = int(nt/ns)
nr = 10

abromfom = np.empty([801,2*nr]) # for diagrams
a_window = np.empty([window_size,nr])
b_window = np.empty([window_size,nr])

# load data
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
w_mean = data['wm']; Phiw = data['Phiw']
t_mean = data['tm']; Phit = data['Phit']
s_mean = data['sm']; Phis = data['Phis']

# init
n=0; time=0;
w,s,t = initial(nx,ny)
export_data_test(nx,ny,0,w,s,t)
w_1d = w.reshape([-1,])
w_spread_1d = w_1d - w_mean
a_window[0,:] = podproj_svd(w_spread_1d,Phiw)
t_1d = t.reshape([-1,])
t_spread_1d = t_1d - t_mean
b_window[0,:] = podproj_svd(t_spread_1d,Phit)

# for the first window_size steps
#     get alpha beta for each time step

for i in range(1, window_size):
    time = time+dt*nt/ns; n = n+1
    #run fom, get psi omega theta for each time step
    for j in range (0,int(nt/ns)):
        w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    #run phi_psi phi_omega phi_theta projection on psi omega theta,
    w_1d = w.reshape([-1,])
    w_spread_1d = w_1d - w_mean
    a_window[i,:] = podproj_svd(w_spread_1d,Phiw)
    t_1d = t.reshape([-1,])
    t_spread_1d = t_1d - t_mean
    b_window[i,:] = podproj_svd(t_spread_1d,Phit)
    export_data_test(nx,ny,i,w,s,t)
print('initial foms are done!')
print("w: ", w); print("s: ", s); print("t: ", t);

ab_window = np.concatenate((a_window, b_window), axis=1)
filename = './results/scaler_'+ str(nx) + 'x' + str(ny) + '.npz'
data2 = np.load(filename)
scalermin = data2['scalermin']; scalermax = data2['scalermax']
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit([scalermin,scalermax])
ab_window = scaler.transform(ab_window)
abromfom[0:window_size,:] = ab_window # for diagrams

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')

for i in range(window_size, ns+1):
    time = time+dt*nt/ns; n = n+1
    # use fom_energy on psi omega theta, get theta_new
    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt*nt/ns)
    #run model on a window of alpha beta, get alpha_new (rom/ml)
    ab_lstm = model.predict(np.expand_dims(ab_window, axis=0))
    a_new = ab_lstm[:,0:nr]
    # update w s using alpha_new and phiw phis
    ab_lstm_invscal = scaler.inverse_transform(ab_lstm)
    a_new_invscal = ab_lstm_invscal[:,0:nr]
    w_1d = podrec_svd(a_new_invscal, Phiw) + w_mean.reshape([-1,1])
    w = w_1d.reshape([nx+1,-1])
    s_1d = podrec_svd(a_new_invscal, Phis) + s_mean.reshape([-1,1])
    s = s_1d.reshape([nx+1,-1])
    # project theta_spread on phi_theta, get beta_new
    t_1d = t.reshape([-1,])
    b_new = np.expand_dims(podproj_svd(t_1d - t_mean,Phit), axis=0)
    # make ab_new and put it at the end of ab_window and shift the rest back
    ab_new = scaler.transform(np.concatenate((a_new_invscal,\
        b_new), axis=1))
    for j in range(1, window_size):
        ab_window[j-1,:] = ab_window[j,:]
    ab_window[window_size-1,:] = ab_new
    abromfom[i,:] = ab_new # for diagrams
    export_data_test(nx,ny,i,w,s,t) # for romfom contour

#%% the FOM, and ROM, part of the diagrams
# Load Data
nx = 256; ny = int(nx/8)
ws=5;
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
aTrue = data['aTrue']; bTrue = data['bTrue']
print('aTrue.shape: ', aTrue.shape)
print('bTrue.shape: ', bTrue.shape)

# Scale Data
data = np.concatenate((aTrue, bTrue), axis=1) # axes 0:snapshots 1:states
scaler2 = MinMaxScaler(feature_range=(-1,1))
scaled = scaler2.fit_transform(data)
print('scaled.shape: ', scaled.shape)

t = np.arange(0, 801, 1)

ablstm = np.empty([801,scaled.shape[1]])
xtest = np.empty([1,ws,scaled.shape[1]])

print("ablstm.shape: ", ablstm.shape)

ablstm[0:ws,:] = scaled[0:ws,:]
print("ablstm[0:ws,1]: ", ablstm[0:ws,1])

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')
xtest = np.copy(np.expand_dims(ablstm[0:ws,:], axis=0))
for i in range(ws, 801):
    print(i)
    ablstm[i,:] = model.predict(xtest)
    for j in range(ws-1):
        xtest[0,j,:] = xtest[0,j+1,:]
    xtest[0,ws-1,:] = ablstm[i,:]

for n in range(0,20):
    s1 = scaled[:,n]
    print("s1.shape: ", s1.shape)
    s2 = ablstm[:,n]
    print("s2.shape: ", s2.shape)
    s3 = abromfom[:,n]
    print("s3.shape: ", s3.shape)

    fig, ax = plt.subplots()
    ax.plot(t, s1)
    ax.plot(t, s2)
    ax.plot(t, s3)
    ax.legend(["fom", "rom","romfom"])
    ax.set(xlabel='timeStep', ylabel='y',
           title='ab['+str(n)+'] evolution')
    ax.grid()
    #plt.ylim(-1,1)
    fig.savefig('./results/'+str(nx)+'_ab['+str(n)+']')
    plt.show()
    plt.clf()


