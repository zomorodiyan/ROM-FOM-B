import numpy as np
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS, podrec_svd, RK3t,\
        BoussRHS_t, export_data_test
from sklearn.preprocessing import MinMaxScaler

# run a coarse fom, get psi omega theta for each step and each mesh
# (run fom.py get fom_nx'x'ny)

# run svd on psi phi theta, get phi_psi phi_omega phi_theta for each mesh
#   and alpha beta for number of modes and each time step
# (run pod.py get pod_nx'x'ny'.npz')

# run lstm on alpha beta, get model
# (run lstm.py get lstm_nx'x'ny'.h5')

# then run this

# Load inputs
from inputs import lx, ly, nx, ny, Re, Ri, Pr, dt, nt, ns, freq, nr, n_each, ws

#%% grid
dx = lx/nx
dy = ly/ny
x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')


abromfom = np.empty([ns+1,2*nr])
a_window = np.empty([ws*n_each,nr])
b_window = np.empty([ws*n_each,nr])

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
for i in range(1, ws*n_each):
    print('initFOM ',"{:.0f}".format(i/(ws*n_each-1)*100), '%   ', end='\r')
    time = time+dt*nt/ns; n = n+1
    #run fom, get psi omega theta for each time step
    for j in range (0,freq):
        w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    # make omega_spread_1d to use for projection to get alpha
    w_1d = w.reshape([-1,])
    w_spread_1d = w_1d - w_mean
    # get new <<alpha>>, project w_spread on Phiw
    a_window[i,:] = podproj_svd(w_spread_1d,Phiw)
    # make theta_spread_1d to use for projection to get beta
    t_1d = t.reshape([-1,])
    t_spread_1d = t_1d - t_mean
    # get new <<beta>>, project w_spread on Phit
    b_window[i,:] = podproj_svd(t_spread_1d,Phit)
    export_data_test(nx,ny,i,w,s,t) # for test contour
print('')

# concatenate alpha beta together
ab_window = np.concatenate((a_window, b_window), axis=1)
# reconstruct the same scaler as the training time using the saved min and max
filename = './results/scaler_'+ str(nx) + 'x' + str(ny) + '.npz'
data2 = np.load(filename)
scalermin = data2['scalermin']; scalermax = data2['scalermax']
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit([scalermin,scalermax])
# scale alphabeta_window
ab_window = scaler.transform(ab_window)
abromfom[0:ws*n_each,:] = ab_window
ab_window_each = np.copy(abromfom[0:ws*n_each:n_each,:])
# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')
for i in range(ws*n_each, ns+1):
    print('ROMFOM ',"{:.0f}".format((i-ws*n_each)/(ns+1-ws*n_each)*100), '%   ', end='\r')
    time = time+dt*nt/ns; n = n+1
    # make sure about the BC. (doesn't make a difference in results)
    for j in range(w.shape[0]):
        w[j,0] = 0; w[j,-1]=0
    for j in range(w.shape[1]):
        w[0,j] = 0; w[-1,j]=0
    # update <<theta>>, use fom_energy on psi omega theta
    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt*freq)
    # get new <<alpha>>, run model on a window of alpha beta
    ab_lstm = model.predict(np.expand_dims(ab_window_each, axis=0))
    a_new = ab_lstm[:,0:nr]
    # inverse scale alphabeta and extract alpha to use in rec. of omega psi
    ab_lstm_invscal = scaler.inverse_transform(ab_lstm)
    a_new_invscal = ab_lstm_invscal[:,0:nr]
    # update <<omega>> using alpha_new_invscal and phiw
    w_1d = podrec_svd(a_new_invscal, Phiw) + w_mean.reshape([-1,1])
    w = w_1d.reshape([nx+1,-1])
    # update <<psi>> using alpha_new_invscal and phis
    s_1d = podrec_svd(a_new_invscal, Phis) + s_mean.reshape([-1,1])
    s = s_1d.reshape([nx+1,-1])
    # get new <<beta>>, project theta_spread on phi_theta
    t_1d = t.reshape([-1,])
    b_new = np.expand_dims(podproj_svd(t_1d - t_mean,Phit), axis=0)
    # make ab_new and scale so it can be used for the model
    ab_new = scaler.transform(np.concatenate((a_new_invscal,\
        b_new), axis=1))
    # put ab_new at the end of ab_window and shift the rest back
    abromfom[i,:] = ab_new
    ab_window_each = np.copy(abromfom[i-ws*n_each+1:i-n_each+2:n_each,:])
    #export_data_test(nx,ny,i,w,s,t) # for romfom contour
print('')

#%% the FOM, and ROM, part of the diagrams
# Load Data
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
aTrue = data['aTrue']; bTrue = data['bTrue']

# Scale Data
data = np.concatenate((aTrue, bTrue), axis=1) # axes 0:snapshots 1:states
scaler2 = MinMaxScaler(feature_range=(-1,1))
scaled = scaler2.fit_transform(data)
t = np.arange(0, ns+1, 1)
ablstm = np.empty([ns+1,scaled.shape[1]])
xtest = np.empty([1,ws,scaled.shape[1]])
ablstm[0:ws*n_each,:] = scaled[0:ws*n_each,:]

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')
#xtest = np.copy(np.expand_dims(ablstm[0:ws,:], axis=0))
xtest = np.copy(np.expand_dims(ablstm[0:ws*n_each:n_each,:], axis=0))
for i in range(ws*n_each, ns+1):
    print('ROM ',"{:.0f}".format((i-ws*n_each)/(ns+1-ws*n_each)*100), '%   ', end='\r')
    ablstm[i,:] = model.predict(xtest)
    xtest = np.copy(np.expand_dims(ablstm[i-ws*n_each+1:i-n_each+2:n_each,:], axis=0))

for n in range(0,20):
    s1 = scaled[:ns+1,n]
    s2 = ablstm[:ns+1,n]
    s3 = abromfom[:ns+1,n]

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
    if(n==0):
        plt.show()
    plt.clf()
