import numpy as np
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS, podrec_svd, RK3t,\
        BoussRHS_t, export_data_test, export_data2, export_data3, show_percent
from sklearn.preprocessing import MinMaxScaler

# Load inputs
from inputs import lx, ly, nx, ny, Re, Ri, Pr, dt, nt, ns, freq, nr, n_each, ws

#%% define required variables
abromfom = np.empty([ns+1,2*nr])
length = ws*n_each
az = np.empty([length,nr])
bz = np.empty([length,nr])
ablstm = np.empty([ns+1,2*nr])
xtest = np.empty([1,ws,2*nr])
dx = lx/nx; dy = ly/ny

# load data
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
w_mean = data['wm']; Phiw = data['Phiw'] # for ROM-FOM
t_mean = data['tm']; Phit = data['Phit']
s_mean = data['sm']; Phis = data['Phis']
aTrue = data['aTrue']; bTrue = data['bTrue'] # for ROM-ROM

# reconstruct the same scaler as the training time using the saved min and max
filename = './results/scaler_'+ str(nx) + 'x' + str(ny) + '.npz'
data2 = np.load(filename)
scalermin = data2['scalermin']; scalermax = data2['scalermax']
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit([scalermin,scalermax])

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')

# --------------------------------ROM-FOM--------------------------------------
# init
w,s,t = initial(nx,ny)

# for the first window_size steps get a, b
for i in range(length):
    show_percent('initFOM',i,0,length)
    export_data2(nx,ny,i,w,s,t,'romfom') # for romfom contour
    w_1d = w.reshape([-1,]); w_spread_1d = w_1d - w_mean #evaluate w_spread_1d
    az[i,:] = podproj_svd(w_spread_1d,Phiw) # get new a
    t_1d = t.reshape([-1,]); t_spread_1d = t_1d - t_mean #evaluate w_spread_1d
    bz[i,:] = podproj_svd(t_spread_1d,Phit) # get new b
    if(i != length-1): #run fom for freq. times; update w,s,t
        for j in range (freq):
            w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)

# scale alphabetaz
abromfom[0:length,:] = scaler.transform(np.concatenate((az, bz), axis=1))

# for the rest of steps get a, b
for i in range(length, ns+1):
    show_percent('ROMFOM',i,length,ns+1)
    # update <<theta>>, use fom_energy on psi omega theta
    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt*freq)
    # get new <<alpha>>, run model on a window of alpha beta
    abz = np.copy(abromfom[i-length:i-n_each+1:n_each,:])
    ab_lstm = model.predict(np.expand_dims(abz, axis=0))
    a_new = ab_lstm[:,0:nr] # first nr ones are the alphas
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
    # put ab_new at the end of abromfom
    abromfom[i,:] = ab_new
    export_data2(nx,ny,i,w,s,t,'romfom') # for romfom contour

# --------------------------------ROM-ROM--------------------------------------
# Scale Data
ablstm[0:length,:] = scaled[0:length,:] # already did it for ROM-FOM

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')

for i in range(length, ns+1):
    show_percent('ROM',i,length,ns+1)
    xtest = np.copy(np.expand_dims(ablstm[i-length:i-n_each+1:n_each,:], axis=0))
    ablstm[i,:] = model.predict(xtest)

#%% export alpha and beta for rom, fom, romfom
export_data3(nx,ny,ablstm,'rom')
export_data3(nx,ny,scaled,'fom')
export_data3(nx,ny,abromfom,'romfom')
