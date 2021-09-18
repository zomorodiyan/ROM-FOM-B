import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tools import winLen, make_win

# Load inputs
from inputs import nx, ny, Re, Ri, Pr, dt, nt, ns, freq, eqspace, ws, epochs,\
        nr

filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
aTrue = data['aTrue']; bTrue = data['bTrue']

data = np.concatenate((aTrue, bTrue), axis=1) # axes 0:snapshots 1:states
percent = 100
data = data[:int(data.shape[0]*percent/100),:]

# Scale Data
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_data = scaler.fit_transform(data)
filename = './results/scaler_' + str(nx) + 'x' + str(ny) + '.npz'
np.savez(filename, scalermin = scaler.data_min_, scalermax = scaler.data_max_)

#==============================================================================
scaled_aTrue = scaled_data[:,:nr] #m2
scaled_bTrue = scaled_data[:,nr:] #m2
serie = np.concatenate((scaled_aTrue,scaled_bTrue),axis=1) #m2
#==============================================================================

serie_len = serie.shape[0]
n_states = serie.shape[1]

xtrain = np.empty((0,ws,n_states), float)
ytrain = np.empty((0,n_states), float)
for i in range(winLen(),serie_len):
    winx = np.expand_dims(make_win(i,serie),axis=0)
    winy = np.expand_dims(serie[i,:],axis=0)
    xtrain = np.vstack((xtrain, winx))
    ytrain = np.vstack((ytrain, winy))

#Shuffling data
seed(1) # this line & next, what will they affect qqq
tf.random.set_seed(0)
perm = np.random.permutation(xtrain.shape[0]) # xtarin.shape[0] is (n_snapshots - ws)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

#create the LSTM architecture
model = Sequential()
model.add(LSTM(80, input_shape=(ws, n_states), return_sequences=True, activation='tanh'))
model.add(LSTM(80, input_shape=(ws, n_states), activation='tanh'))
model.add(Dense(n_states))

#compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

#run the model
history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=800,
        validation_split=0.20, verbose=0)

#evaluate the model
scores = model.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'results/loss_' + str(nx) + 'x' + str(ny) + '.png'
plt.savefig(filename, dpi = 200)
plt.show()

#Removing old models
model_name = 'results/lstm_' + str(nx) + 'x' + str(ny) + '.h5'
if os.path.isfile(model_name):
   os.remove(model_name)
#Save the model
model.save(model_name)
