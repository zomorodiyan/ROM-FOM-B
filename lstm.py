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
from tools import window_data

#%% Main program
# Load Data
nx = 128; ny = int(nx/8)
filename = './results/pod_'+ str(nx) + '_' + str(ny) + '.npz'
data = np.load(filename)
aTrue = data['aTrue']; bTrue = data['bTrue']

# Scale Data
data = np.concatenate((aTrue, bTrue), axis=1) # axes 0:snapshots 1:states
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_data = scaler.fit_transform(data)

# Training Data X & Y
serie = scaled_data
n_states = serie.shape[1]
window_size = 3
xtrain, ytrain = window_data(serie=serie,window_size=window_size)

#Shuffling data
seed(1) # this line & next, what will they affect qqq
tf.random.set_seed(0)
perm = np.random.permutation(xtrain.shape[0]) # xtarin.shape[0] is (n_snapshots - window_size)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

#create the LSTM architecture
model = Sequential()
model.add(LSTM(30, input_shape=(window_size, n_states), return_sequences=True, activation='tanh'))
model.add(LSTM(30, input_shape=(window_size, n_states), activation='tanh'))
model.add(Dense(n_states))

#compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

#run the model
history = model.fit(xtrain, ytrain, epochs=100, batch_size=64,
        validation_split=0.20, verbose=1)

#evaluate the model
scores = model.evaluate(xtrain, ytrain, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'results/loss_' + str(n_states) + '.png'
plt.savefig(filename, dpi = 200)
plt.show()

#Removing old models
model_name = 'results/lstm_' + str(n_states) + '.h5'
if os.path.isfile(model_name):
   os.remove(model_name)
#Save the model
model.save(model_name)
