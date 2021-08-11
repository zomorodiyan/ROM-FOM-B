import matplotlib.pyplot as plt
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load Data
nx = 256; ny = int(nx/8)
ws=5
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
aTrue = data['aTrue']; bTrue = data['bTrue']
print('aTrue.shape: ', aTrue.shape)
print('bTrue.shape: ', bTrue.shape)

# Scale Data
data = np.concatenate((aTrue, bTrue), axis=1) # axes 0:snapshots 1:states
scaler = MinMaxScaler(feature_range=(-1,1))
scaled = scaler.fit_transform(data)
print('scaled.shape: ', scaled.shape)

t = np.arange(0, 1601, 1)

ablstm = np.empty([1601,scaled.shape[1]])
xtest = np.empty([1,ws,scaled.shape[1]])

n_each = 10
ablstm[0:ws*n_each,:] = scaled[0:ws*n_each,:]

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')


xtest = np.copy(np.expand_dims(ablstm[0:ws*n_each:n_each,:], axis=0))
for i in range(ws*n_each, 1601):
    print(i)
    ablstm[i,:] = model.predict(xtest)
    xtest = np.copy(np.expand_dims(ablstm[i-ws*n_each+1:i-n_each+2:n_each,:], axis=0))

n=9
s1 = scaled[:,n]
print("s1.shape: ", s1.shape)
s2 = ablstm[:,n]
print("s2.shape: ", s2.shape)

fig, ax = plt.subplots()
ax.plot(t, s1)
ax.plot(t, s2)
ax.legend(["fom", "lstm"])


ax.set(xlabel='timeStep', ylabel='y',
       title='some alphas evolution')
ax.grid()
#plt.ylim(-1,1)
fig.savefig("test.png")
plt.show()
