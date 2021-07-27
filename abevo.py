import matplotlib.pyplot as plt
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load Data
nx = 1024; ny = int(nx/8)
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

t = np.arange(0, 801, 1)

ws=3
ablstm = np.empty([801,2*10])
xtest = np.empty([1,ws,2*10])

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

n=1
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
