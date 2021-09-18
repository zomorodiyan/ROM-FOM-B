import numpy as np

ns = 800 #number of snapshots
eqspace = 5 #once every eqspace time snapshots is used for lstm's data-windows
nr = 20 #number of pod modes

nx = 512 #number of meshes in x direction
ny = int(nx/8) #number of meshes in y direction
nplot = ns+1 # number of points to plot (including 0)
lx = 8 #length in x direction
ly = 1 #length in y direction

Re = 5e3 #Reynolds Number: inertial/viscous
Ri = 4 #Richardson Number: Buoyancy/flow_shear
Pr = 1 #Prandtl Number: momentum_diffusivity/thermal_diffusivity

Tm = 8 #maximum time
dt = 1e-3 #time_step_size
nt = np.int(np.round(Tm/dt)) #number of time_steps
freq = np.int(nt/ns) #every freq time_step we export data
ws = 5 #window size
winName = 'addspace' # options are 'regular', 'eqspace', 'addspace'

epochs = 3000
