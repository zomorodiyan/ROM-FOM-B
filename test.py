from tools import import_data
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from tools import import_data_test

nt = int(np.round(8/5e-4))
nx = 256; ny = int(nx/8)
lx = 8
ly = 1
x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% Load FOM results for t=0,2,4,8
w0,s0,t0 = import_data_test(nx,ny,0)
w2,s2,t2 = import_data_test(nx,ny,200)
w4,s4,t4 = import_data_test(nx,ny,400)
w8,s8,t8 = import_data_test(nx,ny,800)

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font) # qqq

nlvls = 31
x_ticks = [0,1,2,3,4,5,6,7,8]
y_ticks = [0,1]

#colormap = 'viridis'
#colormap = 'gnuplot'
#colormap = 'inferno'
colormap = 'seismic'

# qqq
v = np.linspace(1.05, 1.45, nlvls, endpoint=True)
ctick = np.linspace(1.05, 1.45, 5, endpoint=True)

fig, axs = plt.subplots(4,1,figsize=(10,8.5))
axs= axs.flat

cs = axs[0].contour(X,Y,t0,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[1].contour(X,Y,t2,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[2].contour(X,Y,t4,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[3].contour(X,Y,t8,v,cmap=colormap,linewidths=0.5)#, rasterized=True)
cs.set_clim([1.05, 1.45])

for i in range(4):
    axs[i].set_xticks(x_ticks)
    axs[i].set_xlabel('$x$')
    axs[i].set_yticks(y_ticks)
    axs[i].set_ylabel('$y$')

# Add titles
fig.text(0.92, 0.83, '$t=0$', va='center')
fig.text(0.92, 0.63, '$t=2$', va='center')
fig.text(0.92, 0.43, '$t=4$', va='center')
fig.text(0.92, 0.23, '$t=8$', va='center')

fig.subplots_adjust(bottom=0.18, hspace=1)
cbar_ax = fig.add_axes([0.125, 0.03, 0.775, 0.045])
CB = fig.colorbar(cs, cax = cbar_ax, ticks=ctick, orientation='horizontal')
CB.ax.get_children()[0].set_linewidths(3.0)

plt.savefig('./results/BSROMFOM.png', dpi = 500, bbox_inches = 'tight')
