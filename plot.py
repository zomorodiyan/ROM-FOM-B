from tools import import_data,import_data2,import_data3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Load inputs
from inputs import lx, ly, nx, ny, nt, ns, nr, Re
x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% Load FOM results for t=0,2,4,8
n=0 #t=0
w0,s0,t0 = import_data(nx,ny,n)
n=int(2*nt/8) #t=2
w2,s2,t2 = import_data(nx,ny,n)
n=int(4*nt/8) #t=4
w4,s4,t4 = import_data(nx,ny,n)
n=int(8*nt/8) #t=8
w8,s8,t8 = import_data(nx,ny,n)

#%% Load ROMFOM results for t=0,2,4,8
n=0 #t=0
w0rf,s0rf,t0rf = import_data2(nx,ny,n,'romfom')
n=int(2*ns/8) #t=2
w2rf,s2rf,t2rf = import_data2(nx,ny,n,'romfom')
n=int(4*ns/8) #t=4
w4rf,s4rf,t4rf = import_data2(nx,ny,n,'romfom')
n=int(8*ns/8) #t=8
w8rf,s8rf,t8rf = import_data2(nx,ny,n,'romfom')

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
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

fig, axs = plt.subplots(4,2,figsize=(20,10))
axs= axs.flat

cs = axs[0].contour(X,Y,t0,v,cmap=colormap,linewidths=3)
cs.set_clim([1.05, 1.45])
cs = axs[2].contour(X,Y,t2,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[4].contour(X,Y,t4,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[6].contour(X,Y,t8,v,cmap=colormap,linewidths=0.5)#, rasterized=True)
cs.set_clim([1.05, 1.45])

cs = axs[1].contour(X,Y,t0rf,v,cmap=colormap,linewidths=3)
cs.set_clim([1.05, 1.45])
cs = axs[3].contour(X,Y,t2rf,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[5].contour(X,Y,t4rf,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[7].contour(X,Y,t8rf,v,cmap=colormap,linewidths=0.5)#, rasterized=True)
cs.set_clim([1.05, 1.45])

for i in range(8):
    axs[i].set_xticks(x_ticks)
    if(i<6):
        axs[i].set_xlabel('$x$')
    elif(i==6):
        axs[i].set_xlabel('$x$\nFOM')
    else:
        axs[i].set_xlabel('$x$\nROM-FOM')
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

if not os.path.exists('./results/nx_'+str(nx)+'_Re_'+str(int(Re))):
    os.makedirs('./results/nx_'+str(nx)+'_Re_'+str(int(Re)))

plt.savefig('./results/nx_'+str(nx)+'_Re_'+str(int(Re))+'/FOM.png', dpi = 500, bbox_inches = 'tight')

s1 = import_data3(nx,ny,'rom')
s2 = import_data3(nx,ny,'fom')
s3 = import_data3(nx,ny,'romfom')
t = np.arange(0, ns+1, 1)
for n in range(2*nr):
    fig, ax = plt.subplots(figsize=(10,8.5))
    ax.plot(t, s1[:ns+1,n])
    ax.plot(t, s2[:ns+1,n])
    ax.plot(t, s3[:ns+1,n])
    ax.legend(["fom", "rom","romfom"], loc='upper left')
    ax.grid()
    #plt.ylim(-1,1)
    if(n<10):
        ax.set(xlabel='timeStep', ylabel='y',
           title='alpha '+str(n)+' evolution')
        fig.savefig('./results/nx_'+str(nx)+'_Re_'+str(int(Re))+'/alhpa['+str(n)+']')
    else:
        ax.set(xlabel='timeStep', ylabel='y',
           title='beta '+str(n-nr)+' evolution')
        fig.savefig('./results/nx_'+str(nx)+'_Re_'+str(int(Re))+'/beta['+str(n)+']')
    if(n==0):
        plt.show()
    plt.clf()
