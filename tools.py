import numpy as np
import os
from scipy.fftpack import dst, idst
from numpy import linalg as LA

def jacobian(nx,ny,dx,dy,q,s):
    # compute jacobian using arakawa scheme
    # computed at all internal physical domain points (1:nx-1,1:ny-1)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    #Arakawa 1:nx,1:ny
    j1 = gg*( (q[2:nx+1,1:ny]-q[0:nx-1,1:ny])*(s[1:nx,2:ny+1]-s[1:nx,0:ny-1]) \
             -(q[1:nx,2:ny+1]-q[1:nx,0:ny-1])*(s[2:nx+1,1:ny]-s[0:nx-1,1:ny]))

    j2 = gg*( q[2:nx+1,1:ny]*(s[2:nx+1,2:ny+1]-s[2:nx+1,0:ny-1]) \
            - q[0:nx-1,1:ny]*(s[0:nx-1,2:ny+1]-s[0:nx-1,0:ny-1]) \
            - q[1:nx,2:ny+1]*(s[2:nx+1,2:ny+1]-s[0:nx-1,2:ny+1]) \
            + q[1:nx,0:ny-1]*(s[2:nx+1,0:ny-1]-s[0:nx-1,0:ny-1]))

    j3 = gg*( q[2:nx+1,2:ny+1]*(s[1:nx,2:ny+1]-s[2:nx+1,1:ny]) \
            - q[0:nx-1,0:ny-1]*(s[0:nx-1,1:ny]-s[1:nx,0:ny-1]) \
            - q[0:nx-1,2:ny+1]*(s[1:nx,2:ny+1]-s[0:nx-1,1:ny]) \
            + q[2:nx+1,0:ny-1]*(s[2:nx+1,1:ny]-s[1:nx,0:ny-1]) )
    jac = (j1+j2+j3)*hh

    return jac

def laplacian(nx,ny,dx,dy,w):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    # 2nd order centered difference scheme
    lap = aa*(w[2:nx+1,1:ny]-2.0*w[1:nx,1:ny]+w[0:nx-1,1:ny]) \
        + bb*(w[1:nx,2:ny+1]-2.0*w[1:nx,1:ny]+w[1:nx,0:ny-1])
    return lap

def initial(nx,ny):
    #resting flow
    w = np.zeros([nx+1,ny+1])
    s = np.zeros([nx+1,ny+1])
    #masrigli flow [initial condition for temperature]
    t = np.zeros([nx+1,ny+1])
    t[:int(nx/2)+1,:] = 1.5
    t[int(nx/2)+1:,:] = 1
    return w,s,t

def RK3(rhs,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt):
    # time integration using third-order Runge Kutta method
    aa = 1.0/3.0
    bb = 2.0/3.0

    ww = np.zeros([nx+1,ny+1])
    tt = np.zeros([nx+1,ny+1])

    ww = np.copy(w)
    tt = np.copy(t)

    #stage-1
    rw,rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,t)
    ww[1:nx,1:ny] = w[1:nx,1:ny] + dt*rw
    tt[1:nx,1:ny] = t[1:nx,1:ny] + dt*rt
    s = poisson_fst(nx,ny,dx,dy,ww)
    tt = tbc(tt)

    #stage-2
    rw,rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,ww,s,tt)
    ww[1:nx,1:ny] = 0.75*w[1:nx,1:ny] + 0.25*ww[1:nx,1:ny] + 0.25*dt*rw
    tt[1:nx,1:ny] = 0.75*t[1:nx,1:ny] + 0.25*tt[1:nx,1:ny] + 0.25*dt*rt
    s = poisson_fst(nx,ny,dx,dy,ww)
    tt = tbc(tt)

    #stage-3
    rw,rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,ww,s,tt)
    w[1:nx,1:ny] = aa*w[1:nx,1:ny] + bb*ww[1:nx,1:ny] + bb*dt*rw
    t[1:nx,1:ny] = aa*t[1:nx,1:ny] + bb*tt[1:nx,1:ny] + bb*dt*rt
    s = poisson_fst(nx,ny,dx,dy,w)
    t = tbc(t)

    return w,s,t

def tbc(t):
    t[0,:] = t[1,:]
    t[-1,:] = t[-2,:]
    t[:,0] = t[:,1]
    t[:,-1] = t[:,-2]
    return t

def poisson_fst(nx,ny,dx,dy,w):
    #Elliptic coupled system solver:
    #For 2D Boussinesq equation:
    f = np.zeros([nx-1,ny-1])
    f = np.copy(-w[1:nx,1:ny])

    #DST: forward transform
    ff = np.zeros([nx-1,ny-1])
    ff = dst(f, axis = 1, type = 1)
    ff = dst(ff, axis = 0, type = 1)

    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])

    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)
    u1 = ff/alpha

    #IDST: inverse transform
    u = idst(u1, axis = 1, type = 1)
    u = idst(u, axis = 0, type = 1)
    u = u/((2.0*nx)*(2.0*ny))

    ue = np.zeros([nx+1,ny+1])
    ue[1:nx,1:ny] = u

    return ue

def BoussRHS(nx,ny,dx,dy,Re,Pr,Ri,w,s,t):
    # w-equation
    rw = np.zeros([nx-1,ny-1])
    rt = np.zeros([nx-1,ny-1])

    #laplacian terms
    Lw = laplacian(nx,ny,dx,dy,w)
    Lt = laplacian(nx,ny,dx,dy,t)

    #conduction term
    dd = 1.0/(2.0*dx)
    Cw = dd*(t[2:nx+1,1:ny]-t[0:nx-1,1:ny])

    #Jacobian terms
    Jw = jacobian(nx,ny,dx,dy,w,s)
    Jt = jacobian(nx,ny,dx,dy,t,s)

    rw = -Jw + (1/Re)*Lw + Ri*Cw
    rt = -Jt + (1/(Re*Pr))*Lt

    return rw,rt

def velocity(nx,ny,dx,dy,s):
    #compute velocity components from streamfunction (internal points)
    u =  np.zeros([nx-1,ny-1])
    u = (s[1:nx,2:ny+1] - s[1:nx,0:ny-1])/(2*dy) # u = ds/dy
    v =  np.zeros([nx-1,ny-1])
    v = -(s[2:nx+1,1:ny] - s[0:nx-1,1:ny])/(2*dx) # v = -ds/dx
    return u,v

def export_data(nx,ny,n,w,s,t):
    folder = 'fom_'+ str(nx) + '_' + str(ny)
    if not os.path.exists('./results/'+folder):
        os.makedirs('./results/'+folder)
    filename = './results/'+folder+'/' + str(int(n))+'.npz'
    np.savez(filename,w=w,s=s,t=t)

def import_data(nx,ny,n):
    folder = 'fom_'+ str(nx) + '_' + str(ny)
    filename = './results/'+folder+'/' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

def pod_svd(nx,ny,dx,dy,nstart,nend,nstep,nr):
    ns = int((nend-nstart)/nstep)
    #compute temporal correlation matrix
    Aw = np.zeros([(nx+1)*(ny+1),ns+1]) #vorticity
    At = np.zeros([(nx+1)*(ny+1),ns+1]) #temperature
    ii = 0
    for i in range(nstart,nend+1,nstep):
        w,s,t = import_data(nx,ny,i)
        Aw[:,ii] = w.reshape([-1,])
        At[:,ii] = t.reshape([-1,])
        ii = ii + 1

    #mean subtraction
    wm = np.mean(Aw,axis=1)
    tm = np.mean(At,axis=1)

    Aw = Aw - wm.reshape([-1,1])
    At = At - tm.reshape([-1,1])

    #singular value decomposition
    Uw, Sw, Vhw = LA.svd(Aw, full_matrices=False)
    Ut, St, Vht = LA.svd(At, full_matrices=False)

    Phiw = Uw[:,:nr]
    Lw = Sw**2
    #compute RIC (relative importance index)
    RICw = sum(Lw[:nr])/sum(Lw)*100

    Phit = Ut[:,:nr]
    Lt = St**2
    #compute RIC (relative importance index)
    RICt = sum(Lt[:nr])/sum(Lt)*100

    return wm,Phiw,Lw/sum(Lw),RICw , tm,Phit,Lt/sum(Lt),RICt

def podproj_svd(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T if shape of a is [ns,nr]
    return a

def podrec_svd(a,Phi): #Reconstruction
    u = np.dot(Phi,a.T)
    return u

def window_data(serie, window_size):
    n_snapshots = serie.shape[0]
    n_states = serie.shape[1]
    ytrain = serie[window_size:,:]
    xtrain = np.zeros((n_snapshots-window_size, window_size, n_states))
    for i in range(n_snapshots-window_size):
        tmp = serie[i,:]
        for j in range(1,window_size):
            tmp = np.vstack((tmp,serie[i+j,:]))
        xtrain[i,:,:] = tmp
    return xtrain , ytrain
