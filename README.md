This is the code for the comming APS DFD conference, "A hybrid modeling
approach for coupling reduced order and full order models of the Boussinesq
system"
here we model evolution of proper orthogonal decomposition modes of the
vorticity transport and continuity equations using long short-term memory (LSTM)
, coupled with the FOM solution of the energy equation. This ROM-FOM coupling
framework solves Boussinesq equations for the lock-exchange problem to 
demonstrate the benefits of this multi fidelity setup.

# run a coarse fom, get psi omega theta for each step and each mesh
# (run fom.py get fom_nx'x'ny)

# run svd on psi phi theta, get phi_psi phi_omega phi_theta for each mesh
#   and alpha beta for number of modes and each time step
# (run pod.py get pod_nx'x'ny'.npz')

# run lstm on alpha beta, get model
# (run lstm.py get lstm_nx'x'ny'.h5')

# then run romfom.py 

# to get results run plot.py
