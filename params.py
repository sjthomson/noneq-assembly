import numpy as np


N = 6 # number of particles
fps = 40 # frames per second for video
dt_dim = 0.025 # dimensional time step

# DIMENSIONAL PARAMETERS
#mu = 0.0379 # g/s # assuming underdamped
mu = 0.0433 # g/s # assuming overdamped

U0 = 0.0450 # depth of capillary potential at closest approach, g cm^2/s^2
#U0 = 0.08
lc = 0.2634 # capillary length, cm
#D = 0.4327 # diffusion coefficient
D = 0.1677 # diffusion coefficient
#D = 1

tau = 0.1589 # correlation time for noise
#D = 1
R = 0.079375 # sphere radius, cm
R_star = 0.5*R # effective radius
L = 4.25 # corral side length
#fcorral_height = 1.6396 # corral force at x = 0.5*L (obtained from MATLAB)
fcorral_height = 0.4412


# SPHERE PROPERTIES
rho_s = 2.2 # PTFE density
ms = rho_s*(4/3)*np.pi*R**3
pr = 0.46 # Poisson ratio of PTFE
E = 575e1 # Youngs modulus of PTFE
E_star = E/(2*(1 - pr**2)) # effective Youngs modulus of PTFE
Fk = (4/3)*E_star*np.sqrt(R_star)
mu_f = 0.1 # sphere friction coefficient
kt = 100 # viscous friction coefficient


# DIMENSIONLESS PARAMETERS
f0 = U0/(2*mu*D) # dimensionless force coefficient
R0 = R/lc # dimensionless sphere radius
L0 = L/lc # dimensionless corral side length
k = lc*1e8/(2*mu*D) # dimensionless elastic modulus
t0 = lc**2/(2*D)
dt = dt_dim/t0 # dimensionless time step

cutoff = 2.1*R0 # nearest-neighbour cutoff

