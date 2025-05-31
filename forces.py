import numpy as np
import geom as gm
from scipy.special import kv
import params as pm

def G(r):
    return pm.fcorral_height*np.exp((r - 0.5*pm.L)/pm.lc) # LSA approximation to sphere/wall interaction
    

def corralForce(r, p=10):
    norm_r_p = np.linalg.norm(r, ord=p, axis=1, keepdims=True)
    norm_r_p = np.where(norm_r_p > 0.0, norm_r_p, 1.0)
    force_magnitude = G(norm_r_p)
    direction = (np.abs(r) ** (p - 1) * np.sign(r)) / (norm_r_p ** (p - 1))
    
    return -(force_magnitude * direction).flatten()

def contactForces(pd_beads, X_diff_norm, tangent, neighbourDist, dV0_t):
    n = 2**8
    pd = pd_beads[:, :, np.newaxis]
    
    Fr = (pm.U0/pm.lc)*kv(1, pd/pm.lc) - (2*pm.R*pm.U0/pm.lc)*kv(1, pd/pm.lc)*(1/pd)*(2*pm.R/pd)**n
    
    capillaryForce = -Fr*X_diff_norm # all pairwise forces
    

    friction = -pm.kt*dV0_t # note: don't be confused by dV0_t -- this can be either dV0_t or dXi_t depending on the input into fn in main.py

    cf_mag = np.sqrt(np.sum(capillaryForce**2, axis = -1)) # magnitude of capillary force
    friction_mag = np.sqrt(np.sum(friction**2, axis = -1)) # magnitude of tangential component of capillary force

    #fric_mag = np.where(friction < pm.mu_f*cf_mag, cft_mag, pm.mu_f*cf_mag)
    #coulomb = np.where(neighbourDist[:, :, np.newaxis], -fric_mag[:, :, np.newaxis]*tangent, 0.0)

    coulomb = np.where(friction_mag[:, :, np.newaxis] <= pm.mu_f*cf_mag[:, :, np.newaxis], friction, -pm.mu_f*cf_mag[:, :, np.newaxis]*tangent)

    coulomb = np.where(neighbourDist[:, :, np.newaxis], coulomb, 0.0) #if spheres are not in contact, zero out friction
    
    totalForce = np.sum(capillaryForce, axis = 1)

    
    return totalForce




    
