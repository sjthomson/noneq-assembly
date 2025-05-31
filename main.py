import numpy as np
import params as pm
import geom as gm
import forces as frce
import argparse
from pathlib import Path
import os

from scipy.io import savemat
from scipy.optimize import root

import psutil
import time
import threading

def monitor_cpu_usage():
    p = psutil.Process()
    while True:
        # Get per-CPU usage across all CPUs
        per_cpu = psutil.cpu_percent(interval=1, percpu=True)
        print(f"Per-CPU usage: {per_cpu}")
        print(f"Process CPU Percent: {p.cpu_percent()}%, Num Threads: {p.num_threads()}")
        time.sleep(2)

inc = 0.005
dt = inc*pm.dt_dim
Tend = 4000
tspan = np.arange(dt, Tend + dt, dt)

#pos = np.zeros((pm.N, 2, len(tspan) + 1))
pos = np.zeros((pm.N, 2, len(np.arange(0, tspan[-1], pm.dt_dim)) + 1))

X0 = pm.R*np.array([[-2, 0], [0, 0], [2, 0], [-1, np.sqrt(3)], [1, np.sqrt(3)], [-1, -np.sqrt(3)]]).flatten()

dXi_t = np.zeros((pm.N,pm.N,2)) #shear displacement

k = 0
pos[:, :, k] = np.reshape(X0, (pm.N, 2))

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--D", type=float, required=True, help="Value of D parameter")
parser.add_argument("--tau", type=float, required=True, help="Value of tau parameter")
args = parser.parse_args()

D = args.D
tau = args.tau

def fn(X, *args):
    X0 = args[0]
    dXi_t = args[1]
    noise = args[2]
    X_diff_norm, pd_beads, neighbourDist, tangents, dV_t = gm.computeDiffs(X, X0, 2*pm.R, dt) 
    
    dXi_t += dt*dV_t.flatten() #update shear displacement - note we don't zero out anything here. This is done in contactForces

    totalForce = frce.contactForces(pd_beads, X_diff_norm, tangents, neighbourDist, np.reshape(dXi_t, (pm.N, pm.N, 2))).flatten() # replace dXi_t with dV_t for local friction rule i.e. relative velocity rather than shear displacement
    return X - X0 - (dt/pm.mu)*(totalForce + frce.corralForce(np.reshape(X0, (pm.N, 2)))) - dt*noise
    
j = 0.0

u = 0.0*np.random.rand(2*pm.N,)

C = np.full((pm.N, pm.N), 0.0)  # Create an n x n matrix filled with 'a'
np.fill_diagonal(C, 1)
L = np.linalg.cholesky(C)


for i in tspan:
    #print(i)
    j += 1.0
    
    nn = (L*np.random.randn(pm.N, pm.N)).flatten()
    u = u*(1 - (dt/tau)) + (np.sqrt(2*D*dt)/tau)*np.random.randn(2*pm.N,)
    
    X = root(fn, X0, args=(X0, dXi_t.flatten(), u))
    
    X_diff_norm, pd_beads, neighbourDist, tangents, dV_t = gm.computeDiffs(np.reshape(X.x, (pm.N, 2)), np.reshape(X0, (pm.N, 2)), 2*pm.R, dt)
    dXi_t += dt*dV_t 
    dXi_t = np.where(neighbourDist[:, :, np.newaxis], dXi_t, 0.0) #zero out non-contact spheres

    X0 = X.x

    if j % (1/inc) == 0:
        
        k+=1
        pos[:, :, k] = np.reshape(X.x, (pm.N, 2))
        #print(i)
    
max_lag = 400 #10/dt_dim
final_frame = np.shape(pos)[2]
dd = final_frame - max_lag

D2 = np.zeros((pm.N, max_lag, dd + 1))

for i in np.arange(0, dd + 1):
    X0 = pos[:, :, i]
    dX = pos[:, :, i:(i + max_lag)] - X0[:, :, np.newaxis]

    D2[:, :, i] = np.sum(dX**2, 1)
    
msd_particle = np.mean(D2, 2)
msd_mean = np.mean(msd_particle, 0)

folder_name = "param_sweep"
output_dir = Path(f"/user/work/wl21287/{folder_name}")

output_dir.mkdir(parents=True, exist_ok=True)

file_path = output_dir / f"tau_{tau:.4f}_D_{D:.4f}.mat"

savemat(file_path, {'pos': pos, 'dt': dt, 'D': D, 'tau': tau, 'msd_particle': msd_particle, 'msd_mean': msd_mean})

