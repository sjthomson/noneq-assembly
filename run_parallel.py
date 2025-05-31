import subprocess
import multiprocessing
import params as pm
import numpy as np

def run_instance(tau_value):
    subprocess.run(["python", "main.py", "--tau", str(tau_value)])

if __name__ == "__main__":
    tau_values = pm.tau*np.array([1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.0])  # Define the range of D values
    
    with multiprocessing.Pool(processes=len(tau_values)) as pool:
        pool.map(run_instance, tau_values)