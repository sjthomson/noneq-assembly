import numpy as np
import params as pm

def initialBeads(n, R, str):
    if n == 1:
        return np.zeros((n, 3))
    
    if str == 'c':
        return np.array([[-2*R, 0, 0],[0, 0, 0], [2*R, 0, 0], [-R, np.sqrt(3)*R, 0], [R, np.sqrt(3)*R, 0], [-R, -np.sqrt(3)*R, 0]])
    elif str == 't':
        return np.array([[-2*R, 0, 0],[0, 0, 0], [2*R, 0, 0], [-R, np.sqrt(3)*R, 0], [R, np.sqrt(3)*R, 0], [0, 2*np.sqrt(3)*R, 0]])
    else:
        return np.array([[-2*R, 0, 0],[0, 0, 0], [2*R, 0, 0], [-R, np.sqrt(3)*R, 0], [R, np.sqrt(3)*R, 0], [3*R, np.sqrt(3)*R, 0]])


# map positions onto periodic domain
def periodicLocation(x, L):
    
    x[x < -0.5*L] += L
    x[x >= 0.5*L] -= L
    return x


def periodicDifferences(dX, L):
 
    dX[dX > 0.5*L] -= L
    dX[dX <= -0.5*L] += L
    return dX

#def computeDiffs(X, X0, V, L, dt, cutoff, R):
def computeDiffs(X, X0, cutoff, dt):
    X = np.reshape(X, (pm.N, 2)) # Silly (maybe unncessary) reshape due to accepted input of "root solver"
    X0 = np.reshape(X0, (pm.N, 2))
    #Pairwise difference vectors between beads, then compute norm to get distances
    # Use this to normalize vectors and then get nearest neighbour matrix
    #X_diffp = periodicDifferences(X[:, np.newaxis, :] - X[np.newaxis, :, :], L) # for periodic
    X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    pd_beads = np.sqrt(np.sum(X_diff ** 2, axis=-1))
    #pd_beads[pd_beads == 0] = 1
    pd_beads = np.where(pd_beads == 0, 1, pd_beads)
    X_diff_norm = X_diff/pd_beads[:, :, np.newaxis]
    neighbourDist = pd_beads < cutoff
    #overlap = 2*R - pd_beads

    V = (X - X0)/dt # new velocity
    dV = V[:, np.newaxis, :] - V[np.newaxis, :, :] # new relative velocity
    dV_t = dV - np.sum(dV * X_diff_norm, axis = -1)[:, :, np.newaxis]*X_diff_norm
    #dV_t_d = np.linalg.norm(dV_t, axis = -1, keepdims = True)
    dV_t_d = np.sqrt(np.sum(dV_t ** 2, axis = -1))
    #dV_t_d[dV_t_d == 0] = 1
    dV_t_d = np.where(dV_t_d == 0, 1, dV_t_d)
    tangents = dV_t/dV_t_d[:, :, np.newaxis]


    #dV0_t = np.where(neighbourDist[:, :, np.newaxis], dV0_t + dt*dV_t, 0.0)

    return X_diff_norm, pd_beads, neighbourDist, tangents, dV_t

    