import numpy as np
#import nashpy as nash

def SimplexProj(x):
    u = np.sort(x)[::-1]
    cum = np.cumsum(u)

    rho = -1
    for j in range(len(u)):
        if u[j] + (1-cum[j])/(j+1) > 0:
            rho = j
    tmp = (1 - cum[rho]) / (rho+1)
    y = np.maximum(x + tmp, 0)
    return y / np.sum(y)

def dualGap(G, x, y): 
    loss_x = np.matmul(G, y)
    rew_y = np.matmul(np.transpose(G), x)
    return np.max(rew_y) - np.min(loss_x)

def evalGame(G, x, y):
    return np.inner(x, np.matmul(G, y)) 

"""
def solveGame_nash(G):
    A = G.shape[0]
    B = G.shape[1]

    H = -G    # to apply nashpy, we convert min-max to max-min

    rps = nash.Game(H)
    eqs = rps.lemke_howson_enumeration()
    eqs = list(eqs)[0]
    x = eqs[0]
    y = eqs[1]
    rho = -rps[x,y][0]    # convert back to min-max

    return rho, x, y
"""

def solveGame(G, x_init=None, y_init=None, eps=None, max_iter=None):
    A = G.shape[0]
    B = G.shape[1]
    
    if A == 1:
        rho = np.max(G)
        x = np.array([1])
        y = np.zeros(len(G))
        y[np.argmax(G)] = 1
        return rho, x, y, 0
    if B == 1: 
        rho = np.min(G)
        x = np.zeros(len(G))
        x[np.argmin(G)] = 1
        y = np.array([1])
        return rho, x, y, 0

    if x_init is None:
        x_init = np.ones(A) / A
    if y_init is None:
        y_init = np.ones(B) / B

    x = x_init
    y = y_init

    it = 0
    eta = 0.125 * np.max(np.abs(G)) / np.sqrt(A+B)
    x_sum = np.zeros(A)
    y_sum = np.zeros(B)

    loss_xpre = np.matmul(G, y)
    rew_ypre = np.matmul(np.transpose(G), x)

    while it <= max_iter:
        loss_x = np.matmul(G, y)
        rew_y = np.matmul(np.transpose(G), x)
        x = SimplexProj(x - 2 * eta * loss_x + eta * loss_xpre)
        y = SimplexProj(y + 2 * eta * rew_y - eta * rew_ypre)
        x_sum += x
        y_sum += y
        it += 1 

        last_gap = dualGap(G, x, y)
        avg_gap = dualGap(G, x_sum/it, y_sum/it)

        if avg_gap < eps or last_gap < eps: 
            break

        loss_xpre = np.copy(loss_x)
        rew_ypre = np.copy(rew_y)
      
    if avg_gap < eps or it > max_iter:
        x = x_sum / it
        y = y_sum / it
    rho = np.inner(x, np.matmul(G,y))

    return rho, x, y, it

def Q_from_V(r, p, V, gamma): 
    if len(r.shape)==3:    # MG case
        return r + gamma * np.einsum('sabq,q->sab', p, V)
    elif len(r.shape)==2:  # MDP case
        return r + gamma * np.einsum('saq,q->sa', p, V)


def MGVI(task, r, p, gamma, x=None, y=None, V_init=None, eps=None, max_iter=None, verbose=False, use_nash=False):
    S = r.shape[0]
    A = r.shape[1]
    B = r.shape[2]

    if V_init is None:
        V = np.zeros(S)
    else:
        V = V_init

    if x is None:
        x = np.ones((S,A)) / A
        assert(task=='solve')
    if y is None:
        y = np.ones((S,B)) / B
        assert(task=='solve')

    converge = False
    it = 0
    using_nash_times = 5
    while (not converge) and (it <= max_iter):
        Vpre = np.copy(V)
        Q = r + gamma * np.einsum('sabq,q->sab', p, Vpre)

        max_err = 0
        inner_it_sum = 0
        for s in range(S):
            if task == 'eval':
                V[s] = evalGame(Q[s,:,:], x[s,:], y[s,:])
            elif task == 'solve':
                if using_nash_times > 0 and use_nash:
                    V[s], x[s,:], y[s,:] = solveGame_nash(Q[s,:,:])
                else: 
                    V[s], x[s,:], y[s,:], inner_it = solveGame(Q[s,:,:], x_init=x[s,:], y_init=y[s,:], 
                        eps=0.1*eps*(1-gamma), max_iter=100)
                    inner_it_sum += inner_it

            max_err = np.maximum(max_err, np.abs(V[s] - Vpre[s]))
        it += 1
        converge = (max_err < eps)

        if verbose: 
            print("  iter=", it, "  err=", max_err, "  eps=", eps)

        if using_nash_times > 0:
            using_nash_times -= 1
        elif inner_it_sum / S > 20:
            using_nash_times = 5 

    if task == 'eval':
        return V
    if task == 'solve':
        return V, x, y
    assert(False) 

def MDPVI(task, r, p, gamma, x=None, V_init=None, eps=None, max_iter=None, verbose=False):
    S = r.shape[0]
    y = np.ones((S,1))
    r_exp = np.expand_dims(r, axis=2)  # S*A -> S*A*1
    p_exp = np.expand_dims(p, axis=2)  # S*A*S' -> S*A*1*S' 
    if task == 'eval': 
        return MGVI(task, r=r_exp, p=p_exp, gamma=gamma, 
                x=x, y=y, V_init=V_init, eps=eps, max_iter=max_iter, verbose=verbose)
    elif task == 'solve':
        V, x, _ = MGVI(task, r=r_exp, p=p_exp, gamma=gamma, 
                V_init=V_init, eps=eps, max_iter=max_iter, verbose=verbose)
        return V, x

def MGdualGap(r, p, gamma, x, y, eps=None, max_iter=None, verbose=False):
    ry = np.einsum('sab,sb->sa', r, y)
    py = np.einsum('sabq,sb->saq', p, y)
    V2, _ = MDPVI('solve', ry, py, gamma, eps=eps, max_iter=max_iter)
    rx = np.einsum('sab,sa->sb', r, x)
    px = np.einsum('sabq,sa->sbq', p, x)
    V1_tmp, _ = MDPVI('solve', -rx, px, gamma, eps=eps, max_iter=max_iter)
    V1 = -V1_tmp
    return np.max(V1-V2), np.argmax(V1-V2)


#print(SimplexProj([0.5,0.5,0.1]))
#G = np.array([[1,-1], [-1,1]])
#v = evalGame(G, [0.6, 0.4], [0.6, 0.4])
#print(v)
#v, x, y = solveGame(G, x_init=[0.8, 0.2], y_init=[0.3, 0.7])
#print(v)
#print(x)
#print(y)


