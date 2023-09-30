import Environment
import util
import numpy as np

class PAgent:   # Decentralized Policy-based Agent
    def __init__(self, S, A, gamma, eta, critic=True, opt=True):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.eta = eta
        self.H = 1/(1-gamma)
        self.critic = critic
        self.opt = opt
        self.reset()

    def reset(self):
        self.V = np.zeros(self.S)
        #self.x = np.ones((self.S, self.A)) / self.A
        #self.xp = np.ones((self.S, self.A)) / self.A
        self.x = np.random.rand(self.S, self.A)
        tmp = np.expand_dims(np.sum(self.x, axis=1), axis=1)
        self.x /= tmp
        self.xp = np.copy(self.x)
        self.t = 1

    def _alpha(self, t):
        return (self.H+1)/(self.H + t)

    def update(self, ry, py): 
        # ry(s,a) = sum_b r(s,a,b) * y_t(s,b)
        # py(s,a,s') = sum_b p(s,a,b,s') * y_t(s,b)
        if self.critic: 
            loss = util.Q_from_V(r=ry, p=py, V=self.V, gamma=self.gamma)
            rho = np.einsum('sa,sa->s', self.x, loss)
            self.V += self._alpha(self.t) * (rho - self.V)
            self._update_policy(loss)
        else:
            self.V = util.MDPVI(task='eval', r=ry, p=py, gamma=self.gamma, x=self.x, V_init=self.V, 
                    eps=0.001, max_iter=np.inf)
            loss = util.Q_from_V(r=ry, p=py, V=self.V, gamma=self.gamma)
            self._update_policy(loss)
            
        self.t += 1

    def _update_policy(self, loss):
        for s in range(self.S):
            self.xp[s,:] = util.SimplexProj(self.xp[s,:] - self.eta * loss[s,:])
            if self.opt: 
                self.x[s,:] = util.SimplexProj(self.xp[s,:] - self.eta * loss[s,:])
            else:
                self.x[s,:] = self.xp[s,:]

    def get_policy(self):
        return self.x


