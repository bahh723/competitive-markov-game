import numpy as np
import util

class Environment:
    def __init__(self, S, A, B, gamma):
        self.S = S 
        self.A = A 
        self.B = B
        self.gamma = gamma

    def sample(self, s, a, b): 
        R = np.random.binomial( 1, self.r[s,a,b] )
        sp = np.random.multinomial( 1, self.p[s,a,b,:] )
        return R, sp

    def sample_init(self):
        return np.random.multinomial(1, self.p1)

    def solve(self, eps=None, max_iter=None, verbose=False):
        self.V, self.x, self.y = util.MGVI(task='solve', r=self.r, p=self.p, gamma=self.gamma, 
                eps=eps, max_iter=max_iter, verbose=verbose)

        self.Q = self.r + self.gamma * np.einsum('sabq,q->sab', self.p, self.V)
        

class RandomMG(Environment): 
    def __init__(self, S, A, B, gamma):
        super().__init__(S, A, B, gamma)
        self.r = np.random.rand(S, A, B)
        self.p = np.random.rand(S, A, B, S)
        self.p /= np.expand_dims(np.sum(self.p, 3), axis=3)   # normalize
        self.p1 = np.random.rand(S)   # initial state distirbution
        self.p1 /= np.sum(self.p1) 

class MatchingPennies(Environment):
    def __init__(self):
        super().__init__(S=1, A=2, B=2, gamma=0.95)
        self.r = np.zeros((1,2,2))
        self.r[0,:,:] = np.array([[1,-1], [-1,1]])
        self.p = np.ones((1,2,2,1))
        self.p1 = np.array([1])

#env = RandomMG(3, 4, 5, 3)
#print(np.sum(env.p[0,1,0,:]))
#print(env.r[2,2,2])
