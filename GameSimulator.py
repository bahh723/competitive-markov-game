from Environment import RandomMG
from Agent import PAgent
import numpy as np
import util
import matplotlib.pyplot as plt
import os



class GameSimulator:
    def __init__(self, env,  T):
        self.env = env
        self.T = T
        self.gamma = self.env.gamma

        eta = 0.125 / np.sqrt(self.env.A + self.env.B)
        self.P1 = PAgent(self.env.S, self.env.A, self.gamma, eta)
        self.P2 = PAgent(self.env.S, self.env.B, self.gamma, eta)
        if not os.path.exists('figure/tmp'):
            os.makedirs('figure/tmp')

    def simulate_full_info(self):
        print("=== offline solving the game ===")
        self.env.solve(eps=0.001, max_iter=np.inf, verbose=True)
        self.P1.reset()
        self.P2.reset()

        for t in range(self.T): 
            x = self.P1.get_policy()
            y = self.P2.get_policy()
            ry = np.einsum('sab,sb->sa',  self.env.r, y)    
            py = np.einsum('sabq,sb->saq', self.env.p, y)
            rx = np.einsum('sab,sa->sb', self.env.r, x)
            px = np.einsum('sabq,sa->sbq', self.env.p, x)
            self.P1.update(ry, py)
            self.P2.update(-rx, px)
            
            if t % 100 == 0: 
                self._output_gap(t)
            if t % 10 == 0:
                self._output_figure(t)
            
    def _output_gap(self, t): 
        x = self.P1.get_policy()
        y = self.P2.get_policy()
        gap, maxgap_state = util.MGdualGap(self.env.r, self.env.p, self.gamma, x, y, eps=0.001, max_iter=np.inf)
        print("  t=",t, "  gap=", gap, "  maxgap_state=", maxgap_state)
       
    def _output_figure(self, t): 
        s = 0
        # calculate Q of the current policies
        x = self.P1.get_policy()
        xs = x[s,:]
        y = self.P2.get_policy()
        ys = y[s,:]

        V = util.MGVI(task='eval', r=self.env.r, p=self.env.p, gamma=self.gamma, 
                x=x, y=y, eps=0.001, max_iter=np.inf)
        Q = util.Q_from_V(r=self.env.r, p=self.env.p, V=V, gamma=self.gamma)
        _, xstar, ystar, _ = util.solveGame(Q[s,:,:], eps=0.001, max_iter=np.inf, 
                x_init=xs, y_init=ys)

        plt.figure(t)
        plt.scatter(xstar[0], ystar[0], c='r') 
        plt.scatter(xs[0], ys[0], c='b')
        plt.scatter(self.env.x[s,0], self.env.y[s,0], c='g')
        plt.xlim((0,1))
        plt.ylim((0,1))

        plt.savefig('figure/tmp/%03d.png' % (t/10))
        plt.close(t)



