import numpy as np

class Optimizer:
    def __init__(self, lr : float):
        self.lr = lr

class SGD(Optimizer):

    def update(self, steps):
        for step in steps:
            if step["optimize"]:
                grad = np.mean(step["content"].grad, axis=0)
                step["content"].array = step["content"].array - self.lr*grad
                

class Adam(Optimizer):
    def __init__(self, lr : float, beta : float = 0.9, gamma : float = 0.999):
        super().__init__(lr)
        self.m_t = 0
        self.v_t = 0
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 5e-8
        self.t = 0
        
    def reinit(self):
        self.t = 0
        
    def update(self, steps):
        for step in steps:
            if step["optimize"]:
                grad = np.mean(step["content"].grad, axis=0)
                
                m_tp1 = self.beta*self.m_t + (1 - self.beta)*grad
                v_tp1 = self.gamma*self.v_t + (1 - self.gamma)*grad**2
                
                mm_tp1 = m_tp1/(1 - self.beta**(self.t + 1))
                vv_tp1 = v_tp1/(1 - self.gamma**(self.t + 1))
                
                step["content"].array = step["content"].array - self.lr*mm_tp1/(np.sqrt(vv_tp1) + self.epsilon)
        
        self.t += 1