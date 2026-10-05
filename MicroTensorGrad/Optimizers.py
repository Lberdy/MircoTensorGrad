import numpy as np
from Module import Module

class Optimizer:
    def __init__(self, model : Module, lr : float):
        self.components = model.components
        self.lr = lr

class SGD(Optimizer):
    def __init__(self, model : Module, lr : float):
        super().__init__(model, lr)

    def update(self):
        for component in self.components:
            for weights in component.weights:
                grad = np.mean(weights.grad, axis=0)
                weights.array = weights.array - self.lr*grad
                

class Adam(Optimizer):
    def __init__(self, model : Module, lr : float, beta : float = 0.9, gamma : float = 0.999):
        super().__init__(model, lr)
        self.m_t = 0
        self.v_t = 0
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 5e-8
        self.t = 0
        
    def reinit(self):
        self.t = 0
        
    def update(self):
        for component in self.components:
            for weights in component.weights:
                grad = np.mean(weights.grad, axis=0)
                
                m_tp1 = self.beta*self.m_t + (1 - self.beta)*grad
                v_tp1 = self.gamma*self.v_t + (1 - self.gamma)*grad**2
                
                mm_tp1 = m_tp1/(1 - self.beta**(self.t + 1))
                vv_tp1 = v_tp1/(1 - self.gamma**(self.t + 1))
                
                weights.array = weights.array - self.lr*mm_tp1/(np.sqrt(vv_tp1) + self.epsilon)
        
        self.t += 1