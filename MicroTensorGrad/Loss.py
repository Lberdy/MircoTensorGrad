from Engine import MicroTensor
import numpy as np

class LossFunction:
    def __init__(self, Topo : list[MicroTensor], Y : np.ndarray):
        self.topo = Topo
        self.Y = Y
    
    def backward(self):
        for element in reversed(self.topo):
            element.backward_()

class MSE(LossFunction):
    def __init__(self, Topo : list[MicroTensor], Y : np.ndarray):
        super().__init__(Topo, Y)
        self.topo.append(self.topo[-1].MSE(self.Y))

class CrossEntropy(LossFunction):
    def __init__(self, Topo : list[MicroTensor], Y : np.ndarray):
        super().__init__(Topo, Y)
        self.topo.append(self.topo[-1].CrossEntopyLoss(self.Y))