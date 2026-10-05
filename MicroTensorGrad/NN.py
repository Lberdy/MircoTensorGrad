from Engine import MicroTensor
import numpy as np

def HE_Initialization(in_ : int, out_ : int):
    return np.random.normal(loc=0, scale=np.sqrt(2/in_), size=(in_, out_))

class NN:
    def __init__(self):
        self.weights = []
    
    def forward(self, Topo : list[MicroTensor]):
        pass
    
    def zero_grad(self):
        for weights in self.weights:
            weights.zero_grad()

class Linear(NN):
    def __init__(self, in_ : int, out_ : int):
        super().__init__()
        self.weights = [
            MicroTensor(HE_Initialization(in_, out_)),
            MicroTensor(np.zeros(shape=(out_, 1)))
        ]
    
    def forward(self, Topo : list[MicroTensor]):
        Topo.append(self.weights[0]@Topo[-1])
        Topo.append(Topo[-1] + self.weights[1])

class BatchNorm(NN):
    def __init__(self, in_ : int):
        super().__init__()
        
        self.weights = [
            MicroTensor(np.ones(shape=(in_, 1))),
            MicroTensor(np.zeros(shape=(in_, 1)))
        ]
    
    def forward(self, Topo : list[MicroTensor]):
        Topo.append(Topo[-1].BatchNorm())
        Topo.append(self.weights[0]*Topo[-1])
        Topo.append(Topo[-1] + self.weights[1])

class RELU(NN):
    def __init__(self):
        super().__init__()
    
    def forward(self, Topo : list[MicroTensor]):
        Topo.append(Topo[-1].RELU())

class Tanh(NN):
    def __init__(self):
        super().__init__()
        
    def forward(self, Topo : list[MicroTensor]):
        Topo.append(Topo[-1].Tanh())

class Sigmoid(NN):
    def __init__(self):
        super().__init__()
        
    def forward(self, Topo : list[MicroTensor]):
        Topo.append(Topo[-1].Sigmoid())