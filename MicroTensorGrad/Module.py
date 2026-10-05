from NN import NN
from Engine import MicroTensor
import numpy as np

class Module:
    def __init__(self, components : list[NN]):
        
        self.components = components
            
    def zero_grad(self):
        for component in self.components:
            component.zero_grad() 
    
    def __call__(self, X : MicroTensor):
        topo : list[MicroTensor] = []
        topo.append(X)
        for component in self.components:
            component.forward(topo)
            
        if MicroTensor.training:
            return topo
        else:
            return topo[-1].array
