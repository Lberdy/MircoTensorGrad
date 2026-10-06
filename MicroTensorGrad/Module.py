from nn import NN
from Engine import MicroTensor
from copy import deepcopy

class Module:
    def __init__(self, components : list[NN]):
        self.components = components
            
    def zero_grad(self):
        for component in self.components:
            component.zero_grad()
            
    def BackUp_Weights(self):
        weightsBackUp = []
        for component in self.components:
            weightsBackUp.append(deepcopy(component.weights))
        
        return weightsBackUp
            
    def Restore_Weights(self, BackUp : list):
        for weights, component in zip(BackUp, self.components):
            component.weights = weights
    
    def __call__(self, X : MicroTensor):
        topo : list[MicroTensor] = []
        topo.append(X)
        for component in self.components:
            component.forward(topo)
            
        if MicroTensor.training:
            return topo
        else:
            return topo[-1].array
