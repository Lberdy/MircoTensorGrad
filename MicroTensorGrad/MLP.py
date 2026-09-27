import numpy as np
from Engine import MicroTensor
from typing import Literal

def HE_Initialization(in_ : int, out_ : int):
    return np.random.normal(loc=0, scale=np.sqrt(2/in_), size=(in_, out_))

def create_Batches(X : np.ndarray, Y : np.ndarray, bacthSize : int):
    dataset_size = len(X)
    i = 0
    Batches_X = []
    Batches_Y = []
    while i < dataset_size:
        start = i
        end = i + bacthSize if i + bacthSize < dataset_size else dataset_size
        Batches_X.append(X[start:end])
        Batches_Y.append(Y[start:end])
        
        i += bacthSize
    
    return (Batches_X, Batches_Y)

class Optimizer:
    def __init__(self, lr : float):
        self.lr = lr

class SGD(Optimizer):
        
    def update(self, steps):
        for step in steps:
            if step["optimize"]:
                grad = np.mean(step["content"].grad, axis=0)
                step["content"].array = step["content"].array - self.lr*grad

class MLP:
    def __init__(self, components : list, LossFunction : Literal["MSE", "CrossEntropy"], optimizer : Optimizer):
        
        self.training = True
        
        self.optimizer = optimizer
        
        print(components)
        
        self.steps = []
        for compoenent in components:
            if compoenent[0].lower() == "linear":
                in_ = compoenent[1]
                out_ = compoenent[2]
                print(f"in : {in_}, out : {out_}")
                self.steps.append(
                    {
                        "name" : "matmul",
                        "optimize" : True,
                        "content" : MicroTensor(HE_Initialization(in_, out_))
                    }
                )
                self.steps.append(
                    {
                        "name" : "add",
                        "optimize" : True,
                        "content" : MicroTensor(np.zeros(shape=(out_, 1)))
                    }
                )
            elif compoenent[0].lower() == "relu":
                self.steps.append(
                    {
                        "name" : "RELU",
                        "optimize" : False
                    }
                )
            elif compoenent[0].lower() == "tanh":
                self.steps.append(
                    {
                        "name" : "Tanh",
                        "optimize" : False
                    }
                )
            elif compoenent[0].lower() == "sigmoid":
                self.steps.append(
                    {
                        "name" : "Sigmoid",
                        "optimize" : False
                    }
                )
            else:
                raise ValueError("you can choose only [Linear, Relu, Tanh, Sigmoid] in components")
        
        if LossFunction == "MSE":
            self.steps.append(
                {
                    "name" : "MSE",
                    "optimize" : False
                }
            )
        else:
            self.steps.append(
                {
                    "name" : "CrossEntropy",
                    "optimize" : False
                }
            )
    
    def update(self):
        self.optimizer.update(self.steps)
        
    
    def __call__(self, X : MicroTensor, Y : None | np.ndarray = None):
        topo = []
        topo.append(X)
        for step in self.steps:
            if step["name"] == "matmul":
                other = topo[-1]
                topo.append(step["content"])
                topo.append(step["content"]@other)
            if step["name"] == "add":
                other = topo[-1]
                topo.append(step["content"])
                topo.append(step["content"] + other)
            if step["name"] == "RELU":
                other = topo[-1]
                topo.append(other.RELU())
            if step["name"] == "Tanh":
                other = topo[-1]
                topo.append(other.Tanh())
            if step["name"] == "Sigmoid":
                other = topo[-1]
                topo.append(other.Sigmoid())
                
        if self.training:
            if Y is None:
                raise ValueError("You should pass Y (True labels), if you seeking the network output, set 'training' to False")
            
            if self.steps[-1]["name"] == "MSE":
                other = topo[-1]
                topo.append(other.MSE(Y))
            if self.steps[-1]["name"] == "CrossEntropy":
                other = topo[-1]
                topo.append(other.CrossEntopyLoss(Y))
            
            class TOPO:
                def __init__(self):
                    self.topo = topo
                
                def backward(self):
                    for element in reversed(self.topo):
                        element.backward_()
            
            return TOPO()
        else:
            return topo[-1].array
