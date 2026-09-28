import numpy as np

class MicroTensor:
    def __init__(self, np_array : np.ndarray):
        self.array = np_array
        self.grad = 0
        self.backward_ = lambda : None
        
    def zero_grad(self):
        self.grad = 0
    
    def __matmul__(self, other):
        out = MicroTensor(self.array.T@other.array)
        
        def backward():
            self.grad += (out.grad@other.array.transpose(0, 2, 1)).transpose(0, 2, 1)
            other.grad += self.array@out.grad
            
        out.backward_ = backward
        
        return out
        
    def __add__(self, other):
        out = MicroTensor(self.array+other.array)
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
                    
        out.backward_ = backward
                
        return out
    
    def RELU(self):
        out = MicroTensor(np.where(self.array < 0, 0, self.array))
        
        def backward():
            self.grad += (out.array > 0)*out.grad
        
        out.backward_ = backward
        
        return out
    
    def Tanh(self):
        out = MicroTensor(np.tanh(self.array))
        
        def backward():
            self.grad += (1 - out.array**2)*out.grad
            
        out.backward_ = backward
        
        return out
    
    def Sigmoid(self):
        sigmoid = lambda x:1/(1 + np.exp(-x))
                    
        out = MicroTensor(sigmoid(self.array))
        
        def backward():
            self.grad += (out.array*(1 - out.array))*out.grad
            
        out.backward_ = backward
        
        return out
    
    def Softmax(self):
        e_x = np.exp(self.array - np.max(self.array, axis=1, keepdims=True))
        return e_x / (np.sum(e_x, axis=1, keepdims=True))
    
    def CrossEntopyLoss(self, Y : np.ndarray):
        softmax = self.Softmax()
        
        out = MicroTensor(np.mean(-np.sum(Y*np.log(softmax.clip(1e-15, 1)), axis=1), axis=0))
        
        def backward():
            self.grad += softmax - Y
            
        out.backward_ = backward
        
        return out
    
    def MSE(self, Y : np.ndarray):
        out = MicroTensor(np.mean(np.sum(((Y - self.array)**2)/2, axis=1), axis=0))
        
        def backward():
            self.grad += self.array - Y
            
        out.backward_ = backward
        
        return out
    
    def __repr__(self):
        return f"array :\n{self.array},\ngrad :\n{self.grad})"