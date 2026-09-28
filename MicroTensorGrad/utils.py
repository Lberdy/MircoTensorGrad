import numpy as np

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