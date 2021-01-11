import numpy as np


# define softmax function
def softmax(x):
    if x.ndim == 2:
        x = np.exp(x)
        x = np.divide(x, x.sum(axis=1, keepdims=True) + 1e-7)
    elif x.ndim == 1:
        x = np.exp(x)
        x = np.divide(x, np.sum(x) + 1e-7)
        
    return x

# define cross entropy
def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    
    return -(1/batch_size) * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))

# Define FC layer
class Dense():
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        w = self.params[0]
        out = np.dot(x, w)
        self.x = x
        
        return out
    
    def backward(self, dout):
        w = self.params[0]
        dx = np.dot(dout, w.T) #dx = dy * W^T
        dw = np.dot(self.x.T, dout) #dw = x^T * dx
        self.grads[0][...] = dw
        
        return dx
    
# define softmax with cross entropy layer
class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.t = None
        self.y = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        loss = cross_entropy(self.y, self.t)
        
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = self.y.copy() #softmax output
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size
        
        return dx