class SGD:
    '''
    Stochastic Gradient Descent
    '''
    def __init__(self, lr=0.1):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]