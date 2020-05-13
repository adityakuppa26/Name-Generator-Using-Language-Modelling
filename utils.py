import numpy as np

def softmax(x):
    z = x - np.max(x)
    return np.exp(z)/np.sum(np.exp(z), axis=0)

def clip_grads(gradients, maxVal):    
    for key, gradient in gradients.items():
        gradients[key] = np.clip(gradient, -maxVal, maxVal)
    return gradients
        
def initialize_parameters(n_a, n_x, n_y):
    Wax = np.random.randn(n_a, n_x)*0.01
    Waa = np.random.randn(n_a, n_a)*0.01
    Wya = np.random.randn(n_y, n_a)*0.01
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))

    return {'Wax':Wax, 'Waa':Waa, 'Wya':Wya, 'by':by, 'b':b}

def rnn_forward(X, a_prev, Y, parameters):
    x={}
    a={}
    y_pred={}
    vocab_size=27
    a[-1]=a_prev
    loss=0
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if X[t]!=None:
            x[t][X[t]]=1
        
        a[t] = np.tanh(np.dot(Wax, x[t]) + np.dot(Waa, a[t-1]) + b)
        y_pred[t] = softmax(np.dot(Wya, a[t]) + by)
        loss -= np.log(y_pred[t][Y[t]])       
    cache = (x, a)
    
    return y_pred, loss, cache

def rnn_backward(X, Y, y_pred, parameters, cache):
    gradients = {}
    (x, a) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    for t in reversed(range(len(X))):
        dy = np.copy(y_pred[t])
        dy[Y[t]] -= 1
        gradients['dWya'] += np.dot(dy, a[t].T)
        gradients['dby'] += dy
        da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] 
        dz = (1 - a[t] * a[t]) * da 
        gradients['db'] += dz
        gradients['dWax'] += np.dot(dz, x[t].T)
        gradients['dWaa'] += np.dot(dz, a[t-1].T)
        gradients['da_next'] = np.dot(parameters['Waa'].T, dz)
        
    return gradients

def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    
    return parameters
       