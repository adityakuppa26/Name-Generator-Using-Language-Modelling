import numpy as np
from utils import *

data = open("names.txt", 'r').read()
data = data.lower()
chars = sorted(list(set(data)))

chars_to_num = {c:i for i,c in enumerate(chars)}
num_to_chars = {i:c for i,c in enumerate(chars)}

def sample(parameters, chars_to_num):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros((Wax.shape[1],1))
    a_prev = np.zeros((n_a, 1))
    indices=[]
    counter=0
    idx=-1
    new_line_char_index = chars_to_num['\n']
    
    while idx != new_line_char_index and counter != 50:
        a = np.tanh(np.dot(Wax, x)+np.dot(Waa, a_prev)+b)
        y = softmax(np.dot(Wya, a)+by)
        
        idx = np.random.choice(list(chars_to_num.values()), p=y.ravel())
        indices.append(idx)
        x = np.zeros((Wax.shape[1],1))
        x[idx]=1
        a_prev=a
        counter += 1
        
    if counter == 50:
        indices.append(chars_to_num['\n'])
    
    return indices

def optimize(X, a_prev, Y, parameters, lr):
    y_pred, loss, cache = rnn_forward(X, a_prev, Y, parameters)

def model(chars_to_num, num_to_chars, n_a=100, lr=0.01, iterations=2000000, vocab_size=len(chars_to_num), params=None):
    
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # to train model further on trained_params
    if params is not None:    
        parameters=params
    
    with open("names.txt", 'r') as m:
        examples = m.readlines()
    examples = [exmp.lower().strip() for exmp in examples]
    
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a,1))
    
    dWax, dWaa, dWya = np.zeros_like(parameters['Wax']), np.zeros_like(parameters['Waa']), np.zeros_like(parameters['Wya'])
    db, dby = np.zeros_like(parameters['b']), np.zeros_like(parameters['by'])
    batch_grads={}
    batch_losses=[]
    batch_loss=0
    global_loss=0
    
    for it in range(iterations):
        idx = it%len(examples)
        example = examples[idx]
        example_chars = [c for c in example]
        example_chars_to_nums = [chars_to_num[c] for c in example_chars]
        
        X = [None]+example_chars_to_nums
        Y = [X[i+1] for i in range(len(X)-1)] + [chars_to_num['\n']]
        
        y_pred, currentLoss, cache = rnn_forward(X, a_prev, Y, parameters)
        gradients = rnn_backward(X, Y, y_pred, parameters, cache)
        dWax += gradients['dWax']
        dWaa += gradients['dWaa']
        dWya += gradients['dWya']
        dby += gradients['dby']
        db += gradients['db']
        
        batch_loss += currentLoss
        global_loss += currentLoss
        
        if it % 64 == 0:
            batch_loss /= 64
            batch_losses.append(batch_loss)
            batch_loss=0
            batch_grads['dWax'] = dWax/64
            batch_grads['dWaa'] = dWaa/64
            batch_grads['dWya'] = dWya/64
            batch_grads['db'] = db/64
            batch_grads['dby'] = dby/64
            batch_grads = clip_grads(batch_grads, maxVal=5)
            parameters = update_parameters(parameters, batch_grads, lr)
            dWax, dWaa, dWya = np.zeros_like(parameters['Wax']), np.zeros_like(parameters['Waa']), np.zeros_like(parameters['Wya'])
            db, dby = np.zeros_like(parameters['b']), np.zeros_like(parameters['by'])
            batch_grads={}
                        
        if it % 10000 == 0:
            print("Iteration: {}, Loss: {}".format(it, global_loss/10000))
            for name in range(10):
                indices = sample(parameters, chars_to_num)
                display_text = ''.join(num_to_chars[i] for i in indices)
                display_text = display_text[0].upper() + display_text[1:]
                print(display_text)
            global_loss = 0
            
    return parameters

trained_params = model(chars_to_num, num_to_chars)
       