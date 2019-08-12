import numpy as np

x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y   = np.array([0,1,1,1])

w = np.random.normal(2)

print ( "weights are ",w)

# learning rate
alpha = 0.0001
print( w)

def forward_prop(x,w) :
    y_hat = np.dot(x,w)
    if y_hat > 0 :
        return 1
    else :
        return 0

def backward_prop(y_hat, y, x, w) :
    w[0] = w[0] + alpha * (y - y_hat) * x[0]
    w[1] = w[1] + alpha * (y - y_hat) * x[1]
    return w

# number of epochs
for epoch in range(20) :
    
    # for each row in x
    for row in range(x.shape[0]) :
        
        # for each row in x, predict y_hat
        y_hat = forward_prop(x[row],w)

        # for each row calculate weights
        w = backward_prop(y_hat,y[row],x[row],w)

print ( w)



