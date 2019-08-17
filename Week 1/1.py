import numpy as np

x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# numpy has a built in operator called logical_xor that can calculate the XOR operation of two arrays.
# This returns a True or False 
y = np.logical_xor(x[:,0], x[:,1])

# convert the returned True/False to integers
y = y.astype(int)

# convert y into a 2-d array (instead of 1) to make matrix multiplication easier.
# this only works in case the output layer is of node count 1.
y = y.reshape(-1,1)

class TwoLayerPerceptron :
    
    def __init__ (self,input_nodes, hidden_nodes, 
                  output_nodes, learning_rate = 0.001) :
        
        # structure of the NN. These variables represent the number of nodes
        # in each of the layers of the NN
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # learning rate
        self.learning_rate = learning_rate
        
        # weights matrix
        # ih = input-> hidden & ho = hidden-> output
        # fill weights with random numbers of mean = 0, sd=2
        self.weights_ih = np.random.normal(loc = 0.0, 
                                           scale = 2,
                                           size = (self.input_nodes, self.hidden_nodes))
        self.weights_ho = np.random.normal(loc = 0.0, 
                                           scale = 2,
                                           size = (self.hidden_nodes, self.output_nodes))        
    def forward_prop(self, record) :
        #input->hidden layer
        weighted_sum  = np.dot(record,self.weights_ih)
        # apply activation function in the hidden layer
        hidden_output = weighted_sum >=0  # this returns a True or False
        self.hidden_output = hidden_output.astype(int)
        
        #hidden->output layer
        weighted_sum  = np.dot(hidden_output,self.weights_ho)
        #apply activation function in the output layer
        output_output = weighted_sum >=0  
        output_output = output_output.astype(int)
        self.y_hat = output_output
    
    def backward_prop(self, record, index) :
        
        # we start off with the error in the output layer(y - y_hat)
        output_error = y[index] - self.y_hat
        
        # update the weights between the hidden layer and output layer
        self.weights_ho += self.learning_rate * np.dot( self.hidden_output.T, output_error) 
        
        # error in the hidden layer.. to T weights_ho or not ??
        hidden_error = np.dot(self.weights_ho, output_error)
        
        # update the weights between the input layer and hidden layer
        self.weights_ih += self.learning_rate * np.dot(record, hidden_error)
        pass

        # this is where training happens
    def fit(self,X,y,epoch = 10) :
        
        for i in range(epoch) :
            # run forward and back prop for each row of data
            for index, record in enumerate(X):
                record = np.array(record,ndmin=2)
                self.forward_prop(record)
                self.backward_prop(record, index)
        pass
    
    # given the input data, predict the output by doing forward_prop        
    def predict(self,X):
        
        y = [] # to hold the predicted output        
        for record in X :
            record = np.array(record, ndmin=2 )
            self.forward_prop(record)
            y.append(self.y_hat)
        return y

nn = TwoLayerPerceptron(2,2,1)
nn.fit(x,y)
y_pred = nn.predict(x)
print ( "x = ",x)
print ( "y = ",y)
print (y_pred)