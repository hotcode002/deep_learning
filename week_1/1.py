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
                  output_nodes, learning_rate = 0.01) :
        
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

        # bias matrix
        # h = hidden, o = output (input layer doesn't require biases)
        self.bias_h = np.random.normal(loc = 0.0, scale = 2,
                                       size = (self.hidden_nodes,1) )
        self.bias_o = np.random.normal(loc = 0.0, scale = 2,
                                       size = (self.output_nodes,1) )      

    def sigmoid (self,z) :
        return 1/(1 + np.exp(-z))
    
    def sigmoid_der (self,z) :
        return sigmoid(z) * (1 - sigmoid(z))

    def forward_prop(self, record) :
        #input->hidden layer
        weighted_sum  = np.dot(record,self.weights_ih) + self.bias_h.T
        # apply activation function in the hidden layer
        # hidden_output = weighted_sum >=0  # this returns a True or False
        # self.hidden_output = hidden_output.astype(int)
        self.hidden_output = self.sigmoid(weighted_sum)
        
        #hidden->output layer
        weighted_sum  = np.dot(self.hidden_output,self.weights_ho) + self.bias_o.T
        #apply activation function in the output layer
        # output_output = weighted_sum >=0  
        # output_output = output_output.astype(int)
        self.output_output = self.sigmoid(weighted_sum)
        self.y_hat = self.output_output
    
    def backward_prop(self, record, index) :
        
        # we start off with the error in the output layer(y - y_hat)
        output_error = y[index] - self.y_hat
        
        # update the weights between the hidden layer and output layer
        self.weights_ho += self.learning_rate * np.dot( self.hidden_output.T, 
                                                        output_error * (self.output_output) * (1 - self.output_output) ) 

        # update the bias in the output layer
        self.bias_o += self.learning_rate * output_error
        
        # error in the hidden layer.. to T weights_ho or not ??
        hidden_error = np.dot(self.weights_ho, output_error)
        
        # update the weights between the input layer and hidden layer
        self.weights_ih += self.learning_rate * np.dot(record, 
                                                       hidden_error * self.hidden_output * (1 - self.hidden_output))
        
        # update the bias in the hidden layer
        self.bias_h += self.learning_rate * hidden_error       


    # this is where training happens
    def fit(self,X,y,epoch = 1000) :
        
        for i in range(epoch) :
            # run forward and back prop for each row of data
            for index, record in enumerate(X):
                record = np.array(record,ndmin=2)
                self.forward_prop(record)
                self.backward_prop(record, index)
                print ( self.weights_ho)
                # print ( self.weights_ho)
        pass
    
    # given the input data, predict the output by doing forward_prop        
    def predict(self,X):
        
        y = np.empty([4,1]) # to hold the predicted output        
        for index,record in enumerate(X) :
            record = np.array(record, ndmin=2 )
            self.forward_prop(record)
            y[index,0] = self.y_hat[0,0]
        return y

nn = TwoLayerPerceptron(2,3,1)
nn.fit(x,y)
y_pred = nn.predict(x)
print ( "x = ",x)
print ( "y = ",y)
print ("y_pred = "  ,y_pred)