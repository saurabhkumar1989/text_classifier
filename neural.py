import sys
total = 5000

from fileread.TweetRead import importData
import numpy as np
from bagofwords.textToVector import bagofWords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

X, y = importData()
X = X[1:]
y = y[1:]
X, y = shuffle(X, y, random_state=0)

#X = np.array(X)
#vectorizer = CountVectorizer(min_df=2)
#X = vectorizer.fit_transform(X)

X = np.array(bagofWords(X))

y = np.array(map(int,list(y)))
num_examples =len(X)

ratio = .6
# training data
X_train = X[1:num_examples*ratio]
y_train = y[1:num_examples*ratio]

#test data
X_test = X[num_examples*ratio:]
y_test = y[num_examples*ratio:]


barLength = 10 
status = ""


nn_input_dim = len(X[0])#neural netwoek input dimension
nn_output_dim = 2# true or false
neuron_number = 6 # in a layer
# Gradient descent parameters (I picked these by hand)
alpha = 0.001 # learning rate for gradient descent
reg_lambda = 0.001 # regularization strength
num_passe = 10000
def weight_init(L1,L2):
    return np.sqrt(6)/np.sqrt(L1 + L2) 

def calculate_loss(model):
    W1,  W2 = model['W1'], model['W2']
    # Forward propagation to calculate our predictions
    num_ex = len(X_train)
    z1 = X_train.dot(W1)
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) 
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_ex), y_train])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_ex * data_loss
def status(i,model):
    progress = (float(i)/num_passe)
    block = int(round(barLength*progress))
    sys.stdout.write('\r')
    text = "[{0}] {1}% Completed.".format( "#"*block + "-"*(barLength-block), format(progress*100,".2f"),status)
    sys.stdout.write(text)
    sys.stdout.write ("           Current Loss %.5f." %(calculate_loss(model)))
    sys.stdout.flush()
    
    
def predict(model, x):
    W1,  W2 = model['W1'],  model['W2']
    # Forward propagation
    z1 = x.dot(W1) 
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) 
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, keepdims=True)
    return np.argmax(probs)

def build_model(nn_hdim, num_passes=num_passe, print_loss=False):
    #Initilization of weight
    #L1 number of input in the given layer
    #L2 number of input in the given layer
    L1 = nn_input_dim
    L2 = neuron_number
    esp_init = weight_init(L1,L2)
    W1 = np.random.uniform(-esp_init,esp_init,[nn_input_dim,nn_hdim])
    L1 = neuron_number
    L2 = nn_output_dim
    esp_init = weight_init(L1,L2)
    W2 = np.random.uniform(-esp_init,esp_init,[nn_hdim, nn_output_dim])
    # This is what we return at the end
    model = {}
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
        # Forward propagation
        z1 = X_train.dot(W1) 
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) 
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        num_ex = len(X_train)
        # Backpropagation
        delta3 = probs
        delta3[range(num_ex), y_train] -= 1
        dW2 = (a1.T).dot(delta3)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))# diff of tanh -- in future i will use gradient desent to calculate this
        dW1 = np.dot(X_train.T, delta2)
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 =dW2 + (reg_lambda * W2)
        dW1 = dW1 + (reg_lambda * W1)
        # Gradient descent parameter update
        W1 = W1 -(alpha * dW1)
        W2 = W2 -(alpha * dW2)     
        # Assign new parameters to the model
        model = { 'W1': W1, 'W2': W2}
        status(i,model)        
    return model
    
# Build a model
def test():
    model = build_model(neuron_number, print_loss=True)
    test = "Heavy traffic at vest avenue"
#predict(model, x)