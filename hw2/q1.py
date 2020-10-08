import numpy as np
EPOCH = 100
def dev_w(X, y, t, delta):
    return np.mean(np.where(np.abs(y-t)<=delta, np.multiply(X,y-t), np.where((y-t)>=0, X*delta, -X*delta)),0).reshape((X.shape[1],1))
def dev_b(y, t, delta):
    return np.mean(np.where(np.abs(y-t)<=delta, y-t, np.where((y-t)>=0, np.full((y.size,1),delta), np.full((y.size,1),-delta))),0)
def train_huber(X, targets, delta):
    lr = 0.01 ## learning rate

    '''Input : X, targets [data and outcome], delta value
       Output : w,b 
    '''
    targets = targets.reshape((X.shape[0],1))
    w = np.zeros((X.shape[1],1))
    b = np.zeros(1)
    for i in range(EPOCH):
        y = np.matmul(X,w) + b
        w = w - lr*dev_w(X,y,targets,delta)
        b = b - lr*dev_b(y,targets,delta)
    return w,b
    

# Ground truth weights and bias used to generate toy data
w_gt = [1, 2, 3, 4]
b_gt = 5

# Generate 100 training examples with 4 features
X = np.random.randn(100, 4)
targets = np.dot(X, w_gt) + b_gt
#X = np.ones((100,4))
#targets = np.zeros((100,1))
#y = np.linspace(-50,49,100).reshape((100,1))
#delta = 20
#print (X)
#print (y)
#print ("dev w is ", dev_w(X,y,targets,delta))
#print ("dev b is ", dev_b(y,targets,delta))
# Gradient descent
w, b = train_huber(X, targets, delta=2)
print ("w is ", w, "b is ", b)

