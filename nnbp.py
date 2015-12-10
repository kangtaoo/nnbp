# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string
import os

random.seed(0)

def datafile(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'data', filename)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in xrange(I):
        m.append([fill]*J)
    return m

# need to use inverse tangent as transfer function math.atan(x)
# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    # return math.tanh(x)
    return math.atan(x)

# derivative of inverse tangent 1/(1 + x**2) need to write it in terms of output y
# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    # return 1.0 - y**2
    return 1/(1+math.tan(y)**2)

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in xrange(self.nh):
            for k in xrange(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs: expecting {0}, got {1}'.format(self.ni-1, len(inputs)))

        # input activations
        for i in xrange(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in xrange(self.nh):
            _sum = 0.0
            for i in xrange(self.ni):
                _sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(_sum)

        # output activations
        for k in xrange(self.no):
            _sum = 0.0
            for j in xrange(self.nh):
                _sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(_sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in xrange(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in xrange(self.nh):
            error = 0.0
            for k in xrange(self.no):
                error += output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
    #####
    ######Below code need to change the the L2 Regularization Term
        # update output weights
        for j in xrange(self.nh):
            for k in xrange(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in xrange(self.ni):
            for j in xrange(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def test_one(self, pattern):
        return self.update(pattern)

    def weights(self):
        print('Input weights:')
        for i in xrange(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in xrange(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.3, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)

def output_nodes(n):
    output = [0,0,0,0,0,0,0,0,0,0]
    output[n] = 1
    return output

def which_number(output_nodes):
    _max = -1
    ret = 0
    for i, y in enumerate(output_nodes):
        if y > _max:
            _max = y
            ret = i
    return ret

def mnist_demo():
    nn = NN(784, 10, 10)
    
    training_set = []
    with open(datafile('mnist-train.csv'), 'r') as training_data:
        for i, line in enumerate(training_data,1):
            if i > 1000: break
            _list = map(int, line.split(','))
            target = output_nodes(_list.pop(0))
            training_set.append([_list, target])
    
    nn.train(training_set)
    #nn.test(training_set)
    #return
    cnt = 0
    with open(datafile('mnist-test.csv'), 'r') as testing_data:
        for i, line in enumerate(testing_data,1):
            if i > 100: break
            _list = map(int, line.split(','))
            target = output_nodes(_list.pop(0))
            target_num = which_number(target)
            result = nn.test_one(_list)
            result_num = which_number(result)
            correct = target_num == result_num
            if target_num == result_num:
                cnt = cnt + 1
            else:
                cnt = cnt
            print 'target:{0}, result:{1}, {2}'.format(target_num, result_num, correct)
        print float(cnt)/i

if __name__ == '__main__':
    #demo()
    mnist_demo()