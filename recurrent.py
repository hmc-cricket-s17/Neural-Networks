# backprop.py
# author: Shota Yasunaga
# date: 31 January 2016

# description:
# FFnet is a basic multilevel neural network class
# The number of layers is arbitrary.
# Each layer has a function, as specified using the class ActivationFunction.
# Each layer has its own learning rate.

import math
import random
from random import shuffle


class RTRLnet:
    def __init__(nn, name, numNeurons, neuronType, learningRate, pattern, numSlots, runLength):
        """ Feedforward Neural Network                                    """
        """ nn is the 'self' reference, used in each method               """
        """ name is a string naming this network.                         """
        """ size is a list of the layer sizes:                            """
        """     The first element of size is understood as the number of  """
        """         inputs to the network.                                """
        """     The remaining elements of size are the number of neurons  """
        """         in each layer.                                        """
        """     Therefore the last element is the number of outputs       """
        """         of the network.                                       """
        """ function is a list of the activation functions in each layer. """
        """ deriv is a list of the corresponding function derivatives.    """
        """ rate is a list of the learning rate for each layer.           """

        nn.name = name
        nn.pattern = pattern
        nn.numInput = len(pattern)
        nn.numSlots = numSlots
        # Treated input and recurrent input as different layers
        # The third layer is the recurrent part 
        # Added 1 to incorporate bias with in the input
        nn.size = [nn.numInput+numNeurons + 1, numNeurons]
        nn.output = [[0 for neuron in range(s)] # output values for all layers, 
                        for s in nn.size]     # counting the input as layer 0
        nn.recurrent = [0 for neurons in range(nn.size[1])]

        # dummy is used because the input layer does not have weights
        # but we want the indices to conform.
        nn.function = neuronType
        nn.rate = learningRate
        # initialize weights and biases
        nn.weight = [[randomWeight() for output in range(nn.size[1])]
                                     for layer in range(nn.size[0])]

        # Change the weight for the output so that it doesn't use everything
        nn.sensitivity = [0 for out in range(nn.size[1])]
        nn.weight_change = [[0  for output in range(nn.size[1])]
                               for layer in range(nn.size[0])]

        nn.act = [0 for neuron in range(nn.size[1])]
        nn.dif_out = [[[randomWeight() for next_output in range(nn.size[1])]
                                      for inp in range(nn.size[0])]
                                      for output in range(nn.size[1])]
        # The first half of the equation to compute the value of differential
        # Equations on the previous output with respect to w_ij
        nn.fksk = [0 for neuron in range(nn.size[1])]

        # Indication of which phase the machine is in terms of slot
        nn.phase = 0
        nn.runLength = runLength

    def describe(nn, noisy):
        """ describe prints a description of this network. """
        print "---------------------------------------------------------------"
        print "network", nn.name + ":"
        print "size =", nn.size
        print "function =", nn.function.name
        print "learning rate =", nn.rate
        if noisy:    
            print "weight =", roundall(nn.weight[1:], 3)


    def forward(nn):
        """ forward runs the network, given an input vector. """
        """ All act values and output values are saved.      """
        """ The output of the last layer is returned as a    """
        """ convenience for later testing.                   """
        """ Also updates nn.fksk                             """

        input = nn.output[0]  # set input layer
        fun = nn.function.fun
        # Iterate over all neurons in all layers.
        for out in range(nn.size[1]):
            weight_out = [weight[out] for weight in nn.weight]
            nn.act[out] = inner(weight_out, input)
            nn.output[1][out] = fun(nn.act[out])

        difq = nn.function.deriv

        for neuron in range(nn.size[1]):
            nn.fksk[neuron] = difq(nn.act[neuron], nn.output[1][neuron])

        
    def updateDifOut(nn):
        """Update the value of dy_k(t+1)/dw_ij"""
        """In terms of the value of paper, 
            next_out = k
            j = inp
            i = out                           """
        kronecker = 0
        for out in range(nn.size[1]):
            for inp in range(nn.size[0]):
                for next_out in range(nn.size[1]):
                    kronecker = 1 if next_out == out else 0

                    nn.dif_out[out][inp][next_out] = nn.fksk[next_out] * \
                    (inner(nn.weight[next_out][nn.numInput+1:-1], nn.dif_out[out][inp][nn.numInput+1:-1])
                     + kronecker*nn.output[0][inp])

    def backward(nn):
        """ backward runs the backpropagation step, """
        """ computing and saving all sensitivities  """
        """ based on the desired output vector.     """

        # Iterate over all neurons in the last layer.        
        # The sensitivites are based on the error and derivatives
        # evaluated at the activation values, which were saved during forward.
        desired = nn.output[0][:nn.numInput]
        error_vec =[0 for out in range(nn.numInput)]
        for ind in range(nn.numInput):
            error_vec[ind] = desired[ind] - nn.output[1][ind]

        for out in range(nn.size[1]):
            for inp in range(nn.size[0]):
                nn.weight_change[inp][out] = inner(error_vec,
                                            nn.dif_out[out][inp][:nn.numInput])
        return error_vec

    """return the input of the current phase"""
    """and then increase the phase by 1"""
    # NOT YET DONE
    def updateinput(nn):
        music = [1 if nn.phase in instrument else 0 for instrument in nn.pattern]
        if nn.phase == nn.numSlots - 1:
            nn.phase = 0
        else:
            nn.phase += 1
        # Add one for the bias
        nn.output[0][:nn.numInput + 1] = music + [1]



    def learn(nn):
        """ learn learns by forward propagating input,  """
        """ back propagating error, then updating.      """
        """ It returns the output vector and the error. """
        change_sum = [[0  for output in range(nn.size[1])]
                          for layer in range(nn.size[0])]
        error =[]
        nn.updateinput()
        nn.forward()
        # Make a vector with erro in every step within one slot
        error = nn.backward()
        nn.updateDifOut()
        # We need to somehow add up change vectors
        #change_sum = addAll(change_sum, nn.weight_change)
        nn.runLength -= 1
        nn.weight = addAll(nn.weight, nn.weight_change)
        #change_ave = [[num / nn.numSlots for num in change_sum[i]]
        #                             for i in range(len(change_sum))]
        #nn.weight = addAll(nn.weight, change_ave)
        wrong = countWrong(error, 0.1)
        return [error, wrong]

    
    def train(nn, noisy):
        """ Trainsthe network using the specified set of samples,    """
        """ for the specified number of epochs.                      """
        """ displayInterval indicates how often to display progress. """
        """ If using as a classifier, assumes the first component in """
        """ the output vector is the classification.                 """
        previousMSE = float("inf")
        # nn.runLength is dcreased by learn(nn)
        wrong = 1 # Just to initialize wrong and makeing sure it's not 0.
        while wrong != 0 and nn.runLength > 0:
            SSE = 0
            wrong = 0
            [error, wrong] = nn.learn()
            num_input = len(error)
            SSE += inner(error, error)
            MSE = SSE/num_input
            wrongpc = 100.0*wrong/num_input
            if wrong == 0:
                break   # stop if classification is correct
            displayInterval = 100
            if nn.runLength%(displayInterval * nn.numSlots) == 0:
                direction = "decreasing" if MSE < previousMSE else "increasing"
                print nn.name, "length", nn.runLength, "MSE =", round(MSE, 3), "wrong =", \
                    str(wrong) + " (" + str(round(wrongpc, 3)) + "%)", direction
            previousMSE = MSE

        if noisy:
            print nn.name, "final weight =", roundall(nn.weight[1:], 3)




class ActivationFunction:
    """ ActivationFunction packages a function together with its derivative. """
    """ This prevents getting the wrong derivative for a given function.     """
    """ Because some derivatives are computable from the function's value,   """
    """ the derivative has two arguments: one for the argument and one for   """
    """ the value of the corresponding function. Typically only one is use.  """

    def __init__(af, name, fun, deriv):
        af.name = name
        af.fun = fun
        af.deriv = deriv

    def fun(af, x):
        return af.fun(x)

    def deriv(af, x, y):
        return af.deriv(x, y)

logsig = ActivationFunction("logsig",
                            lambda x: 1.0/(1.0 + math.exp(-x)),
                            lambda x,y: y*(1.0-y))

tansig = ActivationFunction("tansig",
                            lambda x: math.tanh(x),
                            lambda x,y: 1.0 - y*y)

purelin = ActivationFunction("purelin",
                             lambda x: x,
                             lambda x,y: 1)

def randomWeight():
    """ returns a random weight value between -0.5 and 0.5 """
    return random.random()-0.5

def inner(x, y):
    """ Returns the inner product of two equal-length vectors. """
    n = len(x)
    assert len(y) == n
    sum = 0
    for i in range(0, n):
        sum += x[i]*y[i]
    return sum

def subtract(x, y):
    """ Returns the first vector minus the second. """
    n = len(x)
    assert len(y) == n
    return map(lambda i: x[i]-y[i], range(0, n))

def countWrong(L, tolerance):
    """ Returns the number of elements of L with an absolute """
    """ value above the specified tolerance. """
    return reduce(lambda x,y:x+y,
                  map(lambda x:1 if abs(x)>tolerance else 0, L))

def roundall(item, n):
    """ Round a list, list of lists, etc. to n decimal places. """
    if type(item) is list:
        return map(lambda x:roundall(x, n), item)
    return round(item, n)

def addAll(matrix_0, matrix_1):
    """returns dot addition of two 2-D matrices"""
    if isinstance(matrix_0, list):
        length = len(matrix_0)
        answer = [[] for i in range(length)]
        for i in range(length):
            answer[i] = addAll(matrix_0[i],matrix_1[i])
        
        return answer

    else:
        return matrix_0 + matrix_1

def simple():
    nnet = RTRLnet("simple", 10, logsig, 10, [[1]], 4, 4000)
    nnet.describe(True)
    nnet.train(True)


def main():
    simple()


main()

