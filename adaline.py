import copy
from cancer import *
from math import sqrt

def train(samples, rate, limit, mode, regression):
    """ Train a perceptron with the set of samples. """
    """ samples is a list of pairs of the form [input vector, desired output] """
    """ limit is a limit on the number of epochs """
    """ Returns a list of the number of epochs used, """
    """                   the number wrong at the end, and """
    """                   the final weights """
    
    nsamples = len(samples)
    wrong = nsamples # initially assume every classification is wrong
    if mode == 'stochastic':
        trained = stochastic(samples, rate, limit, nsamples, regression)
    elif mode == 'batch':
        trained = batch(samples, rate, limit, nsamples, regression)
    elif mode == 'mini_batch':
        trained = mini_batch(samples, rate, limit, nsamples)
    elif mode == 'all':
        trained1 = stochastic(samples, rate, limit, nsamples)
        trained2 = batch(samples, rate, limit, nsamples)
        trained3 = mini_batch(samples, rate, limit, nsamples)
    else:
        return 'wrong'
    [epoch, wrong, weights, num_pocket, pocket] = trained
    print("Rate of", rate,"took", epoch-1, "epochs till the weight gets optimized. The final weight was", weights, " and the error was", wrong)
    print("Pocket weights was", pocket, "and error was", nsamples - num_pocket)
    print("Pocket error percentage:", ((nsamples-num_pocket )/ nsamples)*100, " error:",(nsamples-num_pocket ))
    print("num:", nsamples)
    return [epoch-1, wrong, weights]

def batch(samples, rate, limit, nsamples, regression):
    weights = [0]* (len(samples[0][0]) + 1 )# initialize weights to all 0
    pocket = copy.copy(weights)
    num_pocket = 0
    wrong = nsamples
    n = len(weights)
    epoch = 1
    error_sum = 1000 
    if regression:
        while error_sum > 10 and epoch <= limit:
            wrong = 0
            #print("epoch ", epoch, ": weights = ", weights)
            num_correct = 0
            error_sum = 0
            for sample in samples:
                input = [1] + sample[0]
                desired = sample[1]
                output = inner(weights, input)
                error = desired - output
                error_sum += error

            for sample in samples:
                for i in range(n):
                    weights[i] = round(weights[i] + rate*error_sum*([1] + sample[0])[i], 3)
            #print("wrong: ", wrong)
            epoch = epoch + 1
            wrong = error_sum
    else:
        while wrong > 0 and epoch <= limit:
            wrong = 0
            #print("epoch ", epoch, ": weights = ", weights)
            num_correct = 0
            error_sum = 0
            for sample in samples:
                input = [1] + sample[0]
                desired = sample[1]
                output = inner(weights, input)
                error = desired - output
                error_sum += error
            for sample in samples:
                for i in range(n):
                    weights[i] = round(weights[i] + rate*error_sum*([1] + sample[0])[i], 3)
            wrong = test(weights, samples)
            #print("wrong: ", wrong)
            epoch = epoch + 1
    return [epoch, wrong, weights, num_pocket, pocket]

def mini_batch():
           pass       
    
def stochastic(samples, rate, limit, nsamples, regression):
    weights = [0]* (len(samples[0][0]) + 1 )# initialize weights to all 0
    pocket = copy.copy(weights)
    num_pocket = 0
    wrong = nsamples
    n = len(weights)
    epoch = 1
    if regression:
        sme = 10000
        while sme > 1 and epoch <= limit:
            #print("epoch ", epoch, ": weights = ", weights)
            num_correct = 0
            error_sum = 0
            for sample in samples:
                input = [1] + sample[0]
                desired = sample[1]
                error = desired - inner(weights, input)
                error_sum += error**2

                for i in range(0, n):
                    weights[i] = round(weights[i] + rate*error*input[i], 3) # update              

            sme = error_sum / nsamples
            print(sme)
            #print("wrong: ", wrong)
            epoch = epoch + 1

    else:
        while wrong > 0 and epoch <= limit:
            wrong = 0
            #print("epoch ", epoch, ": weights = ", weights)
            num_correct = 0
            for sample in samples:
                input = [1] + sample[0]
                desired = sample[1]
                error = desired - inner(weights, input)
                if desired - classifier(weights, input) == 0:
                    num_correct += 1
                else:
                    wrong += 1

                if num_pocket < num_correct:
                    pocket = copy.copy(weights)
                    num_pocket = num_correct

                for i in range(0, n):
                    weights[i] = round(weights[i] + rate*error*input[i], 3) # update              
     

            #print("wrong: ", wrong)
            epoch = epoch + 1
    return [epoch, wrong, weights, num_pocket, pocket]

def classifier(weights, input):
    """ The transfer function for a Perceptron """
    """     weights is the weight vector """
    """     input is the input vector """
    return 1 if inner(weights, input) > 0.5 else 0


def inner(x, y):
    """ Returns the inner product of two vectors with the same number of components """
    n = len(x)
    assert len(y) == n
    sum = 0
    for i in range(0, n):
        sum = sum + x[i]*y[i]
    return sum

def test(weights, samples):
    wrong = 0
    for sample in samples:
        if classifier(weights, [1] + sample[0]) != sample[1]:
            wrong += 1
    print("Percentage error:", wrong/len(samples))
    print("num:", len(samples))
    return wrong
def reg_test(weights, samples):
    for sample in samples:
        print(round(inner(weights, [1] + sample[0]), 3))
    return "finished"

def normalization(samples, limit):
    for sample in samples:
        if (sample[0][0] == 0 and sample[0][1] == 0) == 0:
            divider = (sample[0][0]**2 + sample[0][1]**2)**0.5
            sample[0] = [round(sample[0][0] / divider, 3), round(sample[0][1] / divider, 3)]
    train(samples, limit)
nandSamples = [[[0, 0], 1], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]
iffSamples = [[[0, 0], 1], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]]

print('///////////////////////')
#print(train(cancertrainingSamples, 1, 200, 'stochastic', False))
#print(train(cancertrainingSamples, 0.1, 200, 'stochastic', False))
#print(train(cancertrainingSamples, 0.01, 200, 'stochastic', False))
#print(train(cancertrainingSamples, 0.01, 200, 'stochastic', False))

#trained = train(cancertrainingSamples, 0.01, 200, 'stochastic', False)
#print(test(trained[2] , cancertrainingSamples))
trained = train(widorowHoffTrain, 0.1, 1000, 'stochastic', True)
print(reg_test(trained[2], widorowHoffTest))
#trained = train(horizontalVerticalTrain, 0.01, 200, 'stochastic', False)
#print(test(trained[2], horizontalVerticalTest))