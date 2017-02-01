from cancer import *

def train(list, limit):
	"""input: a list of training set
		output: [weight, """
	nsamples = len(list)
	weight = [0]* (1 + len(list[0][0]))
	wrong = nsamples
	for epoch in range(limit):
		oldweight = weight
		weight = oneEpoch(weight, list)
		if oldweight == weight:
			return [weight, 0, epoch]
	return [weight, wrong, epoch]

def oneEpoch(weight, list):
	"""input weight: the weight for training set
			 list  : one training set
		output: updated weight"""
	for i in range(len(list)):
		weight = oneCase(weight, list[i])[0]
		wrong += oneCase(weight, list[i])[1]
	return [weight, wrong]
	
def oneCase(weight, list):
	"""training for one training set."""
	list[0] = [1] + list[0]
	result = perceptron(weight, list[0])
	update = list[1] - result
	wrong  = update == 0 
	deltaW = [list[0][i]*update for i in range(len(list[0]))]
	weight = [weight[i] + deltaW[i] for i in range(len(weight))]
	return [weight, wrong]


def perceptron(weight, list):
	"""pass"""
	product = dot(weight, list)
	if product > 0:
		return 1

	else:
		return 0

def dot(list1, list2):
	"""input: two lists
		output: dot product of these"""
	length = len(list1)
	total = 0
	for i in range(length):
		total += list1[i]*list2[i]
	return total

nandSamples = [[[0, 0], 1], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]
iffSamples = [[[0, 0], 1], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]]
majoritySamples = [[[0, 0, 0], 0], [[0, 0, 1], 0], [[0, 1, 0], 0], [[1, 0, 0], 0], [[0, 1, 1], 1], [[1, 0, 1], 1],[[1, 1, 0], 1], [[1, 1, 1], 1]]


print(train(cancertrainingSamples, 200))
