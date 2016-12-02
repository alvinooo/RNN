import cPickle, gzip
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import random


class RNN:
     
	def __init__(sf):

		sf.learningRate = 0.1

		sf.input_dim = 256
		sf.hidden_dim = 100
		sf.output_dim = 256

		sf.allData = []
		sf.label = []

		sf.sequenceLen = 10

		sf.buildInputData()
		sf.buildLabelData()

		sf.numEpochs = 2
		#sf.datasetLen = 1
		sf.datasetLen = len(sf.allData) - 1 
		sf.testDataLen = 100

		#make sure these are always reset#
		sf.yIJ = np.array([])
		sf.yJK = []
		#make sure these are always reset#
		
		sf.gradDescOutput = []
		sf.gradDescHidden = []
		sf.gradDescHH = []

		sf.weightsJK = []
		sf.weightsIJ = []
		sf.weightsHH = []
		sf.hiddenActivation = []

		sf.target = []
		sf.charset = []
		asciiLen = 256
		
		for i in range(asciiLen):
		    sf.charset.append(0)

		sf.charset = np.array(sf.charset)


		
		#later ones are all messed up in the data set -> earlier ones learn great!
		#but the earlier ones learn well!
		#training for more epochs does not do that better
		#training for a longer sequence length helps a little bit
		#how to make it work on the book data set because all the letters are messing up when parsing
		#sliding window -> over half words
		#learning for a longer sequence length makes it remember better
		#passing in the last activations to the first one

		#how should we make our network learn better?
		#what does sequence length affect?
		#what does recurrent layer mean?, we are back propping all the deltas so it is tkaing into a/c all deltas for
		#all hidden layers
		#what does hidden layer affect?
		#why does sliding window make it more translational invariant?
		#when I reinitialize the hidden activations then there is a problem, I am basically
		#forgetting the memory of what happened in the past and starting from scracth, this is bad
		#because all the seuences that came before it are now gone!
		#what is expected -> are we expected to produce the book given one character? -->accuracy?

		############## data  set length == 5 ##############
		#sf.allData = [ 'abcde', 'cdefg', 'efghi', 'jklmn', 'lmnop', 'pqrst', 'uvwxy']
		#sf.label = [ 'bcdef', 'defgh', 'fghij', 'klmno', 'mnopq', 'qrstu', 'vwxyz']


		############## data  set length == 3 ##############
		#sf.allData = [ 'abc', 'cde', 'efg', 'jkl', 'lmn', 'pqr', 'uvw']
		#sf.label = [ 'bcd', 'def', 'fgh', 'klm', 'mno', 'qrs', 'vwx']


		sf.buildTarget()
		sf.populateWeights()


	def buildInputData(sf):
		f = open("traindata.txt", 'r')

		sf.allData = []		
		#print "*********************************************************"
		while(True):
			chunk = f.read(sf.sequenceLen)
			if( not chunk ):
			    break
			sf.allData.append( chunk )
		#print sf.allData
		#print "*********************************************************"
		f.close()

	def buildLabelData(sf):
		f = open("traindata.txt", 'r')

		sf.label = []
		#print "*********************************************************"
		while(True):
			chunk = f.read(sf.sequenceLen)
			if( not chunk ):
			    break
			sf.label.append( chunk )
		#print sf.label
		#print "*********************************************************"
		f.close()

	def buildTarget(sf):
		target_one_hot = []
		for i in range(sf.output_dim):
		    target_one_hot.append(0)

		for i in range(sf.output_dim):
		    sf.target.append(list(target_one_hot))

		for i in range(sf.output_dim):
		    sf.target[i][i] = 1

		sf.target = np.array(sf.target)


	def populateWeights(sf):

		#input->hidden layer
		weightsPerClassIJ = []

		for i in range(sf.hidden_dim):
		    mu, sigma = 0, 1.0/(256.0) #mean and standard deviation
		    weightsPerClassIJ = np.random.normal(mu, sigma, sf.input_dim)
		    sf.weightsIJ.append(list(weightsPerClassIJ))

		#converting into numpy array for multiplication
		sf.weightsIJ = np.array(sf.weightsIJ)

		#print "weightsIJ", weightsIJ
		
		#hidden - output layer
		weightsPerClassJK = []
		#append weights 10 times -> for each input -> y_j -> output of the hidden layer  
		for i in range(sf.output_dim):
		    mu, sigma = 0, 1.0/(100.0) # mean and standard deviation
		    weightsPerClassJK = np.random.normal(mu, sigma, sf.hidden_dim)
		    sf.weightsJK.append(list(weightsPerClassJK))
		#converting into numpy array for multiplication
		sf.weightsJK = np.array(sf.weightsJK)

		#print "weightsJK, len/"
		#print len(weightsJK)

		#hidden - output layer
		weightsPerClassHH = []
		#append weights 10 times -> for each input -> y_j -> output of the hidden layer  
		for i in range(sf.hidden_dim):
		    mu, sigma = 0, 1.0/(100.0) # mean and standard deviation
		    weightsPerClassHH = np.random.normal(mu, sigma, sf.hidden_dim)
		    sf.weightsHH.append(list(weightsPerClassHH))
		#converting into numpy array for multiplication
		sf.weightsHH = np.array(sf.weightsHH)


	def setHiddenActivation(sf):    
		sf.hiddenActivation = []

		for ha in range(sf.sequenceLen + 1):

			hiddenActivationStart = []

			for h in range(sf.hidden_dim):
				hiddenActivationStart.append(0.0)

			sf.hiddenActivation.append( list(hiddenActivationStart) )

		sf.hiddenActivation = np.array(sf.hiddenActivation)


	def setHiddenActivation_generate(sf):    
		hiddenActivation_generate = []
		hiddenActivation_generate.append(sf.hiddenActivation[sf.sequenceLen])
		for ha in range(sf.sequenceLen):

			hiddenActivation_generate_start = []

			for h in range(sf.hidden_dim):
				hiddenActivation_generate_start.append(0.0)

			hiddenActivation_generate.append( list( hiddenActivation_generate_start) )

		hiddenActivation_generate = np.array( hiddenActivation_generate)
		sf.hiddenActivation = hiddenActivation_generate
	       
	def generate(sf):

		calc_type = 1
		#expected output: "abcde"
		print "*** testing ***"
		asciiChar = 'A'
		str1 = ""
		str1 += asciiChar
		
		sf.setHiddenActivation_generate()
		
		for a in range(20):

			#asciiChar = str1[-1]
			asciiChar = random.randint(0,255)

			for t in range( sf.sequenceLen ):

				print "---time/sq ex: ", t, "---"
								
				sf.calc_y_hidden_layer(asciiChar, t, calc_type)
				
				asciiChar = sf.calc_y_softmax_output_layer(0, t, calc_type)

				str1 += asciiChar

			print str1

		print "*** testing ***"


	def calc_y_hidden_layer(sf, currData, timestep, calc_type):
	
		if calc_type == 0:
			asciiChar = ord(sf.allData[currData][timestep])
			sf.charset[asciiChar] = 1
		if calc_type == 1:			
			asciiChar = ord(currData)
			sf.charset[asciiChar] = 1

		#print "charset: ", sf.charset
		sf.yIJ = []    
		sf.yIJ = np.tanh( np.dot( sf.weightsHH, sf.hiddenActivation[timestep - 1] ) + np.dot( sf.weightsIJ, sf.charset) )	
		sf.hiddenActivation[timestep] = sf.yIJ
		#print "hiddenActivation[timestep]: ", sf.hiddenActivation[timestep]
		sf.charset[asciiChar] = 0

	    #print sf.yIJ
	    #print "calc_y_hidden_layer: ", currData
	    #print "yIJ", sf.yIJ.shape
	    #print "calc_y_hidden_layer: yIJ", yIJ
	    #reset net_IJ for input to hidden layer	 
	    
	    #print "exiting from calc_y_hidden_layer"

	def calc_y_softmax_output_layer(sf, targetIndx, timestep, calc_type):
	    
		sf.yJK = []
		netJK = []

		#weighted sum
		netJK = np.dot(sf.weightsJK, sf.hiddenActivation[timestep])
		netSum = 0.0  
		for i in range(sf.output_dim):
		    try:
		        netSum += math.exp(netJK[i])
		    except OverflowError:
		        netSum = float('inf')
			
		if calc_type == 0:
			for i in range(sf.output_dim):        
				sf.yJK.append( float(math.exp(netJK[i])) / ( float(netSum)) )

		if calc_type == 1:
			for i in range(sf.output_dim):        
				sf.yJK.append( float(math.exp(netJK[i])) / ( float(netSum)) * 2 ) #adding temperature


		sf.yJK = np.array(sf.yJK)


		#print sf.yJK
		#print "calc_y_softmax_output_layer: yJK", yJK
		#print len(yJK)
		if calc_type == 0:
			print "rcvd res: ", chr(targetIndx)
			print "rcvd res prob: ", sf.yJK[targetIndx]
			print "exp prob: ", sf.target[targetIndx][targetIndx]

		print "recvd char- ascii value: ", np.argmax(sf.yJK)
		return chr( np.argmax(sf.yJK) )

	def forward_back_propogation(sf):

	    training_acc_epochs = []
	    for j in range(sf.numEpochs):
			#reset for every training epoch
			print "------------------- start of epoch: ", j, "-------------------"

			sf.forward_back_prop_single_epoch()

			print "------------------- end of epoch: ", j, "-------------------"
			#training_acc_epochs.append(training_acc)
			#print "training_acc_epochs: ", training_acc_epochs    
			#return training_acc_epochs 

	def resetParameters(sf, i):
		
		sf.gradDescOutput = []
		sf.gradDescHidden = []
		sf.gradDescHiddenHidden = []
		
		if i == 0:
			sf.setHiddenActivation()
		else:
			sf.setHiddenActivation_generate()

	def forward_prop(sf, i, j, calc_type, targetIndx):
		sf.calc_y_hidden_layer(i, j, calc_type)
		sf.calc_y_softmax_output_layer(targetIndx, j, calc_type)
	
	def backward_prop(sf, currData, timestep, targetIndx):
	    delta_K = sf.calc_deltaK_gradient_descent_output_layer(targetIndx, timestep)
	    sf.bptt(delta_K, timestep, currData)	 
	
	def weight_update(sf):
		sf.weightsIJ += np.dot( sf.learningRate, sf.gradDescHidden )
		sf.weightsJK += np.dot( sf.learningRate, sf.gradDescOutput )
		sf.weightsHH += np.dot( sf.learningRate, sf.gradDescHH )

	def forward_back_prop_single_epoch(sf):	
		training_acc = []
		calc_type = 0
		errorCounter = 0

	    #over all training examples
		for i in range(sf.datasetLen):	        
			print "*** begin data ex: ", i, "***"

			sf.resetParameters(i) #resets gradient matrixes

			for t in range( sf.sequenceLen ):

				print "---time/sq ex: ", t, "---"
								
				targetIndx = ord( sf.label[i][t] )	

				sf.forward_prop(i, t, calc_type, targetIndx)
				
				sf.backward_prop(i, t, targetIndx)

				if np.argmax(sf.yJK) != np.argmax(sf.target[targetIndx]):
					errorCounter += 1

			#print "hiddenActivation: ", sf.hiddenActivation
			sf.weight_update()
			
			print "*** end data ex: ", i, "***"

		acc = 1 - (float(errorCounter)/float(sf.datasetLen))

		print "Training Accuracy for ", sf.datasetLen, "data examples in", sf.numEpochs + 1, "th Epoch: ", acc
		return acc*100


	def bptt(sf, delta_K, timestep, currData):

		delta_t = np.dot( sf.weightsJK.T, delta_K )* ( 1 - ( sf.hiddenActivation[timestep] ** 2) ) #at the 0th timestep, hiddenA=0
		for t in range(timestep+1)[::-1]:			
			print "Backprop: timestep=%d & step t=%d " % (timestep, t)

			asciiChar = ord( sf.allData[currData][t] )
			#print "bptt: asciiChar", asciiChar
			sf.charset[asciiChar] = 1

			if sf.gradDescHH != [] and sf.gradDescHidden != []:
				sf.gradDescHH += np.outer( delta_t, sf.hiddenActivation[t-1] ) #what happens when hiddenActivation at t = 0, is 0
				sf.gradDescHidden += np.outer(delta_t, sf.charset )
			else:
				sf.gradDescHH = np.outer( delta_t, sf.hiddenActivation[t-1] )
				sf.gradDescHidden = np.outer(delta_t, sf.charset )

			delta_t = np.dot( sf.weightsHH.T, delta_t ) * (1 - ( sf.hiddenActivation[t - 1] ** 2 ) )

			sf.charset[asciiChar] = 0

		sf.charset[asciiChar] = 0
			

	def calc_deltaK_gradient_descent_output_layer(sf, targetIndx, timestep):

		delta_K = np.array( sf.target[targetIndx] - sf.yJK )
		#print "yJK", sf.yJK
		delta_K = np.array(delta_K)

		if sf.gradDescOutput == []:
			sf.gradDescOutput = np.outer( delta_K, sf.hiddenActivation[timestep] )
		else:
			sf.gradDescOutput += np.outer( delta_K, sf.hiddenActivation[timestep] )

		#print "sf.gradDescOutput: ", sf.gradDescOutput[0]
		return delta_K
		#print "target expected", target[targetIndx]
		#print "yJK->received", yJK

if __name__ == '__main__':
	
	RNN = RNN()
	RNN.forward_back_propogation()
	RNN.generate()


						    