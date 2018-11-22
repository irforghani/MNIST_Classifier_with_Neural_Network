        
#HandWrite Numbers Classification by Neural Network

#import library

import IO_and_Preprocess_Functions as iop
import Neural_Network as NN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#PreProcessing
#iop.PreProcess()


#open weights file with act 89% on test data
f = open("weights.txt")



#read DataSet and split digits and labels
orig_train = pd.read_csv("mnist_train_final.csv")
orig_test = pd.read_csv("mnist_test_final.csv")
print('read csv completed!')


#split labels from train data
labels_data_train = orig_train['label'].values
digits_train = orig_train.drop('label', axis=1)
digits_data_train = digits_train.values

#split labels from test data
labels_data_test = orig_test['label'].values
digits_test = orig_test.drop('label', axis=1)
digits_data_test = digits_test.values

#convert gray scale image to binary image 
digits_data_train[digits_data_train < 127] = 0
digits_data_train[digits_data_train >= 127] = 1
digits_data_test[digits_data_test < 127] = 0
digits_data_test[digits_data_test >= 127] = 1

print('preprocessing completed!')
#print(digits_data_train.shape)



#show_image_test
#iop.image_show(digits_data_train, 0)

#create list of neuron number per layer [hidden layer1, hidden layer2, output layer]
nnl = [15, 15, 10]
#create list of input number per neuron in specific layer
ipn = [784, 15, 15]
#create neural network object with 3 layer and neuron number list and input per neuron list
MCNN = NN.Neural_Network(3, nnl, ipn)
print('network configuration completed!')
#train step
MCNN.train( digits_data_train, labels_data_train) 
print('train completed!')

#save the current weights in the text file
MCNN.save_weight(f)
#set the current weights from file
MCNN.set_weight(f)
#testing the neural network with test data and return accuracy
MCNN.test(digits_data_test, labels_data_test)


#end of code:)
