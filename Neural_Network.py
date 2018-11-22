import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


#neuron class definition to simulate neuron behavior!
class Neuron:
    #contructor function
    def __init__(self, input_number = 0, input:list = [] , weight:list = [], bias = 0.0):
        self.innum = input_number
        self.x = input
        self.w = weight
        self.generate_random_weight()
        self.b = bias
        self.delta = 0
        self.out = 0

    #set the input number of this neuron
    def set_input_number(self, input_number : int):
        self.innum = input_number
    #set neuron inputs
    def set_input(self, input: list):
        self.x = input
    #set delta(Error function output)
    def set_delta(self, dv):
        self.delta = dv
    #get delta(Error function output)
    def get_delta(self):
        return self.delta
    #set neuron weights list
    def set_weight(self, weight: list):
        self.w = weight
    #set specific weight of neuron
    def set_weight2(self, weight , num):
        self.w[num] = weight
    #set bias
    def set_bias(self, bias : int):
        self.b = bias
    #get weight list
    def get_weight(self):
        return self.w
    #get input list
    def get_input(self):
        return self.x
    
    #random weight generate
    def generate_random_weight(self):
        rw = []
        for i in range(0, self.innum):
            rw.append(random.uniform(-1.0, 1.0))
        self.w = rw
    #calculating the net of neuron
    def calc(self):
        net = 0
        for i in range(0, self.innum):
            net += self.x[i].get_output() * self.w[i]
        net += self.b
        self.out = self.activation_function(net)

    #filter the neuron output by activation function (in this example actiovation function is sigmoid)
    def activation_function(self, net_output):
        #sigmoid
        return 1.0/(1.0+np.exp(-net_output))

    #get output of neuron
    def get_output(self):
        return self.out
    #set output of neuron
    def set_output(self, o):
        self.out = o

#Layer class definition for simulate the neural networks layers
class Layer:
    #constructor function
    def __init__(self, Neuron_Number, input_per_neuron: int):
        self.ipn = input_per_neuron
        self.neuron_number = Neuron_Number
        self.Neuron_list = []
        for i in range(0, Neuron_Number):
            self.Neuron_list.append(Neuron(input_per_neuron))
    #calculate all net output in this layer
    def calculate(self):
        for i in range(0, self.neuron_number):
            self.Neuron_list[i].calc()
    #function for set input data
    def read_input(self, input_list : list):
        for i in range(0, self.neuron_number):
            self.Neuron_list[i].set_input_number(self.ipn)
            self.Neuron_list[i].set_input(input_list)
    #return neuron number in this layer
    def get_neuron_number(self):
        return self.neuron_number
    #access to specific neuron
    def get_neuron(self, index):
        return self.Neuron_list[index]
    def get_ipn(self):
        return self.ipn
    
#neural network class definition for simulate neural network
class Neural_Network:
    def __init__(self, layer_number = 0, neuron_number_list = [], input_per_neuron_list = []):
        self.lnum = layer_number
        self.nnl = neuron_number_list
        self.ipn = input_per_neuron_list
        self.Layers = []
        for i in range(0, self.lnum):
            self.Layers.append(Layer(self.nnl[i], self.ipn[i]))
        #connecting
        print('connection step started!')
        for i in range(1, self.lnum):
            for j in range (0, self.nnl[i]):
                for k in range(0, self.ipn[i]):
                    self.Layers[i].Neuron_list[j].set_input(self.Layers[i-1].Neuron_list)
                    
    #convert label to onehot function        
    def convert_label_to_onehot(self, x : int):
        vec = np.zeros(10)
        vec[x] = 1
        return vec
    
    #softmax function
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    #save weights functions
    def save_weight(self, f):
        for i in range(0, self.lnum):
          for j in range(0, self.Layers[i].get_neuron_number()):
             for k in self.Layers[i].Neuron_list[j].get_weight():
                f.write(str(k))
                f.write(" ")
        
    #set weights function
    def set_weight(self, f):
         for i in range(0, self.lnum):
          for j in range(0, self.Layers[i].get_neuron_number()):
            for k in range(0, self.Layers[i].get_ipn()):
              self.Layers[i].Neuron_list[j].set_weight2(float((f.readline())[:-2]), k)
                
    #train function
    def train(self, train_data, train_labels):
        print('train step started!')
        error = 0
        error_temp = np.zeros(10)
        error_avg = 0
        threshold = 0.2
        epoch = 0
        #batch size
        batch_size = 1
        #learning rate
        n = 0.7        
        for e in range (0,4):
          
          n -= 0.1
          epoch = 0
          while (epoch < 59000):

              epoch += 1
  

              if(epoch%5000 == 0):
                print(epoch)
              #n -= 0.05
              #if n == 0:
              #    break
              #print('epoch:' + str(epoch))
              error_temp = np.zeros(10)
              for i in range(((epoch - 1) * batch_size), (((epoch - 1) * batch_size) + batch_size)):
                  #print('epoch:' + str(epoch) + '::::: image number:' + str(i))
                  #print('|', end = '')
                  temp = []
                  for j in range(0, self.ipn[0]):
                      temp2 = Neuron()
                      temp2.set_output(train_data[i][j])
                      temp.append(temp2)
                  self.Layers[0].read_input(temp)
                  self.Layers[0].calculate()
                  #print('layer 0 calculated!')
                  for j in range(1, self.lnum):
                      self.Layers[j].calculate()
                      #print('layer ' + str(j) + ' calculated!')

                  error = []
                  for j in range(0, self.Layers[self.lnum - 1].get_neuron_number()):
                      #print(self.convert_label_to_onehot(train_labels[i]))
                      if(e > 2 and epoch > 58980):
                        print(str(self.Layers[self.lnum - 1].Neuron_list[j].get_output()) + '   des: ' + str(self.convert_label_to_onehot(train_labels[i])[j]) )
                      error.append(self.convert_label_to_onehot(train_labels[i])[j] - self.Layers[self.lnum - 1].Neuron_list[j].get_output())
                      #print(self.convert_label_to_onehot(train_labels[i])[j] - self.Layers[self.lnum - 1].Neuron_list[j].get_output())
                  error_temp += np.asarray(error)

              #print('')
              error_avg = error_temp / batch_size
              tmp = 0
              for b in error_avg:
                  tmp += pow(b,2)
              tmp /= len(error_avg)
              if(e > 2 and epoch > 58980):
                 print('MSE = ' + str(tmp))
             # print(error_avg[0])
              #print(error_temp.sum())
              #print('error:' + str(error_avg))               


              #Back Propagation
              #print('Back Propagation Started!')
              i = self.lnum - 1
              while ( i != -1):
                  if (i == self.lnum - 1):
                      for j in range(0, self.Layers[i].get_neuron_number()):
                          self.Layers[i].Neuron_list[j].set_delta(self.Layers[i].Neuron_list[j].get_output()
                                                                              * (1 - self.Layers[i].Neuron_list[j].get_output())
                                                                              * (error_avg[j]))
                          general_delta = n * self.Layers[i].Neuron_list[j].get_delta()
                          wnew = []
                          for k in range(0, self.Layers[i].get_ipn()):
                              wnew.append(self.Layers[i].Neuron_list[j].get_weight()[k] + (general_delta * self.Layers[i].Neuron_list[j].get_input()[k].get_output()))
                          self.Layers[i].Neuron_list[j].set_weight((0.5 * np.asarray(wnew)) + (0.5 * np.asarray(self.Layers[i].Neuron_list[j].get_weight())))
                  else:
                      for j in range(0, self.Layers[i].get_neuron_number()):
                          sigma = 0
                          for k in range(0, self.Layers[i+1].get_neuron_number()):
                              sigma += self.Layers[i+1].Neuron_list[k].get_weight()[j] * self.Layers[i+1].Neuron_list[k].get_delta()
                          self.Layers[i].Neuron_list[j].set_delta(self.Layers[i].Neuron_list[j].get_output()
                                                                  * (1 - self.Layers[i].Neuron_list[j].get_output())
                                                                  * sigma)
                          general_delta = n * self.Layers[i].Neuron_list[j].get_delta()
                          wnew = []
                          for k in range(0, self.Layers[i].get_ipn()):
                              wnew.append(self.Layers[i].Neuron_list[j].get_weight()[k] + (general_delta * self.Layers[i].Neuron_list[j].get_input()[k].get_output()))                    
                          self.Layers[i].Neuron_list[j].set_weight((0.5 * np.asarray(wnew)) + (0.5 * np.asarray(self.Layers[i].Neuron_list[j].get_weight())))

                  i -= 1
         
     #test function               
    def test(self, test_data, test_labels):
        truepositive = 0
        for i in range(0, 9999):
            if (i%500 == 0):
              print(i)
            temp = []
            for j in range(0, self.ipn[0]):
                temp2 = Neuron()
                temp2.set_output(test_data[i][j])
                temp.append(temp2)
            self.Layers[0].read_input(temp)
            self.Layers[0].calculate()
            #print('layer 0 calculated!')
            for j in range(1, self.lnum):
                self.Layers[j].calculate()
                #print('layer ' + str(j) + ' calculated!')
            
            #error = []
            out = []
            for j in range(0, self.Layers[self.lnum - 1].get_neuron_number()):
                #print(self.convert_label_to_onehot(train_labels[i]))
                out.append(self.Layers[self.lnum - 1].Neuron_list[j].get_output())
                #print(str(self.Layers[self.lnum - 1].Neuron_list[j].get_output()) + '   des: ' + str(self.convert_label_to_onehot(test_labels[i])[j]) )
                #error.append(self.convert_label_to_onehot(train_labels[i])[j] - self.Layers[self.lnum - 1].Neuron_list[j].get_output())
                #print(self.convert_label_to_onehot(train_labels[i])[j] - self.Layers[self.lnum - 1].Neuron_list[j].get_output())
            #error_temp += np.asarray(error)
            if (np.argmax(out) == np.argmax(self.convert_label_to_onehot(test_labels[i]))):
                truepositive += 1  
                                                                                          
                                                                                    
            '''
            print('')
            error_avg = error_temp / batch_size
            tmp = 0
            for b in error_avg:
                tmp += pow(b,2)
            tmp /= len(error_avg)
            print('MSE = ' + str(tmp))
           # print(error_avg[0])
            #print(error_temp.sum())
            #print('error:' + str(error_avg))               
            '''
        act = truepositive / 9999        
        print('accuracy = ' + str(act))
        
#end of code :)
