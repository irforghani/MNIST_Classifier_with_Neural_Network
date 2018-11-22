# MNIST_Classifier_with_Neural_Network
MNIST Classifier with use the neural network

![alt text](http://s9.picofile.com/file/8343555000/1.jpeg)


The project is implemented based on object-oriented concepts, which can be personalized and modified by the number of layers and neurons in the neural network.  The activation function of this project is intended to be Sigmoid by default. We have used two techniques to improve the learning process for updating weights. The first technique is to use the learning rate parameter to determine the length of the jump step to the minimum point of the function, which over time and with the assumption that we are closer to the minimum level over time, we reduce the amount of this parameter gradually. The second technique is to use the experience of weight values ​​and its impact on gaining new weight, which makes the uncoordinated samples do not cause severe changes in weights. Finally, according to the learning and testing processes performed on Google's servers in two steps, we first arrived at a percentage accuracy of 89, and then to an accuracy of 92. It is expected that with increasing number of repetitions of the learning process, this precision can be Increased by about 95%. But it should be noted that excessive learning increases the probability of over fit.
