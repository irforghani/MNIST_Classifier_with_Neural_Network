# Input / Output and Preprocessing Functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#show image in mnist dataset
def image_show(digit_data, Index):
    digit = digit_data[Index]
    first_image = np.array(digit, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

#convert mnist to csv function
def convert_mnist_to_csv(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def PreProcess():
    #converting mnist to csv
    convert_mnist_to_csv("train-images.idx3-ubyte", "train-labels.idx1-ubyte","mnist_train.csv", 60000)
    convert_mnist_to_csv("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte","mnist_test.csv", 10000)

    # read csv file
    original_train_set = pd.read_csv('mnist_train.csv')
    original_test_set = pd.read_csv('mnist_test.csv')

    # preprocess columns name
    original_train_set.rename(columns={'5': 'label'}, inplace=True)
    original_test_set.rename(columns={'7': 'label'}, inplace=True)

    # generate final csv dataset train and test files
    original_train_set.to_csv('mnist_train_final.csv', index=False)
    original_test_set.to_csv('mnist_test_final.csv', index=False)

#end of code:)
