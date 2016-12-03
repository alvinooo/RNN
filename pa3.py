import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.datasets import cifar100, cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, SpatialDropout2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave
import theano
import h5py

def load_data(dataset, images_per_cat=None):

    if dataset == "cifar100":
        data = cifar100.load_data()
    elif dataset =="cifar10":
        data = cifar10.load_data()

    train_data = data[0][0]
    train_labels = data[0][1]
    test_data = data[1][0]
    test_labels = data[1][1]

    if images_per_cat:
        num_cat = max(train_labels) + 1
        
        for c in range(num_cat):
            T_cat = T == c
            ind = np.nonzero(T_cat)

    return train_data, train_labels, test_data, test_labels

class TestLoss(Callback):
    def __init__(self, test_data, test_plot,model):
        super(Callback,self).__init__()
        self.test_data = test_data
        self.model = model
        self.test_plot = test_plot

    def on_epoch_end(self, batch, logs={}):
        X_test, T_test_onehot = self.test_data
        print "Test"
        loss, acc = self.model.evaluate(X_test,T_test_onehot, batch_size=32)
        print "loss:",loss,"Acc:",acc
        self.test_plot.append((loss, acc))
"""
def show_image(ind):
    image = train_data[ind,:,:,:]
    plt.imshow(image)
    plt.show()
""" 
def build_net():
    (X,T,X_test,T_test) = load_data("cifar100")

   # X = X[0 : 2000,:]
   # T = T[0 : 2000, :]
    X = X.astype(float)
    X_test = X_test.astype(float)

    T_onehot = to_categorical(T,100)
    T_test_onehot = to_categorical(T_test,100)

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)# horizontal_flip=True)

    datagen.fit(X)
    datagen_test.fit(X_test)

    model = Sequential()
    # model.add(BatchNormalization(input_shape=(32, 32, 3)))

    #ConvLayer 1
    model.add(Convolution2D(32, 3, 3, init='normal', input_shape=(32, 32, 3)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #ConvLayer 2
    model.add(Convolution2D(64, 3, 3, init='normal'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.15))
    
    #ConvLayer 3
    model.add(Convolution2D(128, 3, 3, init='normal'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.2))

    #ConvLayer 4
    #model.add(Convolution2D(128,3,3, init='normal'))
    #model.add(BatchNormalization(mode=0,axis=1))
    #model.add(Activation("relu"))
    #model.add(SpatialDropout2D(0.2))
    
    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(512, init='normal'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    #Output Softmax Layer
    model.add(Dense(100, init='normal', activation="softmax"))
 
    #Train

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    print summary(model)
    #test_plot = []
    #train_plot = model.fit(X,T_onehot,nb_epoch=100, batch_size=32, validation_data=(X_test,T_test_onehot))#callbacks=[TestLoss((X_test,T_test_onehot), test_plot, model)],batch_size=32)

    #train_plot =  model.fit_generator(datagen.flow(X,T_onehot, batch_size=32), validation_data=datagen_test.flow(X_test,T_test_onehot),nb_val_samples=len(X_test), nb_epoch=5, samples_per_epoch=(len(X)))
    model.save("cifar100_model.h5")

    #print train_plot.history["val_acc"]
"""    
    plt.plot(train_plot.history['acc'])

    plt.plot(test_plot)
   
    #model.metrics_names

    #calculate mean
    mean = np.mean(train_data)
    
    #go through all pixels and sub mean
    train_data = (train_data - mean)/mean
"""    
    # maybe not -> https://piazza.com/class/iteh1o6zzoa2wy?cid=345    

def summary(model):
    cols = ["Layer", "Type", "Input Size", "Kernel Size", "# Filters", "Nonlinearity", "Pooling", "Stride", "Size", "Output Shape", "Parameters"]
    summary = {}
    nlayers = []
    model.summary()
    for i in range(len(model.layers)):
        if "conv" in model.layers[i].name or "dense" in model.layers[i].name:
            nlayers += [i]

    for i in nlayers:
        nlayer = i+1
        summary[nlayer] = {c:"" for c in cols[1:]}

        summary[nlayer][cols[1]] = model.layers[i].name
        summary[nlayer][cols[2]] = str(model.layers[i].input_shape)
        summary[nlayer][cols[9]] = str(model.layers[i].output_shape)
        if "conv" in model.layers[i].name or "dense" in model.layers[i].name:
            summary[nlayer][cols[5]] = get_nonlinearity(model,i)
        weights = model.layers[i].get_weights()
        if "dense" in summary[nlayer][cols[1]]:
            units = weights[0].shape[0] * weights[0].shape[1]
            bias = weights[1].shape[0]
            summary[nlayer][cols[10]] = units + bias

        if "conv" in summary[nlayer][cols[1]]:
            summary[nlayer][cols[3]] = (model.layers[i].nb_row, model.layers[i].nb_col)
            summary[nlayer][cols[4]] = model.layers[i].nb_filter    
            summary[nlayer][cols[7]] = model.layers[i].subsample
            pooling_name, size = get_pooling(model,i)
            summary[nlayer][cols[6]] = pooling_name
            summary[nlayer][cols[8]] = size
            
            kernel = weights[0].shape[0] * weights[0].shape[1]
            channels = weights[0].shape[2]
            filters = weights[0].shape[3]
            bias = weights[1].shape[0]
            summary[nlayer][cols[10]] = kernel * channels * filters + bias

    for c in cols:
        print "%-20s" %(c) + "|",
    print ""
    n = 0
    for l in sorted(summary):
        n += 1
        print "%-20i" %(n) +"|",
        for c in cols[1:]:
            print "%-20s" % (str(summary[l][c])) +"|",
        print ""

def get_nonlinearity(model,i):
    if model.layers[i].activation.__name__ != "linear":
        return str(model.layers[i].activation.__name__)
    for n in range(i+1, len(model.layers)):
        if "conv" in model.layers[n].name or "dense" in model.layers[n].name or "flat" in model.layers[n].name:
            return ""
        elif "act" in model.layers[n].name:
            return str(model.layers[n].activation.__name__)

def get_pooling(model,i):
    for n in range(i+1, len(model.layers)):
        if "conv" in model.layers[n].name or "dense" in model.layers[n].name or "flat" in model.layers[n].name:
            return ""
        elif "pool" in model.layers[n].name:
            return str(model.layers[n].name), model.layers[n].pool_size
        

def fine_tuning_cifar10():
    (X,T,X_test,T_test) = load_data("cifar10")

    X = X.astype(float)
    X_test = X_test.astype(float)

    T_onehot = to_categorical(T,10)
    T_test_onehot = to_categorical(T_test,10)

    # Data Preprocesing
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)# horizontal_flip=True)

    datagen.fit(X)
    datagen_test.fit(X_test)
   
    # Cifar100 model delete last layer and set the first layers to be non-trainable
    model = load_model("cifar100_model.h5")
    model.pop()
    for l in model.layers:
        if l.name == "convolution2d_3":
            break
        l.trainable = False

    # Model built on top of Cifar100
    model_top = Sequential()
    model_top.add(Dense(10, init='normal', activation="softmax", input_shape=(512,)))
    
    model.add(model_top)
    
    #Train

    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    test_plot = []

    train_plot =  model.fit_generator(datagen.flow(X,T_onehot, batch_size=32), validation_data=datagen_test.flow(X_test,T_test_onehot),nb_val_samples=len(X_test), nb_epoch=100, samples_per_epoch=(len(X)))

build_net()
#fine_tuning_cifar10()
