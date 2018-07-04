import os,sys
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import merge, Conv2D, Conv2DTranspose, Dropout, Add, Flatten, Dense, Input, Activation, MaxPooling2D, AveragePooling2D, Lambda
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras import regularizers
from sklearn.manifold import TSNE
from keras.preprocessing.image import ImageDataGenerator

def initial_env(num=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = num
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    session_conf = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

def load_test_data(path= './Fashion_MNIST_student/test/'):
    if path[-1] != '/':
        path += '/' 
    test_X = []
    i = []
    for f in sorted(os.listdir(path)):
        i.append(os.path.splitext(f)[0])
    i.sort(key=int)
    for j in range(len(i)):
        test_X.append(plt.imread(path + i[j] + '.png'))
    test_X = np.array(test_X)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    return np.array(test_X) 


def load_train_data(path='./Fashion_MNIST_student/train/'):
    if path[-1] != '/':
        path += '/' 
    train_X = []
    class_path = sorted(os.listdir(path))
    for i in class_path:   
        for j in sorted(os.listdir(path+i)):
            train_X.append(plt.imread(path +i+'/'+j))
    train_Y = np.linspace(0, 10, num=2000,dtype=int,endpoint=False, retstep=False)
    train_X = np.array(train_X)
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
    return train_X , train_Y


def normalize_data(train_X,test_X):
    n_train_X = np.zeros((train_X.shape),dtype=np.float)
    n_test_X = np.zeros((test_X.shape),dtype=np.float)
    for i in range(train_X.shape[0]):
        n_train_X[i] = train_X[i] / (train_X[i].max()/2.0) -1
    for i in range(test_X.shape[0]):
        n_test_X[i] = test_X[i] / (test_X[i].max()/2.0) -1
    return n_train_X, n_test_X





class model_A():
    def __init__(self):
        
        self.height = 28
        self.width = 28
        self.channel = 1
        self.img_shape = (self.height, self.width, self.channel)
        self.w_init = 'glorot_normal'
        self.optimizer = Adam(1e-5, 0.5, 0.999)

        self.cnn = self.resnet()
        
        
    def zero_pad_channels(self,x, pad=0):
        pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
        return tf.pad(x, pattern)


    def residual_block(self,x, nb_filters=16, subsample_factor=1):

        prev_nb_channels = K.int_shape(x)[3]

        if subsample_factor > 1:
            subsample = (subsample_factor, subsample_factor)
            shortcut = AveragePooling2D(pool_size=subsample)(x)
        else:
            subsample = (1, 1)
            shortcut = x

        if nb_filters > prev_nb_channels:
            shortcut = Lambda(self.zero_pad_channels,
                              arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

        y = BatchNormalization(axis=3)(x)
        y = Activation('relu')(y)
        y = Conv2D(nb_filters, kernel_size=(3, 3), strides=subsample,
                          kernel_initializer='glorot_normal', padding='same')(y)
        y = BatchNormalization(axis=3,momentum=0.1)(y)
        y = Activation('relu')(y)
        y = Conv2D(nb_filters, kernel_size=(3, 3), strides=(1, 1),
                          kernel_initializer='glorot_normal', padding='same')(y)

        y = Dropout(0.5)(y)
#         out = Add()([y, shortcut])
        out = merge([y, shortcut], mode='sum')

        return out

    
    def resnet(self):
        img_rows, img_cols = 28, 28
        img_channels = 1

        blocks_per_group = 4
        widening_factor = 10

        inputs = Input(shape=(img_rows, img_cols, img_channels))

        x = Conv2D(16, kernel_size=(3, 3), 
                          kernel_initializer='glorot_normal', padding='same')(inputs)

        for i in range(0, blocks_per_group):
            nb_filters = 16 * widening_factor
            x = self.residual_block(x, nb_filters=nb_filters, subsample_factor=1)

        for i in range(0, blocks_per_group):
            nb_filters = 32 * widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            x = self.residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

        for i in range(0, blocks_per_group):
            nb_filters = 64 * widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            x = self.residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

        x = BatchNormalization(axis=3,momentum=0.1)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(7, 7), strides=None, padding='valid')(x)
        x = Flatten()(x)

        predictions = Dense(10, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.summary()
        
        model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['sparse_categorical_accuracy'])
        
        return model
        

    def train(self, train_X=None, train_Y=None, epochs=10, batch_size=32):
        best = 1000000.0
        train_loss = []
        train_acc = []


        
        with open( './plot_wideresnet.csv', 'w') as csv:
            csv.write('loss,acc'+'\n')

            for i in range(epochs):
                logg = self.cnn.fit(train_X, train_Y,initial_epoch=i
                                       ,epochs=i+1)
                train_loss.append(logg.history['loss'][0])
                train_acc.append(logg.history['sparse_categorical_accuracy'][0])

                csv.write(str(logg.history['loss'][0]) + ',' + str(logg.history['sparse_categorical_accuracy'][0])+'\n')
                current = logg.history['loss'][0]
                if current < best:
                    self.cnn.save_weights('./weights/cnn_%d_best'%(i+1))
                    best = current
                elif ((i+1) % 50 == 0):
                    self.cnn.save_weights('./weights/cnn_%d'%(i+1))

                fig = plt.figure(figsize=(16,4))
                plt.subplot(1,2,1)
                plt.title("Sparse_categorical_crossentropy Loss")
                plt.xlabel("Epochs")
                plt.plot(train_loss, label = 'Training')
                plt.legend()

                plt.subplot(1,2,2)
                plt.title("Classification Accuracy")
                plt.xlabel("Epochs")
                plt.plot(train_acc, label = 'Training')
                plt.legend()
                fig.savefig('./result/task1_learn_curve.jpg',dpi=fig.dpi, bbox_inches='tight')
                plt.close()

            recongition = self.cnn.evaluate(train_X,train_Y)
            print("Loss: %s  Recongition rate: %s"%(recongition[0],recongition[1]))


    def predict(self, test_X=None, path='./'):  
        if path[-1] != '/':
            path = path + '/'
        with open( path+'task1.csv', 'w') as csv:
            output_labels = self.cnn.predict(test_X)
            labels = np.argmax(output_labels,axis=1)
            csv.write('image_id,predicted_label'+'\n')
            for i in range(len(labels)):
                csv.write(str(i) + ',' + str(labels[i])+'\n')
            

            
    def reload_model(self,i=None):
        self.cnn.load_weights('./weights/cnn_'+ str(i))
    



if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    gpu_num = sys.argv[3]
    model_name = sys.argv[4]
    if data_path[-1] != '/':
        data_path += '/'
    # data_path= './Fashion_MNIST_student/'
    initial_env(num=gpu_num)
    train_X, train_Y = load_train_data(path=data_path + 'train/')
    test_X = load_test_data(path=data_path + 'test/')
    train_X,test_X = normalize_data(train_X,test_X)
    model = model_A()
    model.cnn.load_weights('./cnn_500')
    model.predict(test_X=test_X)
