import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, Activation
from keras.layers import Embedding, Input, Dense, Dropout, Lambda, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

#len(ALPHABET)=68
ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
FEATURE_LEN = 1024 #maxlen
path = '../data/'
TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'

def get_char_dict():
    char_dict={}
    for i,c in enumerate(ALPHABET):
        char_dict[c]=i+1
    return char_dict

def char2vec(text, max_length=FEATURE_LEN):
    char_dict = get_char_dict()
    data=np.zeros(max_length)
    
    for i in range(0, len(text)):
        if i >= max_length:
            return data
        
        elif text[i] in char_dict:
            data[i] = char_dict[text[i]]
        
        else:
            data[i]=68
    return data
    

def conv_shape(conv):
    return conv.get_shape().as_list()[1:]

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

#dataset where x=train_df['text'] and y=train_df['class'] 
#binary class
list_sentences_train = train_df["text"].fillna("NA").values
y = train_df['class'].values
list_sentences_test = test_df["text"].fillna("NA").values
data=[]
for text in list_sentences_train:
    data.append(char2vec(text.lower()))
data=np.array(data)

test_data = []
for text in list_sentences_test:
    test_data.append(char2vec(text.lower()))
test_data=np.array(test_data)


def ConvolutionalBlock(input_shape, num_filters):
    model=Sequential()

    #1st conv layer
    model.add(Conv1D(filters = num_filters, kernel_size = 3, strides = 1, padding = "same", input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    #2nd conv layer
    model.add(Conv1D(filters = num_filters, kernel_size = 3, strides = 1, padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    return model

#https://www.tensorflow.org/api_docs/python/tf/nn/top_k
def top_kmax(x):
    x=tf.transpose(x, [0, 2, 1])
    k_max = tf.nn.top_k(x, k=top_k)
    return tf.reshape(k_max[0], (-1, num_filters[-1]*top_k))

def vdcnn_model(num_filters, num_classes, sequence_max_length, num_chars, embedding_size, top_k, learning_rate=0.001):
    
    inputs=Input(shape=(sequence_max_length, ), dtype='int32', name='input')
    
    embedded_seq = Embedding(num_chars, embedding_size, input_length=sequence_max_length)(inputs)
    embedded_seq = BatchNormalization()(embedded_seq)
    #1st Layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_seq)
    
    #ConvBlocks
    for i in range(len(num_filters)):
        conv = ConvolutionalBlock(conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)
        
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
    
    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)
    
    #fully connected layers
    # in original paper they didn't used dropouts
    fc1=Dense(512, activation='relu', kernel_initializer='he_normal')(k_max)
    fc1=Dropout(0.3)(fc1)
    fc2=Dense(512, activation='relu', kernel_initializer='he_normal')(fc1)
    fc2=Dropout(0.3)(fc2)
    out=Dense(num_classes, activation='sigmoid')(fc2)
    
    #optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
    
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

num_filters = [64, 128, 256, 512]
model=vdcnn_model(num_filters=num_filters, num_classes=6,num_chars=69, sequence_max_length=FEATURE_LEN,embedding_size=16,top_k=3)
model.summary()
model.fit(data,y)

model.predict(test_data)