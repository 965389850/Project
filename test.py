import numpy as np
import pandas as pd
import tensorflow as tf
import os, time
from matplotlib import pyplot as plt
from pandas import DataFrame as df
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Dense, LSTM, Bidirectional, Flatten, Layer
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from tensorflow.keras import Input
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import regularizers
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve

# Attention机制
class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim
        })
        return config

x_train_savepath = './datasave1/KDD_x_train.npy'
x_train_cnn_savepath = './datasave1/KDD_x_train_cnn.npy'
y_train_savepath = './datasave1/KDD_y_train.npy'


x_train = np.load(x_train_savepath)
x_train_cnn = np.load(x_train_cnn_savepath)
y_train = np.load(y_train_savepath)

## ---------- 模型预测 ----------##
modelcnn1 = load_model('./modle/CNN_attention1.hdf5',custom_objects={'Self_Attention': Self_Attention})
modelcnn2 = load_model('./modle/CNN_attention2.hdf5',custom_objects={'Self_Attention': Self_Attention})
modelcnn3 = load_model('./modle/CNN_attention3.hdf5',custom_objects={'Self_Attention': Self_Attention})
modellstm1 = load_model('./modle/BiLSTM_attention1.hdf5',custom_objects={'Self_Attention': Self_Attention})
modellstm2 = load_model('./modle/BiLSTM_attention2.hdf5',custom_objects={'Self_Attention': Self_Attention})
modellstm3 = load_model('./modle/BiLSTM_attention3.hdf5',custom_objects={'Self_Attention': Self_Attention})


predcnn1 = df(modelcnn1.predict(x_train_cnn).reshape(-1,5))
predcnn2 = df(modelcnn2.predict(x_train_cnn).reshape(-1,5))
predcnn3 = df(modelcnn3.predict(x_train_cnn).reshape(-1,5))

predlstm1 = df(modellstm1.predict(x_train).reshape(-1,5))
predlstm2 = df(modellstm2.predict(x_train).reshape(-1,5))
predlstm3 = df(modellstm3.predict(x_train).reshape(-1,5))

pred = pd.concat([predcnn1,predcnn2,predcnn3,predlstm1,predlstm2,predlstm3],axis=1)

model = load_model('./modle/MLP_attention_test.hdf5')

a = df(model.predict(pred))
Fresult = df()
for i in range(0,len(a)):
    b = a.loc[i].argmax()
    Fresult = Fresult.append(b)
c = accuracy_score(Fresult, y_train)
r = recall_score(Fresult, y_train,average='macro')
print(c,'\n',r)