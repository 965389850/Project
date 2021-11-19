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
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, \
    precision_recall_curve, roc_curve  # 准确率，精确度，召回率，混淆矩阵，精度曲线，ROC
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, Conv1D, \
    MaxPool1D


# ONE-HOT 编码
def onehot(dataset, mod=None, integer_values=None):
    values = array(dataset)
    print(values)
    # integer encode
    if mod == None:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        print(integer_encoded)
    else:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(integer_values)
        print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    # invert first example
    # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)
    return onehot_encoded


# 标准化
def autoStd(dataset):
    ##method2 Z-socre by Skit-Learn
    std = StandardScaler()
    x_std = std.fit_transform(dataset)
    print(x_std)
    return x_std


# 归一化
def autoMinMax(dataset):
    std = MinMaxScaler()
    x_std = std.fit_transform(dataset)
    print(x_std)
    return x_std


# 预处理
def prepocessing(data, f1, f2, mod=0, value=None):
    result = []
    result = df(result)

    def insert(dataset):
        x = result.shape[1]
        m = 0
        for i in range(0, dataset.shape[1]):
            result.insert(i + x, i + x, dataset.iloc[:, m])
            m = m + 1
        return (result)

    # onehot编码
    result = []
    result = df(result)
    a = data.iloc[:, 0:f1]
    insert(a)
    if mod == 0:
        for n in range(f1, f2 + 1):
            a = onehot(data[n])
            a = df(a)
            insert(a)
            print(result)
    else:
        for n in range(f1, f2 + 1):
            a = onehot(data[n], mod=1, integer_values=array(value[n]))
            a = df(a)
            insert(a)
            print(result)

    a = data.iloc[:, f2 + 1:]
    insert(a)
    print(result)
    label = result.iloc[:, [-2, -1]]
    result = result.iloc[:, 0:-2]
    result = autoStd(result)
    result = autoMinMax(result)
    return (label, result)


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


# ------------------------------模型-------------------------------#


def modcnn():
    model = tf.keras.Sequential([
        Conv2D(filters=25, kernel_size=(3, 3), padding='valid'),
        BatchNormalization(),
        Activation('relu'),  # 激活层
        MaxPool2D(pool_size=(2, 2), padding='valid'),  # 池化层
        Dropout(0.5),  # dropout层

        Conv2D(filters=20, kernel_size=(2, 2), padding='valid'),
        BatchNormalization(),
        Activation('relu'),  # 激活层
        MaxPool2D(pool_size=(2, 2), padding='valid'),  # 池化层
        Dropout(0.5),  # dropout层

        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    return (model)


def modlstm():
    model = tf.keras.Sequential([
        LSTM(32, activation='tanh', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    return (model)


# BiLSTM-DNN_attention
def BiLSTMDNN_attention(LSTM_num, dense, dense1):  # LSTM_num=64
    K.clear_session()
    inputs = Input(shape=(1, 121))  # 输入层
    # BiLSTM层
    a = Bidirectional(LSTM(LSTM_num, activation='tanh', return_sequences=True))(inputs)
    a = Dropout(0.5)(a)
    # DNN层
    a = Dense(dense, activation='tanh')(a)
    a = Dropout(0.5)(a)
    a = Dense(dense, activation='tanh')(a)
    a = Dropout(0.5)(a)
    a = Dense(dense, activation='tanh')(a)
    a = Dropout(0.5)(a)
    a = Dense(dense, activation='tanh')(a)
    # attention层
    outputs = Self_Attention(dense)(a)
    # 输出层
    outputs = Dense(dense1, activation='softmax')(outputs)

    model = Model(inputs=[inputs], outputs=outputs)
    return (model)


# BiLSTM_attention
def BiLSTM_attention(LSTM_num, dense1=5):
    K.clear_session()
    inputs = Input(shape=(1, 121))  # 输入层

    # BiLSTM层
    a = Bidirectional(
        LSTM(LSTM_num, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(0.1)))(inputs)
    a = Dropout(0.5)(a)

    # attention层
    outputs = Self_Attention(LSTM_num * 2)(a)

    # 输出层
    outputs = Dense(dense1, activation='softmax')(outputs)

    model = Model(inputs=[inputs], outputs=outputs)
    return (model)


# CNN_attention
def CNN_attention(dense):  # dense=90
    K.clear_session()
    inputs = Input(shape=(11, 11, 1))  # 输入层
    # CNN
    a = Conv2D(filters=25, kernel_size=(3, 3), padding='valid', kernel_regularizer=regularizers.l2(0.1))(inputs)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)  # 激活层
    a = MaxPool2D(pool_size=(2, 2), padding='valid')(a)  # 池化层
    a = Dropout(0.5)(a)  # dropout层

    a = Conv2D(filters=20, kernel_size=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.1))(a)
    a = BatchNormalization()(a)
    a = Activation('relu')(a)  # 激活层
    a = MaxPool2D(pool_size=(2, 2), padding='valid')(a)  # 池化层
    a = Dropout(0.5)(a)  # dropout层

    a = Flatten()(a)
    a = Dense(dense, activation='relu', kernel_regularizer=regularizers.l2(0.1))(a)
    a = tf.reshape(a, (-1, 1, dense))
    # attention层
    outputs = Self_Attention(dense)(a)
    # 输出层
    outputs = Dense(5, activation='softmax')(outputs)

    model = Model(inputs=[inputs], outputs=outputs)
    return (model)


def pred(data):
    x_train = data.reshape(-1, 1, 121)
    x_train_cnn = np.array(x_train)
    x_train_cnn = x_train_cnn.reshape(-1, 11, 11)
    x_train_cnn = x_train_cnn[:, :, :, np.newaxis]

    x_train = x_train.astype('float64')
    x_train_cnn = x_train_cnn.astype('float64')

    # 模型预测

    modelcnn1 = load_model('./modle/CNN_attention1.hdf5', custom_objects={'Self_Attention': Self_Attention})
    modelcnn2 = load_model('./modle/CNN_attention2.hdf5', custom_objects={'Self_Attention': Self_Attention})
    modelcnn3 = load_model('./modle/CNN_attention3.hdf5', custom_objects={'Self_Attention': Self_Attention})
    modellstm1 = load_model('./modle/BiLSTM_attention1.hdf5', custom_objects={'Self_Attention': Self_Attention})
    modellstm2 = load_model('./modle/BiLSTM_attention2.hdf5', custom_objects={'Self_Attention': Self_Attention})
    modellstm3 = load_model('./modle/BiLSTM_attention3.hdf5', custom_objects={'Self_Attention': Self_Attention})

    predcnn1 = df(modelcnn1.predict(x_train_cnn).reshape(-1, 5))
    predcnn2 = df(modelcnn2.predict(x_train_cnn).reshape(-1, 5))
    predcnn3 = df(modelcnn3.predict(x_train_cnn).reshape(-1, 5))

    predlstm1 = df(modellstm1.predict(x_train).reshape(-1, 5))
    predlstm2 = df(modellstm2.predict(x_train).reshape(-1, 5))
    predlstm3 = df(modellstm3.predict(x_train).reshape(-1, 5))

    pred = pd.concat([predcnn1, predcnn2, predcnn3, predlstm1, predlstm2, predlstm3], axis=1)
    model = load_model('./modle/MLP_attention_test.hdf5')
    a = df(model.predict(pred))
    Fresult = np.array(())
    for i in range(0, len(a)):
        b = a.loc[i].argmax()

        if b == 0:
            c = 'normal'
        elif b == 1:
            c = 'dos'
        elif b == 2:
            c = 'u2r'
        elif b == 3:
            c = 'r2l'
        elif b == 4:
            c = 'probe'

        Fresult = np.append(Fresult, c)

        print(i)
    print(Fresult)
    return Fresult

