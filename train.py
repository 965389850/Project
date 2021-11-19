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
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, Conv1D, \
    MaxPool1D


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




# 读取数据
data = pd.read_csv('./数据集/KDDTrain+.csv', names=range(0, 43))

x_train_savepath = './datasave1/KDD20_Preprocessing.csv'
y_train_sc_savepath = './datasave1/KDD20_Preprocessing_label_sc.csv'
y_train_mc_savepath = './datasave1/KDD20_Preprocessing_label_mc.csv'
if os.path.exists(x_train_savepath) and os.path.exists(y_train_sc_savepath) and os.path.exists(y_train_mc_savepath):
    print('-------------------Load Datasets-------------------')
    result = pd.read_csv(x_train_savepath, header=None)
    label_sc = pd.read_csv(y_train_sc_savepath, header=None)
    label_mc = pd.read_csv(y_train_mc_savepath, header=None)
else:
    print('-------------------Generate Datasets-------------------')
    # 预处理
    dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
    u2r = ['buffer_overflow', 'loadmodule', 'rootkit']
    r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'perl']
    probe = ['ipsweep', 'nmap', 'portsweep', 'satan']
    # 训练集
    label, result = prepocessing(data, 1, 3)
    result = df(result)
    label = df(label)
    label_sc = np.array([])
    label_mc = np.array([])
    # 二分类标签
    for i in range(0, len(label.iloc[:, 0])):
        if label.iloc[i, 0] == 'normal':
            a = 0
            label_sc = np.append(label_sc, a)
        else:
            a = 1
            label_sc = np.append(label_sc, a)

    # 多分类标签
    for i in range(0, len(label.iloc[:, 0])):
        if label.iloc[i, 0] in ['normal']:
            a = 0
            label_mc = np.append(label_mc, a)

        elif label.iloc[i, 0] in dos:
            a = 1
            label_mc = np.append(label_mc, a)

        elif label.iloc[i, 0] in u2r:
            a = 2
            label_mc = np.append(label_mc, a)

        elif label.iloc[i, 0] in r2l:
            a = 3
            label_mc = np.append(label_mc, a)

        elif label.iloc[i, 0] in probe:
            a = 4
            label_mc = np.append(label_mc, a)

        else:
            a = 5
            label_mc = np.append(label_mc, a)

    label_mc = df(label_mc)
    label_sc = df(label_sc)
    # 存储数据
    result.to_csv(x_train_savepath, index=False, header=None)
    label_sc.to_csv(y_train_sc_savepath, index=False, header=None)
    label_mc.to_csv(y_train_mc_savepath, index=False, header=None)

result = result.drop([100],axis = 1)
result.columns = range(0,121)

result.to_csv('./datasave1/CV.csv',index=False,header=None)

x_train_savepath = './datasave1/KDD_x_train.npy'
x_train_cnn_savepath = './datasave1/KDD_x_train_cnn.npy'
y_train_savepath = './datasave1/KDD_y_train.npy'

x_train = result
y_train = label_mc
x_train=np.array(x_train)
x_train=x_train.reshape(-1,1,121)
x_train_cnn = x_train.reshape(-1,11,11)

np.save(x_train_savepath, x_train)
np.save(x_train_cnn_savepath, x_train_cnn)
np.save(y_train_savepath, y_train)

'''
x_train = np.load(x_train_savepath)
x_train_cnn = np.load(x_train_cnn_savepath)
y_train = np.load(y_train_savepath)
'''

n = round(len(x_train)/3)
x_train1 = x_train[0:n]
x_train2 = x_train[n:2*n]
x_train3 = x_train[2*n:]
y_train1 = y_train[0:n]
y_train2 = y_train[n:2*n]
y_train3 = y_train[2*n:]

x_train_lstm1 = np.concatenate((x_train2,x_train3), 0)
y_train_1 = np.concatenate((y_train2,y_train3), 0)
x_test_lstm1 = x_train1
y_test_1 = y_train1
x_train_cnn1 = x_train_lstm1.reshape(-1,11,11)
x_train_cnn1 = x_train_cnn1[:,:,:,np.newaxis]
x_test_cnn1 = x_test_lstm1.reshape(-1,11,11)
x_test_cnn1 = x_test_cnn1[:,:,:,np.newaxis]

x_train_lstm2 = np.concatenate((x_train1,x_train3), 0)
y_train_2 = np.concatenate((y_train1,y_train3), 0)
x_test_lstm2 = x_train2
y_test_2 = y_train2
x_train_cnn2 = x_train_lstm2.reshape(-1,11,11)
x_train_cnn2 = x_train_cnn2[:,:,:,np.newaxis]
x_test_cnn2 = x_test_lstm2.reshape(-1,11,11)
x_test_cnn2 = x_test_cnn2[:,:,:,np.newaxis]


x_train_lstm3 = np.concatenate((x_train1,x_train2), 0)
y_train_3 = np.concatenate((y_train1,y_train2), 0)
x_test_lstm3 = x_train3
y_test_3 = y_train3
x_train_cnn3 = x_train_lstm3.reshape(-1,11,11)
x_train_cnn3 = x_train_cnn3[:,:,:,np.newaxis]
x_test_cnn3 = x_test_lstm3.reshape(-1,11,11)
x_test_cnn3 = x_test_cnn3[:,:,:,np.newaxis]

##----------  模  型  训  练  -----------##

a = input("训练:[y/n]")
if a == 'y':
    a1 = input("训练模型:[0：全部；1：LSTM；2：CNN；3：BiLSTMattention；4：CNNattention；]")
    a2 = input("重新预测数据?:[y/n]")
    predcnn = df()
    predlstm = df()
    predlstmatt = df()
    predcnnatt = df()
    for i in range(0, 3):
        if i == 0:
            x_train = x_train_lstm1
            x_train_cnn = x_train_cnn1
            y_train = y_train_1
            x_test = x_test_lstm1
            x_test_cnn = x_test_cnn1
            y_test = y_test_1

        if i == 1:
            x_train = x_train_lstm2
            x_train_cnn = x_train_cnn2
            y_train = y_train_2
            x_test = x_test_lstm2
            x_test_cnn = x_test_cnn2
            y_test = y_test_2

        if i == 2:
            x_train = x_train_lstm3
            x_train_cnn = x_train_cnn3
            y_train = y_train_3
            x_test = x_test_lstm3
            x_test_cnn = x_test_cnn3
            y_test = y_test_3

            ##-------------  L S T M  -------------##

        model = modlstm()
        if a == 'y' and a1 in ['0', '1']:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 训练时选择哪种优化器
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 训练时选择哪种损失函数
                          metrics=['sparse_categorical_accuracy'])  # 选择哪种评测指标
            history = model.fit(x_train, y_train, batch_size=50, epochs=60)
            model.save('./modle/LSTM' + str(i + 1) + '.hdf5')
        elif a2 == 'y':
            pred = model.predict(x_test)
            pred = df(pred)
            predlstm = predlstm.append(pred)
        del model

        ##-------------   C N N   -------------##
        model = modcnn()
        if a == 'y' and a1 in ['0', '2']:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 训练时选择哪种优化器
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 训练时选择哪种损失函数
                          metrics=['sparse_categorical_accuracy'])  # 选择哪种评测指标
            history = model.fit(x_train_cnn, y_train, batch_size=50, epochs=60)
            model.save('./modle/CNN' + str(i + 1) + '.hdf5')
        elif a2 == 'y':
            pred = model.predict(x_test_cnn)
            pred = df(pred)
            predcnn = predcnn.append(pred)
        del model

        ##-------------   BiLSTM_attention   -------------##
        model = BiLSTM_attention(LSTM_num=128)
        if a == 'y' and a1 in ['0', '3']:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 训练时选择哪种优化器
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 训练时选择哪种损失函数
                          metrics=['sparse_categorical_accuracy'])  # 选择哪种评测指标
            history = model.fit(x_train, y_train, batch_size=50, epochs=80, validation_split=0.03, validation_freq=1,
                                shuffle=True)
            model.save('./modle/BiLSTM_attention_L2' + str(i + 1) + '.hdf5')
        elif a2 == 'y':
            pred = model.predict(x_test)
            pred = df(pred.reshape(-1, 5))
            predlstmatt = predlstmatt.append(pred)
        del model

        ##-------------   CNN_attention   -------------##
        model = CNN_attention(dense=90)
        if a == 'y' and a1 in ['0', '4']:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 训练时选择哪种优化器
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 训练时选择哪种损失函数
                          metrics=['sparse_categorical_accuracy'])  # 选择哪种评测指标
            history = model.fit(x_train_cnn, y_train, batch_size=50, epochs=80, validation_split=0.03,
                                validation_freq=1, shuffle=True)
            model.save('./modle/CNN_attention_L2' + str(i + 1) + '.hdf5')
        elif a2 == 'y':
            pred = model.predict(x_test_cnn)
            pred = df(pred.reshape(-1, 5))
            predcnnatt = predcnnatt.append(pred)
        del model

    if a2 == 'y':
        predlstm.to_csv('./predlstm.csv', index=False, header=None)
        predcnn.to_csv('./predlstm.csv', index=False, header=None)
        predlstmatt.to_csv('./predlstmatt.csv', index=False, header=None)
        predcnnatt.to_csv('./predcnnatt.csv', index=False, header=None)

        predlstm.columns = [0, 1, 2, 3, 4]
        predcnn.columns = [0, 1, 2, 3, 4]
        b = df(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        for i in range(0, 10):
            if i < 5:
                b[i] = predlstm[i]
            if i >= 5:
                b[i] = predcnn[i - 5]
        b.to_csv('./MLPdata.csv', index=False, header=None)

        predlstmatt.columns = [0, 1, 2, 3, 4]
        predcnnatt.columns = [0, 1, 2, 3, 4]
        c = df(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        for i in range(0, 10):
            if i < 5:
                c[i] = predlstmatt[i]
            if i >= 5:
                c[i] = predcnnatt[i - 5]
        c.to_csv('./MLPdata_att.csv', index=False, header=None)


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
model=tf.keras.models.Sequential([
        Dense(64,'relu'),
        Dropout(0.5),
        Dense(16,'relu'),
        Dropout(0.5),
        Dense(5,activation='softmax')
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )

history = model.fit(pred, y_train, batch_size=50, epochs=80,validation_split=0.03, validation_freq=1,shuffle=True)
model.save('./modle/MLP_attention_test.hdf5')
model.summary()




