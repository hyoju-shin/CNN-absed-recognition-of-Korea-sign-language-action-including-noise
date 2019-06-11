"""

숫자 예제
https://github.com/raghakot/keras-vis/blob/master/examples/vggnet/activation_maximization.ipynb
https://github.com/raghakot/keras-vis
https://raghakot.github.io/keras-vis/

> 완성.. 이걸로 사용!!!

"""


from __future__ import print_function


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input

from keras import backend as K

"""
reference 1(17, IEEE) 모델
https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import backend as K
import tensorflow as tf
from keras import applications as app

sess = tf.Session()
K.set_session(sess)




def dataset():
    # Read training and test data files
    train = pd.read_csv("../dataFiles/train_337_datafile2.csv", encoding='utf-8', engine='python').values
    validation = pd.read_csv("../dataFiles/test_84_datafile2.csv", encoding='utf-8', engine='python').values

    # Reshape and normalize training data
    # trainX = train[:, 1:].reshape(train.shape[0],1,128, 1152).astype( 'float32' )
    trainX = train[:, 2:147458].reshape(train.shape[0], 1, 128, 1152).astype('float32')  # (2~147457)
    X_train = trainX / 255.0
    y_train = train[:, 1]  # lable column is [1]

    # Reshape and normalize test data
    # testX = test[:,1:].reshape(test.shape[0],1, 128, 1152).astype('float32')
    testX = validation[:, 2:147458].reshape(validation.shape[0], 1, 128, 1152).astype('float32')
    X_test = testX / 255.0
    y_test = validation[:, 1]

    # from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn import preprocessing

    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    return X_train, y_train, X_test, y_test




# return model
def get_model():
    model = Sequential()
    K.set_image_dim_ordering('th')

    # make the CNN model
    # layer 1: [filter = 16*16, channel =1, filter 수 = 32], activation fun = ReLu
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 128, 1152), activation='relu'))  # nb_filter, nb_row, nb_col
 #   model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4)))  # pooling1 : MaxPooling
#    model.add(AveragePooling2D(pool_size=(4, 4)))  # pooling1 : MaxPooling

#    model.add(Dropout(0.2))  # size 커서 OOM error 생겨서 넣어 봄... and overfitiing 방지

    # layer 2: [filter = 9*9, channel =1, filter 수 = 32], activation fun = ReLu
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(BatchNormalization())
#    model.add(Convolution2D(64, 3, 3, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling1 : MaxPooling
#    model.add(AveragePooling2D(pool_size=(2, 2)))  # pooling2 : AveragePooling


    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling1 : MaxPooling
 #   model.add(AveragePooling2D(pool_size=(2, 2)))  # pooling2 : AveragePooling

    '''

    # layer 3: [filter = 5*5, channel =1, filter 수 = 32], activation fun = ReLu
   # model.add(Convolution2D(256, 3, 3, activation='relu'))
 #   model.add(BatchNormalization())
#    model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling1 : MaxPooling
  #  model.add(AveragePooling2D(pool_size=(2, 2)))  # pooling1 : MaxPooling
#    model.add(Dropout(0.3))  # size 커서 OOM error 생겨서 넣어 봄...

#    model.add(Convolution2D(256, 3, 3, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
 #   model.add(Convolution2D(256, 3, 3, activation='relu'))
    #   model.add(BatchNormalization())



    '''


    model.add(Flatten())

    # model.add(Dense(24576, activation= 'relu' ))
#    model.add(Dense(4096, activation= 'relu' ))
    model.add(Dense(625, activation='relu'))
    model.add(Dense(256, activation='relu'))  # 2^8
    model.add(Dense(10, activation='softmax', name='preds'))

    return model


# https://tykimos.github.io/2017/09/24/Custom_Metric/
def single_class_precision(interesting_class_id):
    # 특정 클래스에 대한 정밀도를 평가하는 함수. 여러개의 클래스를 하나의 함수로 사용할 수 있게 interesting_class_id 인자 사용
    def prec(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        precision_mask = K.cast(K.equal(class_id_pred, interesting_class_id), 'int32')
        class_prec_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * precision_mask
        class_prec = K.cast(K.sum(class_prec_tensor), 'float32') / K.cast(K.maximum(K.sum(precision_mask), 1),
                                                                          'float32')
        return class_prec

    return prec


def single_class_recall(interesting_class_id):
    # 클래스 별로 확인 할 때, 정밀도와 재현율 파악도 도움이 됨
    # 특정 클래스에 대한 재현율 평가
    def recall(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        recall_mask = K.cast(K.equal(class_id_true, interesting_class_id), 'int32')
        class_recall_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * recall_mask
        class_recall = K.cast(K.sum(class_recall_tensor), 'float32') / K.cast(K.maximum(K.sum(recall_mask), 1),
                                                                              'float32')
        return class_recall

    return recall


def result(score, y_test, predic):
    trueD = []
    predicD = np.array(predic)

    for i in range(y_test.shape[0]):
        datum = y_test[i]
        decoded_datum = np.argmax(y_test[i])  # decode(y_test[i])
        trueD.append(decoded_datum)
    #    print('decoded datum: %s' % decoded_datum)

    print('===' * 30)
    print("score : ", score)
    print('test(validation) data : ', len(y_test))
    print('predicD: ', predicD)
    print('trueD : ', trueD)

    count = 0
    for i in range(len(trueD)):
        if (trueD[i] == predicD[i]):
            count += 1

    print('---' * 30)

    print(" 총 맞춘 예측 갯수 : ", count)
    print(" 틀린 예측 갯수 : ", len(trueD) - count)
    print('===' * 30)


##############


def train_visual(hist):
    # 학습과정 시각화
    import matplotlib.pyplot as plt

    plt.plot(hist.history['prec'], label='precision 1')
    plt.plot(hist.history['prec_1'], label='precision 1')
    plt.plot(hist.history['prec_2'], label='precision 2')
    plt.plot(hist.history['prec_3'], label='precision 3')
    plt.plot(hist.history['prec_4'], label='precision 4')
    plt.plot(hist.history['prec_5'], label='precision 5')
    plt.plot(hist.history['prec_6'], label='precision 6')
    plt.plot(hist.history['prec_7'], label='precision 7')
    plt.plot(hist.history['prec_8'], label='precision 8')
    plt.plot(hist.history['prec_9'], label='precision 9')

    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend(loc='lower right')
    plt.show()

    ############

    plt.plot(hist.history['recall'], label='recall 0')
    plt.plot(hist.history['recall_1'], label='recall 1')
    plt.plot(hist.history['recall_2'], label='recall 2')
    plt.plot(hist.history['recall_3'], label='recall 3')
    plt.plot(hist.history['recall_4'], label='recall 4')
    plt.plot(hist.history['recall_5'], label='recall 5')
    plt.plot(hist.history['recall_6'], label='recall 6')
    plt.plot(hist.history['recall_7'], label='recall 7')
    plt.plot(hist.history['recall_8'], label='recall 8')
    plt.plot(hist.history['recall_9'], label='recall 9')
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.legend(loc='lower right')
    plt.show()


def result_visual(score):
    import numpy as np

    metrics = np.array(score[2:])
    idx = np.linspace(0, 19, 20)
    precision = metrics[(idx % 2) == 0]
    recall = metrics[((idx + 1) % 2) == 0]

    import matplotlib.pyplot as plt

    N = 10
    ind = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots()
    prec_bar = ax.bar(ind, precision, width, color='r')
    recall_bar = ax.bar(ind + width, recall, width, color='y')

    ax.set_ylabel('Scores')
    ax.set_title('Precision and Recall')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(
        ('0 : 감사', '1: 괜찮다', '2: 사랑', '3: 기쁘다', '4: 미안', '5: 안되다', '6: 안녕', '7: 늦다', '8: 만나다', '9: 부끄럽다'))

    ax.legend((prec_bar[0], recall_bar[0]), ('Precision', 'Recall'))

    plt.show()


def cnnfeature_vis(model):
    from vis.visualization import visualize_activation
    from vis.utils import utils
    from keras import activations

    from matplotlib import pyplot as plt
    # %matplotlib inline
    plt.rcParams['figure.figsize'] = (18, 6)

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'preds')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    # This is the output node we want to maximize.
    filter_idx = 0
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
    plt.imshow(img[..., 0])

    for output_idx in np.arange(10):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))
        plt.figure()
        plt.title('Networks perception of {}'.format(output_idx))
        plt.imshow(img[..., 0])

    # Visualizations without swapping softmax
    # Swap linear back with softmax
    model.layers[layer_idx].activation = activations.softmax
    model = utils.apply_modifications(model)

    for output_idx in np.arange(10):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        # Visualizations without swapping softmax
        img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))
        plt.figure()
        plt.title('Networks perception of {}'.format(output_idx))
        plt.imshow(img[..., 0])
        plt.show()

################



def main():
    model = get_model()

    X_train, y_train, X_test, y_test = dataset()

    # Compile model
#    model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])  # base




    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                              single_class_precision(0), single_class_recall(0),
                                                                              single_class_precision(1), single_class_recall(1),
                                                                              single_class_precision(2), single_class_recall(2),
                                                                              single_class_precision(3), single_class_recall(3),
                                                                              single_class_precision(4), single_class_recall(4),
                                                                              single_class_precision(5), single_class_recall(5),
                                                                              single_class_precision(6), single_class_recall(6),
                                                                              single_class_precision(7), single_class_recall(7),
                                                                              single_class_precision(8), single_class_recall(8),
                                                                              single_class_precision(9), single_class_recall(9)])



 #   model.fit(X_train, y_train, epochs=5, batch_size=512)  # mini-batch : 64, 128, 256, 512 size(memory size에 맞춰) epochs=300
    hist = model.fit(X_train, y_train, epochs=50, batch_size=128)  # mini-batch : 64, 128, 256, 512 size(memory size에 맞춰) epochs=300
    # small batch = 8


    # model save
    model.save('./models/model_1-2.h5')


    print('model summary : ', model.summary())  # model에 대한 정보(summary) 출력



    score = model.evaluate(X_test, y_test, batch_size=128)
    # predicD = model.predict(testX, verbose=1)
    predic = model.predict_classes(X_test, verbose=1)


    result(score, y_test, predic)
    train_visual(hist)
    result_visual(score)

    #cnnfeature_vis(model)




    K.clear_session()  # 모델 닫기





main()



'''
if __name__ == '__main__':

    result(score, y_test, predic)
    train_visual(hist)
    result_visual(score)


'''

