import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from model_param import ConvulationParameters, MaxPoolParameters

CONV2D_1 = ConvulationParameters(features=64, kernel=(3, 3), strides=(1, 1), padding='same')
CONV2D_2 = ConvulationParameters(features=128, kernel=(5, 5), strides=(1, 1), padding='same')
CONV2D_3 = ConvulationParameters(features=512, kernel=(3, 3), strides=(1, 1), padding='same')
CONV2D_4 = ConvulationParameters(features=512, kernel=(3, 3), strides=(1, 1), padding='same')
MAXPOOL = MaxPoolParameters(pool=(2, 2), stride=(2, 2))
DENSE_UNIT = 256
DROPOUT_RATE = 0.2
EMOTION_CATEGORIES = 7


def generateCNN3(conv_param1=CONV2D_1, conv_param2=CONV2D_2, conv_param3=CONV2D_3, conv_param4=CONV2D_4, maxpool_param=MAXPOOL, 
                 dense_unit=DENSE_UNIT, dropout_rate=DROPOUT_RATE, emotion_categories=EMOTION_CATEGORIES):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=conv_param1.features, kernel_size=conv_param1.kernel, strides=conv_param1.strides,
                            padding=conv_param1.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool,
                               strides=maxpool_param.stride))

    model.add(layers.Conv2D(filters=conv_param2.features, kernel_size=conv_param2.kernel, strides=conv_param2.strides,
                            padding=conv_param2.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool,
                               strides=maxpool_param.stride))

    model.add(layers.Conv2D(filters=conv_param3.features, kernel_size=conv_param3.kernel, strides=conv_param3.strides,
                            padding=conv_param3.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool,
                               strides=maxpool_param.stride))
    
    model.add(layers.Conv2D(filters=conv_param4.features, kernel_size=conv_param4.kernel, strides=conv_param4.strides,
                            padding=conv_param4.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool,
                               strides=maxpool_param.stride))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=2*dense_unit))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(units=dense_unit))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(units=emotion_categories, activation='softmax'))

    return model
