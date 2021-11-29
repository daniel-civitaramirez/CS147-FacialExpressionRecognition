import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from model_param import ConvulationParameters, MaxPoolParameters


def generateCNN3(conv_param1, conv_param2, conv_param3, conv_param4, maxpool_param, dense_unit, dropout_rate, emotion_categories=7):
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
