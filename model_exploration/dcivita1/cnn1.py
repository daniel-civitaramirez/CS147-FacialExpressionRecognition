import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from model_param import ConvulationParameters, MaxPoolParameters

def generateCNN(conv_param, maxpool_param, dropout_rate, dense_unit, emotion_categories=7):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=conv_param.features, kernel_size=conv_param.kernel, strides=conv_param.strides, 
                            padding=conv_param.padding, input_shape=(48, 48, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool, strides=maxpool_param.stride))

    model.add(layers.Conv2D(filters=2*conv_param.features, kernel_size=conv_param.kernel, strides=conv_param.strides,
                            padding=conv_param.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool,strides=maxpool_param.stride))

    model.add(layers.Conv2D(filters=2*2*conv_param.features, kernel_size=conv_param.kernel, strides=conv_param.strides,
                            padding=conv_param.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool, strides=maxpool_param.stride))

    model.add(layers.Conv2D(filters=2*2*2*conv_param.features, kernel_size=conv_param.kernel, strides=conv_param.strides,
                            padding=conv_param.padding))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.MaxPool2D(pool_size=maxpool_param.pool, strides=maxpool_param.stride))
    
    model.add(layers.Flatten())

    model.add(layers.Dense(units=2*2*dense_unit))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(units=2*dense_unit))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(units=dense_unit))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(units=emotion_categories, activation='softmax'))
    
    return model
