
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from regularization import RescaleLayer, DataAugmentLayer

IMG_DIM = 48
EMOTION_CLASSIFICATION = {0: 'Angry', 1: 'Digust', 2: 'Fear',
                          3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def ConvultionLayer():
    return Sequential(
        [
            layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=256, kernel_size=(3, 3),strides=(1, 1), padding='SAME', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=512, kernel_size=(3, 3),strides=(1, 1), padding='SAME', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Dropout(rate=0.2),
        ]
    )

def HiddenLayer():
    return Sequential(
        [
            layers.Dense(units=256, activation='relu'),
            layers.BatchNormalization(),

            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
        
        ]
    )

def generateModel(num_emotion=7):
    return Sequential(
        [
            keras.Input(shape=(IMG_DIM, IMG_DIM, 1)),
            RescaleLayer(),
            DataAugmentLayer(),
            ConvultionLayer(),
            layers.Flatten(),
            HiddenLayer(),
            layers.Dense(units=num_emotion, activation='softmax')
        ]
    )

def trainModel(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=64):
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1, validation_data=(x_val, y_val))
    return model, history

def testModel(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    return y_pred, y_true

def saveModel(model, file_name='cnn_weights.h5'):
    model.save(file_name)

def loadModel(filepath='cnn_weights.h5'):
    return keras.models.load_model(filepath)

def predictEmotion(model, image):
    image = tf.keras.utils.img_to_array(image)
    predicition = model.predict(image)[0]
    prob = np.max(predicition)
    emotion_index = np.argmax(predicition)
    emotion_label = EMOTION_CLASSIFICATION[emotion_index]
    return emotion_index, emotion_label, prob

