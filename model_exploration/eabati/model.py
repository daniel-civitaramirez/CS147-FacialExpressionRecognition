import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.python.keras import models
import sys 
sys.path.append("/Users/elenaabati/Desktop/dl/CS147-FacialExpressionRecognition/model_exploration/eabati/preprocess.py")
from preprocess import get_data, split_data

def init_model():
    num_features = 64
    im_width = 48
    im_height = 48
    num_classes = 7 

    model = models.Sequential()
    model.add(Conv2D(num_features*2*2, (3, 3), activation='relu', input_shape=(im_width, im_height, 1), padding="SAME"))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(num_features*2*2, (3, 3), activation='relu', padding="SAME"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(num_features*2, (3, 3), activation='relu', padding="SAME"))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    # model.add(Conv2D(num_features*2, (3, 3), activation='relu'), padding="SAME")
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(num_features, (3, 3), activation='relu', padding="SAME"))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(num_features, (3, 3), activation='relu', padding="SAME"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256*2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(128*2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, X_train, Y_train, file_name):
    """
    Trains and saves the model.
    Inputs: 
    model -> Uncompiled CNN classifier
    X_train -> training points used in the CNN
    Y_train -> training labels used in the CNN
    file_name -> Name of the file that the model weights are saved as
    Output: None, but a file is saved locallys
    """

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1, validation_split=0.2)
    model.save(file_name)
  
if __name__ == '__main__':
    train_data, test_data = get_data()
    train_x, train_y = split_data(train_data, "train")
    test_x, test_y = split_data(test_data, "test")
    model_1 = init_model()
    train_model(model_1, train_x, train_y, "hello_model")














