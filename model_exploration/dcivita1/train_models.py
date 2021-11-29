import tensorflow as tf
from cnn3 import generateCNN3
from cnn2 import generateCNN2
from cnn1 import generateCNN
from model_param import ConvulationParameters, MaxPoolParameters
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models

import sys
sys.path.append('../../data')
from preprocess import get_data

DEFAULT_TRAIN_FILEPATH = ['../../data/train_data_1.gz', '../../data/train_data_2.gz',
                          '../../data/train_data_3.gz', '../../data/train_data_4.gz']
DEFAULT_TEST_FILEPATH = '../../data/test_data.gz'


def train_model(model, x_train, y_train):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2)
    return history.history['val_accuracy'][-1], history.history['val_loss'][-1]

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"CNN1", "CNN2", "CNN3"}:
        print("USAGE: python train_models.py <CNN#>")
        exit()
    if sys.argv[1] == "CNN1":
        model = generateCNN()
    elif sys.argv[1] == "CNN2":
        model = generateCNN2()
    elif sys.argv[1] == "CNN3":
        model = generateCNN3()
    
    print('Getting Data...')
    x_train, y_train, x_test, y_test = get_data(DEFAULT_TRAIN_FILEPATH, DEFAULT_TEST_FILEPATH)
    print('Training Model...')
    final_validation_acc, final_val_loss = train_model(model, x_train, y_train)
    print('FINAL VALIDATION LOSS: ', final_val_loss)
    print('FINAL VALIDATION ACCURACY: ', final_validation_acc)
