import sys
sys.path.append('../../data')

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from preprocess import get_data
from model_param import ConvulationParameters, MaxPoolParameters
from cnn1 import generateCNN
from cnn2 import generateCNN2
from cnn3 import generateCNN3


def write_to_file(final_accuracy, final_loss, val_accuracy, val_loss, learning_rate,
                  batch_size, conv_param, maxpool_param, dropout_rate, dense_unit):

    output_string = f"CNN: Test Accuracy is {final_accuracy}, Test Loss is {final_loss} \n"
    output_string += f"Val Accuracy is {val_accuracy}, Val Loss is {val_loss} \n"
    output_string += f"Batch size - {batch_size}, Learning Rate - {learning_rate}, Dropout Rate - {dropout_rate}, Dense - {dense_unit} \n"
    output_string += f"Conv2d - {conv_param.GetString()}, MaxPool2D - {maxpool_param.GetString()}. \n\n\n"

    file1 = open('cnn1_tests.txt', 'a')
    file1.write(output_string)
    file1.close()

def explore_models(x, y):
    conv2ds = [ConvulationParameters(features=16, kernel=(3, 3), strides=(1, 1), padding='same'),
               ConvulationParameters(features=32, kernel=(3, 3), strides=(1, 1), padding='same'),
               ConvulationParameters(features=16, kernel=(5, 5), strides=(1, 1), padding='same'),
               ConvulationParameters(features=32, kernel=(5, 5), strides=(1, 1), padding='same')]
    maxpools = [MaxPoolParameters(pool=(2, 2), stride=(2, 2))]
    dropout_rates = [0.1, 0.2]
    dense_units = [16, 32]

    batch_sizes = [64]
    learning_rates = [0.001, 0.01]

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for conv_param in conv2ds:
                for maxpool_param in maxpools:
                    for dropout_rate in dropout_rates:
                        for dense_unit in dense_units:
                            model = generateCNN(conv_param, maxpool_param, dropout_rate, dense_unit)
                            model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
                            history = model.fit(x=x, y=y, batch_size=batch_size, epochs=30, validation_split=0.2, verbose=2)
                            final_accuracy, final_loss = history.history['accuracy'][-1], history.history['loss'][-1]
                            val_accuracy, val_loss = history.history['val_accuracy'][-1], history.history['val_loss'][-1]
                            write_to_file(final_accuracy, final_loss, val_accuracy, val_loss, learning_rate, 
                                        batch_size, conv_param, maxpool_param,dropout_rate, dense_unit)


def write_to_file2(final_accuracy, final_loss, val_accuracy, val_loss,
                   learning_rate, batch_size, maxpool_param, dense_unit,
                   conv_param1, conv_param2, conv_param3):

    output_string = f"CNN: Test Accuracy is {final_accuracy}, Test Loss is {final_loss} \n"
    output_string += f"Val Accuracy is {val_accuracy}, Val Loss is {val_loss} \n"
    output_string += f"Batch size - {batch_size}, Learning Rate - {learning_rate}, Dense - {dense_unit} \n"
    output_string += f"Conv2D-1 - {conv_param1.GetString()} \n"
    output_string += f"Conv2D-2 - {conv_param2.GetString()} \n"
    output_string += f"Conv2D-3 - {conv_param3.GetString()} \n"

    file1 = open('cnn2_tests.txt', 'a')
    file1.write(output_string)
    file1.close()

def explore_model2(x, y):
    conv_param1 = ConvulationParameters(features=64, kernel=(3, 3), strides=(1, 1), padding='same')

    conv2ds_2 = [ConvulationParameters(features=128, kernel=(5, 5), strides=(1, 1), padding='same'), 
                ConvulationParameters(features=128, kernel=(3, 3), strides=(1, 1), padding='same')]

    conv2ds_3 = [ConvulationParameters(features=512, kernel=(3,3), strides=(1,1), padding='same'), 
                ConvulationParameters(features=256, kernel=(3, 3), strides=(1, 1), padding='same')]

    maxpool_param = MaxPoolParameters(pool=(2, 2), stride=(2, 2))
    dense_units = [128, 256]
    batch_size = 64
    learning_rate = 0.01

    for dense_unit in dense_units:
        for conv_param2 in conv2ds_2:
            for conv_param3 in conv2ds_3:
                model = generateCNN2(conv_param1, conv_param2, conv_param3, maxpool_param, dense_unit)
                model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(x=x, y=y, batch_size=batch_size, epochs=1, validation_split=0.2, verbose=2)
                final_accuracy, final_loss = history.history['accuracy'][-1], history.history['loss'][-1]
                val_accuracy, val_loss = history.history['val_accuracy'][-1], history.history['val_loss'][-1]
                write_to_file2(final_accuracy, final_loss, val_accuracy, val_loss, learning_rate, batch_size,
                                maxpool_param, dense_unit, conv_param1, conv_param2, conv_param3)


def write_to_file3(final_accuracy, final_loss, val_accuracy, val_loss,
                   learning_rate, batch_size, maxpool_param, dense_unit,
                   dropout_rate, conv_param1, conv_param2, conv_param3,
                   conv_param4):

    output_string = f"CNN: Test Accuracy is {final_accuracy}, Test Loss is {final_loss} \n"
    output_string += f"Val Accuracy is {val_accuracy}, Val Loss is {val_loss} \n"
    output_string += f"Batch size - {batch_size}, Learning Rate - {learning_rate}, Dense - {dense_unit}, Dropout - {dropout_rate} \n"
    output_string += f"Conv2D-1 - {conv_param1.GetString()} \n"
    output_string += f"Conv2D-2 - {conv_param2.GetString()} \n"
    output_string += f"Conv2D-3 - {conv_param3.GetString()} \n"
    output_string += f"Conv2D-4 - {conv_param3.GetString()} \n"

    file1 = open('cnn3_tests.txt', 'a')
    file1.write(output_string)
    file1.close()


def explore_model3(x, y):
    conv_param1 = ConvulationParameters(features=64, kernel=(3, 3), strides=(1, 1), padding='same')

    conv_param2 = ConvulationParameters(features=128, kernel=(5, 5), strides=(1, 1), padding='same')

    conv2ds_3 = [ConvulationParameters(features=256, kernel=(3,3), strides=(1,1), padding='same'),
                ConvulationParameters(features=512, kernel=(3, 3), strides=(1, 1), padding='same')]

    conv_param4 = ConvulationParameters(features=512, kernel=(3, 3), strides=(1, 1), padding='same')

    maxpool_param = MaxPoolParameters(pool=(2, 2), stride=(2, 2))
    dense_units = [128, 256]
    batch_size = 64
    dropout_rate = 0.2
    learning_rate = 0.001

    for dense_unit in dense_units:
        for conv_param3 in conv2ds_3:
            model = generateCNN3(conv_param1, conv_param2, conv_param3,
                                     conv_param4, maxpool_param, dense_unit, dropout_rate)
            model.compile(optimizer=Adam(lr=learning_rate),
                              loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(x=x, y=y, batch_size=batch_size, epochs=30, validation_split=0.2, verbose=2)

            final_accuracy, final_loss = history.history['accuracy'][-1], history.history['loss'][-1]
            val_accuracy, val_loss = history.history['val_accuracy'][-1], history.history['val_loss'][-1]

            write_to_file3(final_accuracy, final_loss, val_accuracy, val_loss, learning_rate, batch_size,
                               maxpool_param, dense_unit, dropout_rate, conv_param1, conv_param2, conv_param3,
                               conv_param4)


if __name__ == "__main__":
    _DEFAULT_TRAIN_FILEPATH = ['../../data/train_data_1.gz', '../../data/train_data_2.gz',
                               '../../data/train_data_3.gz', '../../data/train_data_4.gz']
    _DEFAULT_TEST_FILEPATH = '../../data/test_data.gz'
    x_train, y_train, x_test, y_test = get_data(_DEFAULT_TRAIN_FILEPATH, _DEFAULT_TEST_FILEPATH)

    explore_model2(x_train, y_train)
    #explore_model3(x_train, y_train)
