from cnn import generateModel, trainModel, testModel, saveModel
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
sys.path.append('../data')
from preprocess import get_data

_DEFAULT_TRAIN_FILEPATH = ['../data/train_data_1.gz',
                           '../data/train_data_2.gz', '../data/train_data_3.gz', '../data/train_data_4.gz']
_DEFAULT_TEST_FILEPATH = '../data/test_data.gz'
_DEFAULT_VALIDATION_FILEPATH = '../data/validation_data.gz'


def viz_training_results(history, epochs=50):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('train_results.png')


def viz_test_confusion_matrix(y_pred, y_true):
    pass

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(
                                                    _DEFAULT_TRAIN_FILEPATH, 
                                                    _DEFAULT_VALIDATION_FILEPATH, 
                                                    _DEFAULT_TEST_FILEPATH)
    model = generateModel()
    model, history = trainModel(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, epochs=50)
    viz_training_results(history=history, epochs=50)

    y_pred, y_true = testModel(model=model, x_test=x_test, y_test=y_test)
    print('Model Test Accuracy: ', accuracy_score(y_true, y_test))
    viz_test_confusion_matrix(y_pred, y_true)

    saveModel(model=model)

if __name__ == "__main__":
    main()
