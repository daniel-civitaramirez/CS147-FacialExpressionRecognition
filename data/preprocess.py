import tensorflow as tf
import pandas as pd
import numpy as np
import gzip


_DEFAULT_TRAIN_FILEPATH = ['train_data_1.gz', 'train_data_2.gz', 'train_data_3.gz', 'train_data_4.gz']
_DEFAULT_TEST_FILEPATH = 'test_data.gz'


def get_data(train_data_file_paths: list = _DEFAULT_TRAIN_FILEPATH, test_data_file_path: str = _DEFAULT_TEST_FILEPATH):
    test_data = pd.read_csv(test_data_file_path, compression='gzip')
    test_data = test_data.drop(['Usage'], axis=1)
    print(test_data.shape)
    train_data_list = []
    for file in train_data_file_paths:
        train_data_1 = pd.read_csv(file, compression='gzip')
        train_data_list.append(train_data_1)
    train_data = pd.concat(train_data_list)
    print(train_data.shape)
    return train_data, test_data
    
    # now we map the emotion numbers to their label
emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def split_data(df,dataName):
    # convert pixel string to list of ints 
    df['pixels'] = df['pixels'].apply(lambda pixel_seq: [int(pixel) for pixel in pixel_seq.split()])
    data_X = tf.convert_to_tensor(np.array(df['pixels'].tolist(), dtype='float32').reshape(-1,48, 48,1)/255.0)
    # one hot enconding 
    data_Y = tf.keras.utils.to_categorical(df['emotion'], len(emotion_labels))  
    return data_X, data_Y

if __name__ == '__main__':
    train_data, test_data = get_data()
    train_x, train_y = split_data(train_data, "train")
    test_x, test_y = split_data(test_data, "test")

