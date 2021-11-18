import tensorflow as tf
import pandas as pd
import numpy as np
import gzip

_DEFAULT_TRAIN_FILEPATH = ['train_data_1.gz', 'train_data_2.gz', 'train_data_3.gz', 'train_data_4.gz']
_DEFAULT_TEST_FILEPATH = 'test_data.gz'


def get_data(train_data_file_paths: list = _DEFAULT_TRAIN_FILEPATH, test_data_file_path: str = _DEFAULT_TEST_FILEPATH):
    test_data = pd.read_csv(test_data_file_path, compression='gzip')




if __name__ == '__main__':
    get_data()
