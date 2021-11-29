import os.path
from os import path
from tensorflow.python.keras import models
import os
import cv2
import numpy as np
from pixellib.tune_bg import alter_bg
from PIL import Image


def filter_application(image_name, emotion):
    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    color_dict = {
        0 : (128,0,0), 1: (0,128,0), 2: (0,0,0), 3: (255,255,0), 4: (0,0,128), 5: (255,255,255), 6: (128,128,128)
    }
    change_bg.color_bg(image_name, colors = color_dict[emotion], output_image_name='output_im.png')
