import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def RescaleLayer():
    return Sequential(
        [
            layers.Rescaling(scale=1./255)
        ]
    )

def DataAugmentLayer(): 
    return Sequential(
        [
            layers.RandomFlip(mode="horizontal"),
            layers.RandomRotation(factor=0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=0.1),
        ]
    )   
