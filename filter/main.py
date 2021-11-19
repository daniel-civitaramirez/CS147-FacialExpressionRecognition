import os.path
from os import path
from tensorflow.python.keras import models
import os
import cv2
import numpy as np
from pixellib.tune_bg import alter_bg



def live_detection():
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(haar_model)
    camera = cv2.VideoCapture(0)
    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    while True:
        ## if this takes to long to have as livestream use the camera to take a picture and then emotion 
        ## is detected and and picture is shown with changed background associated with the emotion. 
        _, frame = camera.read()
        frame_copy = cv2.resize(np.copy(frame), (256,256),0,0, interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x,y,width,height) = cascade.detectMultiScale(gray_frame, 1.3, 7, minSize=(35, 35))[0]

        cropped_face = gray_frame[int(y):int(y+height),x:int(x+width)]
        scaled_face = cv2.resize(cropped_face, (48,48), 0,0, interpolation=cv2.INTER_AREA) / 255

        ##associate color to emotion -- pass the correct color to the output line so that it can be 
        ##changed and shown. 

        cv2.imwrite('image.jpg', frame_copy)

        output = change_bg.color_bg('image.jpg', colors = (0, 128, 0))
        cv2.imshow("Image", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):   
            break


def main():
    live_detection()

if __name__ == '__main__':
    main() 