import os.path
from os import path
from tensorflow.python.keras import models
import os
import cv2
import numpy as np
from pixellib.tune_bg import alter_bg
from PIL import Image
from filter import filter_application


def live_detection():
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(haar_model)
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Camera")
    image_counter = 0
    while image_counter==0:
        ret, frame = camera.read()
        if not ret:
            print("failed to get frame")
            break
        cv2.imshow("Camera", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(image_counter)
            image_counter += 1
            cv2.imwrite(img_name, frame)
            final_frame = frame
            print("{} stored!".format(img_name))
    camera.release()
    cv2.destroyWindow("Camera")
    frame_copy = cv2.resize(np.copy(final_frame), (256,256),0,0, interpolation=cv2.INTER_AREA)

    gray_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
    (x,y,width,height) = cascade.detectMultiScale(gray_frame, 1.3, 7, minSize=(35, 35))[0]

    cropped_face = gray_frame[int(y):int(y+height),x:int(x+width)]
    scaled_face = cv2.resize(cropped_face, (48,48), 0,0, interpolation=cv2.INTER_AREA) / 255

    ##associate color to emotion -- pass the correct color to the output line so that it can be 
    ##changed and shown. 
    print("emotion selected!")

    filter_application(img_name, 6)
    print("filter applied!")
    image = Image.open('output_im.png') 
    image.show()
    

def main():
    live_detection()

if __name__ == '__main__':
    main() 