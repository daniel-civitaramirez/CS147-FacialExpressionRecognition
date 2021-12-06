import os.path
from os import path
from numpy.lib.npyio import load
from tensorflow.python.keras import models
import os
import cv2
import numpy as np
from pixellib.tune_bg import alter_bg
from PIL import Image, ImageDraw, ImageFont
from filter import filter_application
import sys
sys.path.append('../model')
from cnn import predictEmotion, loadModel
import tensorflow as tf


def live_detection(model):
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(haar_model)
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
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
    scaled_face = tf.reshape(tf.convert_to_tensor(cv2.resize(cropped_face, (48,48), 0,0, interpolation=cv2.INTER_AREA)/255), [1,48,48,1])


    emotion_index,emotion,_ = predictEmotion(model, scaled_face)
    print("emotion selected:", emotion)

    filter_application(img_name, emotion_index)
    print("filter applied!")
    image = Image.open('output_im.png') 
    width,height = image.size
    bi = Image.new('RGBA', (width+10, height+(height//5)), 'white')
    bi.paste(image, (5,5,(width+5), (height+5)))

    caption= emotion
    font= ImageFont.truetype("arial.ttf", 45)
    w,h = font.getsize(caption)
    draw = ImageDraw.Draw(bi)
    draw.text(((width-w)/2, (height+((height/5)-h)/2)), caption, font=font, fill="black")
    bi.save("final_output.png")
    bi.show()


def main():
    file_name = "../model/cnn_weights.h5"
    model = loadModel(file_name)
    live_detection(model)

if __name__ == '__main__':
    main() 