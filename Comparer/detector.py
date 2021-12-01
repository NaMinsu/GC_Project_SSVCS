import numpy as np
#import tensorflow as tf
#import yolov4
import comparer
import shutil
import os

# TODO: make object detection process using yolo-v4 to find new object's position value

def get_comparer(video): 
    comparer.video_cap(video)
    frame,mse_val,ssim_val = comparer.comapare_image(video)

    if not os.path.exists("./" + video +"_edit"):
        os.makedirs("./" + video+"_edit")

    for i in range(0,len(frame)):
        shutil.copy('./%s/frame%d.png'%(video,frame[i]),'./%s'%video+"_edit")
        print('frame 번호 : %s'%frame[i])
        print('mse 값: %s'%mse_val[i])
        print('ssim 값 :%s'%ssim_val[i])


if __name__ == '__main__':
    video = input ('enter name of video: ')
    get_comparer(video)

