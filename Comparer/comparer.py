# comparer.py 
# Written by Subin Kim (김수빈)

# comparer.py is code for image comparsion by using MSE & SSIM 
# comparer.py 는 mse와 ssim을 이용하여 image comparsion을 진행하는 코드이다.

import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


# TODO: make image comparison process using opencv library

# function for video capture
def video_cap(video):
    # load the video saved in folder name 'video'
    # video 폴더에 저장된 video 불러옴
    cap = cv2.VideoCapture('./Comparer/video/%s.mp4' % video)
    i = 0

    if not os.path.exists("./" + video):
        os.makedirs("./" + video)

    # origin과 새로 생긴 frame끼리 비교를 진행
    # ex. video 1 / origin.png 와 video / frame0~ .png 비교

    # video1폴더에 video1.mp4를 frame 30 간격? 으로 캡쳐한 이미지를 frame%s.png 형식 저장함
    # ex. frame1.png
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break

        elif ret == True:
            # 현재 기본 사진이 없기 떄문에 첫 캡쳐를 origin으로 저장
            if int(cap.get(1)) == 1:
                cv2.imwrite('./%s/origin.png' % video, frame)
            else:
                cv2.imwrite('./%s/frame%d.png' % (video, i), frame)
                i += 1

    cap.release()
    cv2.destroyAllWindows()
    return i


# mse 구하는 함수
def mse(image_a, image_b):
    # err = np.sum((image_a.astype('float') - image_b.astype('float')) ** 2)
    # err /= float(image_a.shape[0] * image_a.shape[1])
    err = np.mean((image_a - image_b) ** 2)

    return err


# origin과 frame 사이의 비교
def compare_image(video):
    frame = []
    mse_val = []
    ssim_val = []
    m = []
    s = []

    origin = cv2.imread('./%s/origin.png' % (video))
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    for j in range(0, video_cap(video)):
        if j == 0:
            img = cv2.imread('./%s/frame%s.png' % (video, j))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            m.append(mse(origin, img))
            s.append(ssim(origin, img))
        else:
            pv_img = cv2.imread('./%s/frame%s.png' % (video, j-1))
            pv_img = cv2.cvtColor(pv_img, cv2.COLOR_BGR2GRAY)

            img = cv2.imread('./%s/frame%s.png' % (video, j))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            m.append(mse(pv_img, img))
            s.append(ssim(pv_img, img))

    for j in range(0, video_cap(video)):
        # mse와 ssim 출력

        print("%s | mse : %f    ssim : %f" % (j, m[j], s[j]))
        print()
        # 임의로 정한 mse & ssim 조건
        # mse 는 클수록 오차가 심함.
        # ssim은 작을수록 오차가 심함.
        if (m[j] > 4) and (s[j] < 0.99):
            frame.append(j)
            mse_val.append(m[j])
            ssim_val.append(s[j])

    return frame, mse_val, ssim_val, video_cap(video)