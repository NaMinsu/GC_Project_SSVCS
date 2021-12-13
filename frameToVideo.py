import numpy as np
import cv2 as cv
import os
import natsort


def frameToVideo(fileName, checked_frame):
    # frame = cv.imread("video3/origin.png")
    # h, w, c = frame.shape
    # video_name = './test.avi'
    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    # # 너비, 높이 순서
    # out = cv.VideoWriter(video_name, fourcc, 1.0, (w, h))
    #
    # # original 먼저 저장
    # out.write(frame)
    # cnt = 0
    # frame_cnt = 0
    # flag = 1
    # for i in os.listdir('./video3/'):
    #     if i == "origin.png":
    #         break
    #     if flag == 1:
    #         if cnt == checked_frame[frame_cnt]:
    #             path = './video3/'+"SR_frame"+str(checked_frame[frame_cnt])+".png"
    #             frame = cv.imread(path)
    #             out.write(frame)
    #             cnt += 1
    #             frame_cnt += 1
    #             if frame_cnt == len(checked_frame):
    #                 flag = 0
    #
    #             continue
    #     path = './video3/'+i
    #     frame = cv.imread(path)
    #     # h, w, c = frame.shape
    #     out.write(frame)
    #
    #     cnt += 1
    video_name = './test.avi'
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    fps = cv.VideoCapture('./Comparer/video/%s.mp4' % fileName).get(cv.CAP_PROP_FPS)
    frame = cv.imread("./{}/origin.png".format(fileName))
    frame = cv.resize(frame, None, fx=4, fy=4)

    h, w, c = frame.shape
    out = cv.VideoWriter(video_name, fourcc, fps, (w, h))

    out.write(frame)
    frame_list = list(os.listdir('./{}/'.format(fileName)))
    frame_list.remove('origin.png')
    frame_list = natsort.natsorted(frame_list)

    cnt = 0
    frame_cnt = 0
    flag = 1
    for cut in frame_list:
        path = './{}/'.format(fileName) + cut
        frame_image = cv.imread(path)
        if flag == 1:
            if cnt == checked_frame[frame_cnt]:
                out.write(frame_image)
                cnt += 1
                frame_cnt += 1
                if frame_cnt == len(checked_frame):
                    flag = 0
                continue

        frame_image = cv.resize(frame_image, None, fx=4, fy=4)
        out.write(frame_image)
        cnt += 1

    print('end')
    out.release()
    cv.waitKey(0)

    cv.destroyAllWindows()
