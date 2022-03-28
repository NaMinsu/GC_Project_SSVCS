import shutil
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Comparer.main_sr as main_sr
import frameToVideo


# Yolo 로드
def yolo(img, debug):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[j - 1] for j in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지 가져오기
    # img = cv2.resize(img, None, fx=0.8, fy=0.8)
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if (w*h) >= 1000:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # 노이즈 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    xy = []
    print("YOLO 좌표")
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            if label == 'person' or label == 'car':
                # 박스 그림.
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                xy.append([x, y, x + w, y + h])
                # print(xy[0][0])
                print(x, y, x + w, y + h)


            # print()
            # print(w, h) # 너비, 높이
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    # img = cv2.resize(img, None, fx=4, fy=4)
    if debug:
        plt.imshow(img[:, :, ::-1])
        plt.show()

    return img, xy


def crop(frame, x_start, y_start, x_end, y_end, IMG_TRIM):

    img_trim = frame[y_start: y_end, x_start: x_end]

    # cv2.imshow('cut image', img_trim)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    IMG_TRIM.append(img_trim)
    return IMG_TRIM


def add(checked_frame, SR_img, x_start, y_start, x_end, y_end):
    # 자른 이미지를 소스 이미지에 붙인다.
    print("add function")
    print(x_start, y_start, x_end, y_end) # yolo 박스 좌표

    # 높이, 너비 순서
    h, w, c = SR_img.shape  # 박스 크기
    print(h, w, c)
    checked_frame[y_start: y_end, x_start: x_end] = SR_img

    # cv2.imshow("zz",image[y_start : y_end, x_start : x_end])
    # frame[y_start : y_end, x_start : x_end] = img_trim

    # 결과를 출력한다.
    # cv2.imshow('frame added SR image', checked_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return 0


def SR(img_trim, SR_img, debug):

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "EDSR_x4.pb"
    sr.readModel(path)

    sr.setModel("edsr", 4)
    result = sr.upsample(img_trim)

    # cv2.imshow('cut image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('frame1.png', result)

    # # Resized image
    # resized = cv2.resize(img_trim, None, fx=4, fy=4)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    # Original image
    if debug:
        plt.imshow(img_trim[:, :, ::-1])
    plt.subplot(1, 2, 2)
    # SR upscaled
    if debug:
        plt.imshow(result[:, :, ::-1])
        plt.show()

    # result = cv2.resize(result, None, fx=0.25, fy=0.25)
    SR_img.append(result)
    return SR_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Super Resolution")
    parser.add_argument('--video_name', type=str, help='test low resolution video name')
    parser.add_argument('--mode', type=bool, default=False, help='select execute mode(False) or debug mode(True)')
    opt = parser.parse_args()

    frame, mse_val, ssim_val = main_sr.cp(opt.video_name)
    print(frame)
    # frame=[1]
    print("총 :", len(frame))
    for num in range(len(frame)):
        print("현재 : ", num)
        IMG_TRIM = []
        SR_img = []
        checked_frame = cv2.imread("%s/frame%s.png" % (opt.video_name, frame[num]))
        # checked_frame = cv2.imread("video4/frame2.png")

        image, xy = yolo(checked_frame, opt.mode)
        # print("image : ", image.shape)

        for i in range(len(xy)):
            IMG_TRIM = crop(image, xy[i][0], xy[i][1], xy[i][2], xy[i][3], IMG_TRIM)
            SR_img = SR(IMG_TRIM[i], SR_img, opt.mode)

        if opt.mode:
            plt.imshow(checked_frame[:, :, ::-1])
            plt.show()

        resized_frame = cv2.resize(checked_frame, None, fx=4, fy=4)
        for i in range(len(SR_img)):
            add(resized_frame, SR_img[i], 4*xy[i][0], 4*xy[i][1], 4*xy[i][2], 4*xy[i][3])
        if opt.mode:
            plt.imshow(resized_frame[:, :, ::-1])
            plt.show()
        cv2.imwrite('%s/frame%s.png' % (opt.video_name, frame[num]), resized_frame)

    frameToVideo.frameToVideo(opt.video_name, frame)
    shutil.rmtree('./{}/'.format(opt.video_name))
