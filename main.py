import cv2
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt
import Comparer.main_sr as main_sr

# Yolo 로드
def yolo(img):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지 가져오기
    # img = cv2.imread("sample7.png")
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
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
                if (w*h) >= 1000 :
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # 노이즈 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    xy = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            if label == 'person':
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                xy.append([x, y, x + w, y + h])
                # print(xy[0][0])
                print(x, y, x + w, y + h)


            # print()
            # print(w, h) # 너비, 높이
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    # img = cv2.resize(img, None, fx=4, fy=4)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img, xy


IMG_TRIM = []

def crop(frame, x_start, y_start, x_end, y_end):

    # 사진 이름 넣고 테스트
    # image = cv2.imread('sample1.png')
    # image = cv2.resize(frame, None, fx=0.4, fy=0.4)
    # img_trim = image[y_start : y_end, x_start : x_end]
    img_trim = frame[y_start : y_end, x_start : x_end]
    width, height, channel = img_trim.shape
    print(width, height, channel)
    # cv2.imshow('source', frame)
    cv2.imshow('cut image', img_trim)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    IMG_TRIM.append(img_trim)


def add(image, img_trim, x_start, y_start, x_end, y_end):
    # 자른 이미지를 소스 이미지에 붙인다.
    # frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    # print(x_start, y_start, x_end, y_end)
    # cv2.imshow("zzz", img_trim)
    # h,w,c = img_trim.shape
    # print(h,w,c)
    image[y_start: y_end, x_start: x_end] = img_trim

    # cv2.imshow("zz",image[y_start : y_end, x_start : x_end])
    # frame[y_start : y_end, x_start : x_end] = img_trim

    #
    # # 결과를 출력한다.
    cv2.imshow('source', image)
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

SR_img = []
def SR(img_trim):

    # plt.imshow(img_trim[:,:,::-1])
    # plt.show()

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
    # resized = cv2.resize(img_trim, dsize=None, fx=4, fy=4)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    # Original image
    plt.imshow(img_trim[:, :, ::-1])
    plt.subplot(1, 3, 2)
    # SR upscaled
    plt.imshow(result[:, :, ::-1])
    # plt.subplot(1, 3, 3)
    # # OpenCV upscaled
    # plt.imshow(resized[:, :, ::-1])
    plt.show()
    result = cv2.resize(result, None, fx=0.25, fy=0.25)
    SR_img.append(result)

if __name__ == '__main__':
    frame,mse_val,ssim_val = main_sr.cp()
    print(frame,mse_val,ssim_val)

    for num in range(len(frame)):
        checked_frame = cv2.imread("video3/frame%s.png" %frame[num])
        image, xy = yolo(checked_frame)
        for i in range(len(xy)):
            crop(image, xy[i][0], xy[i][1], xy[i][2], xy[i][3])
            SR(IMG_TRIM[i])

    # # # print(im.shape)
    # # # print(image[1],image[2],image[3],image[4])
    # for i in range(len(xy)):
    #     add(image, SR_img[i], xy[i][0], xy[i][1], xy[i][2], xy[i][3])
