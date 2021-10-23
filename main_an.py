# import tracker
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import numpy as np
import cv2 as cv
from PIL import ImageTk
from PIL import Image
import threading


class videoFile:
    def __init__(self, name):
        self.name = name


def select_video():
    global panel
    root.file = filedialog.askopenfile(
        initialdir='path',
        title='select video file',
        filetypes=(('avi files', '*.avi'),
                   ('mp4 files', '*.mp4'),
                   ('all files', '*.*'))
    )
    if '.avi' in root.file.name or '.mp4' in root.file.name:
        targetV.name = root.file.name

        t_cap = cv.VideoCapture(targetV.name)
        t_ret, t_frame = t_cap.read()
        thumbnail = ImageTk.PhotoImage(Image.fromarray(cv.cvtColor(t_frame, cv.COLOR_BGR2RGB)))
        if panel is None:
            panel = Label(image=thumbnail)
            panel.image = thumbnail
            panel.pack()
        else:
            panel.configure(image=thumbnail)
            panel.image = thumbnail
    else:
        messagebox.showinfo(title="Selection Fail", message="This is invalid file format. Please select avi or mp4"
                                                            "file.")


def tracking_video(vfile):
    global panel

    if vfile == "":
        messagebox.showinfo(title="Video Load Error", message="Please select video before tracking.")
        return

    cap = cv.VideoCapture(vfile)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(30, 30),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    (x, y, w, h) = cv.selectROI('Select Window', old_frame, fromCenter=False, showCrosshair=True)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    point_list = []
    for _y in range(y + int(0.4 * h), y + int(0.6 * h), 15):
        for _x in range(x + int(0.4 * w), x + int(0.6 * w), 15):
            point_list.append((_x, _y))
    points = np.array(point_list)
    points = np.float32(points[:, np.newaxis, :])
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    vector = np.array([0, 0])
    cv.destroyWindow('Select Window')
    while cap.isOpened():
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = points[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            a = int(a)
            b = int(b)
            c, d = old.ravel()
            c = int(c)
            d = int(d)
            vector = vector + np.array([a - c, b - d])
            vector[0] = int(vector[0] / 2)
            vector[1] = int(vector[1] / 2)
            mask = cv.line(mask, (vector[0] + c, vector[1] + d), (c, d), color[i].tolist(), 2)
            frame = cv.circle(frame, (vector[0] + c, vector[1] + d), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)

        if panel is None:
            panel = Label(image=image_tk)
            panel.image = image_tk
            panel.pack()
        else:
            panel.configure(image=image_tk)
            panel.image = image_tk

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        points = good_new.reshape(-1, 1, 2)
        if cv.waitKey(30) == 27:
            return


def videoThreadStart():
    thread_video = threading.Thread(target=tracking_video, args=[targetV.name])
    thread_video.daemon = True
    thread_video.start()


targetV = videoFile("")

panel = None

root = Tk()
btn_frame = Frame(root)
btn1 = Button(btn_frame, text="Select Video", command=select_video)
btn2 = Button(btn_frame, text="Tracking", command=videoThreadStart)


if __name__ == "__main__":
    root.title("Object Tracker")
    root.geometry("1600x900+100+100")
    root.resizable(False, False)

    btn_frame.pack(side=tkinter.BOTTOM)
    btn1.grid(row=0, column=0)
    btn2.grid(row=0, column=1)

    root.mainloop()
