
# SSVCS
SSVCS(Smart Surveillance Video Control System) is a project produced for the purpose of efficiently utilizing storage space while improving the quality of images filmed with low-resolution CCTV. In order to properly check CCTV images, the image quality must be high, but if the image quality is high, there is a problem of using a lot of storage space. This project was designed to solve the problem.

Presentation URL: [https://www.youtube.com/watch?v=GY4ud5umE74](https://www.youtube.com/watch?v=GY4ud5umE74)

## Requirement

 - [Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
	 Install Anaconda from the site above.
 - PyTorch
	 Install pytorch and torchvision library using anaconda command.
	> conda install pytorch torchvison -c pytorch
 - OpenCV
	 Install OpenCV library using anaconda command.
 	> conda install opencv
 - Pillow
	 Install Pillow library using anaconda command.
	 >conda install -c anaconda pillow
 - [Yolo & EDSR](https://drive.google.com/drive/folders/1Uk7tVIFd0882iClDJ7mSbh-SFMEtIGUt?usp=sharing)
	 Install Yolov4 and EDSR model to downloading files from google drive above.

## Features
This project consists of two key functions:

 **1. Control video resolution**
	To optimize video storage capacity and video resolution, we suggest 'Selective Partial Super-resolution'. This technique proceeds in the following steps:
	

 1. Image Comparison
	 We assume that fixed cameras shoot the same scene most of the time. SSVCS set the scene as Idle Image, and check on a frame-by-frame basis whether a change has occurred on the screen, such as a new object appearing.
 2. Object Detection
	 When a change has occurred, find the new objects' position to improve the image quality of the location only through Yolo library.
 3. Super Resolution
	 When the locations of the objects are checked, only the image of the corresponding part is improved and stored using the deep learning model (EDSR).

 **2. Object Tracking**
	 One of the methods of using stored CCTV images is to track the path of people or objects. Accordingly, the SSVCS provides a GUI program for tracking a desired object in a stored image.

## Usage

 - **Super Resolution**
	

> python main.py --video_name
> 
> *necessary parameter*
> video_name: the file name of video file without filename extension

 - **Object Tracking**
	

> python main_an.py

Then, you can see the screen below:
![Object Trakcing Program](https://github.com/NaMinsu/GC_Project_SSVCS/blob/master/dataset/Object_tracking.PNG)
There is a screen on which the video is played and a timer bar, and the user can perform the following three functions through a button.

 1. Select Video: selecting video
 2. Tracking: Tracking the object in this video
 3. Stop Video: Stop playing video

## License
This project is releaseed under Gachon University but all source code is free.
