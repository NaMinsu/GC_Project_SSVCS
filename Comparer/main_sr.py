from. import comparer
# from. import detector
# from. import resolutioner


# TODO: make main module for video super-resolution process
# if __name__ == '__main__':
def cp():
    # video 폴더에 저장한 video의 이름을 입력
    # ex) video1.mp4 ==> video1 입력
    video = input ('enter name of video: ')
    # comparer.cap_origin(video)
    comparer.video_cap(video)
    frame,mse_val,ssim_val = comparer.compare_image(video)
    for i in range(0,len(frame)):
        print('frame 번호 : %s'%frame[i])
        print('mse 값: %s'%mse_val[i])
        print('ssim 값 :%s'%ssim_val[i])

    return frame,mse_val,ssim_val
        
    