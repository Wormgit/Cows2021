
# get frames from videos every 0.2s and output them to a file named frames

# structure of videoFile
#0Videos
#--2020-03-08_12-52-4
#--#--RGB.avi
#--2020-03-08_12-52-9
#--#--RGB.avi

import cv2, os, glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--videoFile', default='/home/Desktop/0Videos/', type=str, help = 'a folder contains many subfolders with RGB.avi')
args = parser.parse_args()

time_gap = 0.2
detect_exsistance = 0
video_to_frame = 1
list_of_file = os.listdir(args.videoFile)

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

if video_to_frame:
    for items in list_of_file:
        filename = args.videoFile + items + '/RGB.avi'
        outputFile = args.videoFile + items + '/frames/'
        makedirs(outputFile)

        vc = cv2.VideoCapture(filename)

        if vc.isOpened():
            rval, frame = vc.read()
        else:
            print('openerror!')
            rval = False

        c=0
        rval=vc.isOpened()
        kk = vc.get(cv2.CAP_PROP_FPS)
        gap = kk * time_gap   #视频帧计数间隔频率
        name = 0
        while rval:   #循环读取视频帧
            rval, frame = vc.read()

            milliseconds = vc.get(cv2.CAP_PROP_POS_MSEC)
            seconds = milliseconds // 1000
            milliseconds = milliseconds % 1000
            minutes = 0
            hours = 0
            if seconds >= 60:
                minutes = seconds//60
                seconds = seconds % 60
            if minutes >= 60:
                hours = minutes//60
                minutes = minutes % 60
            #print(int(hours), int(minutes), int(seconds), int(milliseconds))

            if rval:  # for the last item
                if (c % gap == 0):  # 每隔timeF帧进行存储操作
                    cv2.imwrite(outputFile + str(name).zfill(5) + '.jpg', frame)  # 存储为图像 + filename.strip('/RGB.avi') + '_'
                    name += 1
                cv2.waitKey(1)
            else:
                break
            c = c + 1
        vc.release()
    print (f'{kk} FPS ({time_gap} seconds)')


print('\nNext:Open 2\n')
print('\ndone\n')
