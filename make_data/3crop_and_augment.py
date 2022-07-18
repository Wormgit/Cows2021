#!/usr/bin/env python
# coding: utf-8
# crop based on histogram
import pandas as pd
import progressbar
import numpy as np
assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import os, cv2
import sys, math
import argparse
import random
from copy import deepcopy
import shutil
sys.path.insert(0, '../')

from keras_retinanet.utils.image import read_image_bgr, preprocess_image
from keras_retinanet.utils.visualization import get_rbox_poly
from keras_retinanet.utils.visualization import draw_caption
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.bin.cocoapi.PythonAPI.pycocotools.coco import COCO
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt

'''
Crop, remove black files, move images for making datasets
'''

parser = argparse.ArgumentParser()
parser.add_argument("--angle_augment", default=1, type=int)
parser.add_argument('--centre', default=200, type=int, help ='distance between centers') #145
parser.add_argument('--frame_file', default='/home/io18230/Desktop/0Videos/', type=str)#/home/io18230/0Projects/keras-retinanet-master/path/ID/video/
parser.add_argument('--track', default='/home/io18230/Desktop/Track/',type=str)

# address to move the files
parser.add_argument('--save_path', default='/home/io18230/Desktop/Crop', type=str)
parser.add_argument('--save_csv_path', default='/home/io18230/Desktop/Crop_csv_pari', type=str)
args = parser.parse_args()

track_path =args.track
th_centre = args.centre
th_angle = 0.16
w, number_of_id = 1, 30    #30 cow in a video clip

gpu = 0  # use this to change which GPU to use
setup_gpu(gpu)

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def del_dirs(path):  # another format is in dncnn
    if os.path.isdir(path):
        shutil.rmtree(path)

makedirs(track_path)

def last_image(per_image_ann): # initialise the first image
    for n in range(len(per_image_ann)):
        x, y, w, h, angle = per_image_ann[n]['bbox']
        xe = int(x + w / 2)
        ye = int(y + h / 2)
        conti_flag[n][0] = [xe, ye, round(angle, 2)]
    return conti_flag


def compute_centre(ig, per_image_ann, list_last):
    skip=[]
    for n in range(len(per_image_ann)): # now image annotation
        x, y, w, h, angle = per_image_ann[n]['bbox']  # (x1,y1,w,h)
        xe = x + w / 2
        ye = y + h / 2

        # if 1, the later image centre has an corresponding one in the last image.
        last = 200
        match = 0
        flag = 0

        for nu in range(len(list_last)):  # last annotation
            if nu in skip:
                #if len (skip)>1:
                    #print(0)
                continue
            ite = list_last[nu][-1]
            xel = ite[0];  yel = ite[1];   a = ite[2]
            distance_centre = math.sqrt((xel - xe) ** 2 + (yel - ye) ** 2)
            #distance_angle = abs(a - angle) # consider 3.14  or oppsite detection
            #if ig == 13:# and #821 < xe:
                #if distance_centre< 140 and distance_centre>130:
                #print(distance_centre, nu)
            if distance_centre < th_centre: #distance_angle < th_angle:
                if len (ite) == 4:
                    if ig - ite[3] <= 3:
                        if distance_centre < 90:
                            conti_flag[nu].append([int(xe), int(ye), round(angle, 2), ig])
                            match = 1
                            flag = 0
                            skip.append(nu)
                            break
                        else:
                            if min(last,distance_centre) == distance_centre:   # update minimum distance point
                                nu_nunber = nu
                                xec = deepcopy (xe); yec = deepcopy(ye)
                                last = distance_centre
                                flag = 1
                else: # no cow before
                    if distance_centre < 90:
                        conti_flag[nu].append([int(xe), int(ye), round(angle, 2), ig])
                        match = 1
                        flag = 0
                        skip.append(nu)
                        break
                    else:
                        if min(last, distance_centre) == distance_centre:     # update minimum distance point
                            nu_nunber = deepcopy(nu)
                            xec = deepcopy(xe);      yec = deepcopy(ye)
                            last = distance_centre
                            flag = 1

        if flag:
            conti_flag[nu_nunber].append([int(xec), int(yec), round(angle, 2), ig])
            match = 1
            last = 200
            flag = 0
            skip.append(nu_nunber)
            continue

        if match == 0: # if did not match any
            for ind in range(0, number_of_id):
                t = conti_flag[ind][0]
                if t[2] == 200 and len (conti_flag[ind]) == 1: # found a new one
                    conti_flag[ind][0] = [int(xe), int(ye), round(angle, 2),ig]
                    break

def Histogram(image,plot=0):
    histogram_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counts, _ = np.histogram(histogram_gray, bins=256)
    su = sum(counts) - counts[0] - counts[1]
    black = np.sum(counts[10:150]) / su
    white = np.sum(counts[240:]) / su
    ratio = black/white
    if white == 0:
        white = 1
    if plot:
        plt.figure()
        plt.title("ratio{},black{},white{} Histogram Grayscale".format(round(black / white, 3), round(black, 3),
                                                                       round(white, 3)))
        plt.xlabel("Bins")
        plt.ylabel("% of Pixels")

        hist = cv2.calcHist([histogram_gray], [0], None, [256], [0, 256]) #total
        hist = hist / np.sum(hist) #percentage
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    if ratio > 200:
        return 'T' # toally black
    elif ratio > 15:
        return 'B'
    elif ratio > 6.7 -0.3 and black>0.85:
        return 'B' # most black pattern
    elif ratio <= 6.7 + 0.5 and white>0.1:
        return 'W' # obvious white pattern
    else:
        return 'D' # don't know-between B and W

def crop(gt, id_of_image, draw):
    annIds = gt.getAnnIds(imgIds=id_of_image, iscrowd=None)
    anns = gt.loadAnns(annIds)
    crop_n = 0
    max_angle = 0.12  # 7 degree max
    for n in range(len(anns)):
        if anns[n]['crop_value']:
            x, y, ww, hh, angle = anns[n]['bbox']  # (x1,y1,w,h)
            xe = x + ww / 2
            ye = y + hh / 2
            we = ww
            he = hh
            box_pre = get_rbox_poly(xe, ye, we, he, angle)

            h, w = draw.shape[:2]
            mm = box_pre[0]
            xmin = int(round(min(mm[0], mm[2], mm[4], mm[6])))
            xmax = int(round(max(mm[0], mm[2], mm[4], mm[6])))
            ymin = int(round(min(mm[1], mm[3], mm[5], mm[7])))
            ymax = int(round(max(mm[1], mm[3], mm[5], mm[7])))

            xmin_gap = 0
            ymin_gap = 0
            xmax_gap = 0
            ymax_gap = 0

            if xmin < 0:
                tmp = xmin
                xmin = 0
                xmin_gap = abs(xmin - tmp)
            if ymin < 0:
                tmp = ymin
                ymin = 0
                ymin_gap = abs(ymin - tmp)
            if xmax > w:
                tmp = xmax
                xmax = w
                xmax_gap = abs(xmax - tmp)
            if ymax > h:
                tmp = ymax
                ymax = h
                ymax_gap = abs(ymax - tmp)

            # keep the centre, crop large square image
            crop = draw_for_crop[ymin:ymax, xmin:xmax]
            crop_fill = cv2.copyMakeBorder(crop, ymin_gap + 2, ymax_gap + 2, xmin_gap + 2, xmax_gap + 2,
                                           cv2.BORDER_CONSTANT, value=0)

            if crop_fill.shape[0] > crop_fill.shape[1]: # if h>w, padd it to a square
                fillw = int((crop_fill.shape[0] - crop_fill.shape[1])/2)
                fillh = 0
                crop_fill = cv2.copyMakeBorder(crop_fill, fillh,fillh,fillw,fillw, cv2.BORDER_CONSTANT, value=0) #int top, int bottom, int left, int right,
            #cv2.imwrite( '/home/io18230/Desktop/deskth1'  + str("%02d" % crop_n)  + '.jpg', crop_fill)

            # rotate image
            angle_line = []
            angle_line.append(angle)
            if args.angle_augment:
                if abs(angle) < 3.15 - max_angle:
                    angle_line.append(angle + random.uniform(max_angle - 0.05, max_angle))
                    angle_line.append(angle - random.uniform(max_angle - 0.05, max_angle))
                    angle_line.append(angle + random.uniform(0.02, max_angle))
                    angle_line.append(angle - random.uniform(0.02, max_angle))
                else:
                    angle_line.append(angle)
                    angle_line.append(angle)
                    angle_line.append(angle)
                    angle_line.append(angle)

            for index in range(len(angle_line)):
                degree = 180 * angle_line[index] / math.pi
                center = (crop_fill.shape[1] // 2, crop_fill.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, degree, 1)
                rotated = cv2.warpAffine(crop_fill, M, (crop_fill.shape[1], crop_fill.shape[0]))

                # crop rotated image
                fymin = int(round(rotated.shape[0] // 2 - he / 2)) - 1
                fxmin = int(round(rotated.shape[1] // 2 - we / 2)) - 1

                if fymin < 0:
                    fymin = 0
                if fxmin < 0:
                    fxmin = 0

                fxmax = fxmin + int(we) + 1
                fymax = fymin + int(he) + 1
                if fymax > draw.shape[0]:
                    fymax = draw.shape[0]
                if fxmax > draw.shape[1]:
                    fxmax = draw.shape[1]
                rotated_crop = rotated[fymin:fymax, fxmin: fxmax]
                mmm=(fxmax - fxmin) / (fymax - fymin)
                if mmm < 1.83:   # 275 150
                    print(str("%06d" % id_of_image) + '_' + str("%02d" % crop_n) + '_' + str(index)
                    + Histogram(rotated)+'ct_' + str("%04d" % xe) + '_' + str("%04d" % ye) + '_' + str(round(degree, 2)) + '.jpg')

                cv2.imwrite(
                    save_path +'r_' + str("%06d" % id_of_image) + '_' + str("%02d" % crop_n) + '_' + str(index)
                    + Histogram(rotated_crop)+'ct_' + str("%04d" % xe) + '_' + str("%04d" % ye) + '_' + str(round(degree, 2)) + '.jpg',
                    rotated_crop)
            crop_n += 1
        else:
            pass
            #print('Ignore a cattle in image {} '.format(id_of_image)) #as the proportion of exposed area is small
    return (anns)


def draw_track(conti_flag):
    i = 0
    while i < len (conti_flag):
        ooo = len(conti_flag[i])
        if ooo <= 1:
            del conti_flag[i]
        else:
            i += 1


    for im in sorted(os.listdir(File)):  ### load the first images
        if im.endswith(".jpg") or im.endswith(".png"):
            image = read_image_bgr(os.path.join(File, im))
            draw = image.copy()
            break

    color = 0
    for i in conti_flag:
        if len(i) == 1:
            i[0][-1] = -1
        for j in range(0,len(i)):
            xl = i[j][0]
            yl = i[j][1]
            if j < len(i)-1:
                x = i[j + 1][0]
                y = i[j + 1][1]
                cv2.line(draw, (xl, yl), (x, y), color=label_color(color + 11), thickness=5)
            if j==0:
                if len(i[j])==3:
                    i[j].append(-1)
            id_of_image = i[j][3]
            annIds = rgt.getAnnIds(imgIds=id_of_image, iscrowd=None)
            anns = rgt.loadAnns(annIds)

            find = 0
            for n in range(len(anns)):
                if anns[n]['crop_value'] and not anns[n]['ignore']:
                    x1, y1, ww, hh, angle = anns[n]['bbox']  # (x1,y1,w,h)
                    xe = x1 + ww / 2
                    ye = y1 + hh / 2
                    if abs (xe-xl)<=2 and abs (ye-yl)<=3:
                        cv2.line(draw, (xl, yl), (xl+1, yl+1), color=label_color(color + 15), thickness=20)
                        find =1
                        break
            if not find:
                i[j][-1] = -1
        color += 1

    draw_caption(draw, [0, draw.shape[0] - 36], 'Point: Valid individual', left=1)
    draw_caption(draw, [0, draw.shape[0] - 72], 'Line : Individual Track', left=1)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(draw)
    plt.savefig(track_path + items, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')
    for i in conti_flag: # delete blanks from csv format and track points with out valid box
        for j in range(0,len(i)):
            if i[j][-1] == -1:
                i[j] = ''
    return conti_flag


for items in os.listdir(args.frame_file):
    File = args.frame_file + items + '/frames'
    save_path = args.frame_file + items + '/crop_images/'
    csv_path =  args.frame_file + items +'/'
    json_path = args.frame_file + items +'/predicted_bbox'+items+'.json'
    del_dirs(save_path)
    makedirs(save_path)
    makedirs(csv_path)
    rgt = COCO(json_path)  # p

    conti_flag = [[[0, 0, 200] for x in range(w)] for y in range(number_of_id)] # 200 is an impossible angle to occupy position
    compare_ = 0  # to skip the first image

    for im in sorted(os.listdir(File)):  ### load images
        if im.endswith(".jpg") or im.endswith(".png"):
            image = read_image_bgr(os.path.join(File, im))
            draw_for_crop = image.copy()
            image = preprocess_image(image)
            if str.isdigit(im.strip('.jpg')):
                id_of_image = int(im.strip('.jpg'))
            else:
                id_of_image = im.strip('.jpg')

            now = crop(rgt, id_of_image, draw_for_crop)  # crop and return loaded annotations
            if not compare_:
                last_image(now)
            compute_centre(id_of_image, now, conti_flag)
            compare_ = 1

    rrrr = draw_track(conti_flag)
    data1 = pd.DataFrame(rrrr)
    data1.to_csv(csv_path+'same.csv')
    #print('Save relations to :    {}'.format(csv_path+'items.csv'))

print('\ndone\n')
print('\nTest images are from : {}'.format(args.frame_file))




# removed blank folders
destination = '/home/io18230/Desktop/remove/'
makedirs(destination)
for items in os.listdir(args.frame_file):
    File = args.frame_file + items + '/crop_images'
    m = len([lists for lists in os.listdir(File)])
    if not m:
        makedirs(os.path.join(destination,items))
        shutil.move(os.path.join(args.frame_file,items),destination)
print('\nremoved blank folders\n')


# copy cropped images to a dataset
del_dirs(args.save_path)
del_dirs(args.save_csv_path)
makedirs(args.save_path)
makedirs(args.save_csv_path)

for items in os.listdir(args.frame_file):
    sorce = args.frame_file + '/' + items + '/crop_images'
    target = args.save_path + '/' + items
    shutil.copytree(sorce, target)

    sorce_csv = args.frame_file + '/' + items + '/same.csv'
    target_csv = args.save_csv_path +'/csv/' + items
    makedirs(target_csv)
    shutil.copy(sorce_csv, target_csv+'/same.csv')
print('\ncopied cropped images to a dataset\n')
print('Next:Open 4makeVideo4Train and make training folders')
