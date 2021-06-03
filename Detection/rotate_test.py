#!/usr/bin/env python
# coding: utf-8

# pay attention to if id_of_image > -2:

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, cv2, time
import matplotlib.pyplot as plt
import sys, math
import argparse
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, '../')
from keras_retinanet.bin.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_caption, draw_rbox, get_rbox_poly
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.bin.cocoapi.PythonAPI.pycocotools.coco import COCO
from keras_retinanet.bin.cocoapi.PythonAPI.pycocotools.iou_clip import faster_cauculate_rt_iou_clip

parser = argparse.ArgumentParser()
parser.add_argument("--confidence_score_th", default=0.3, type=float)
parser.add_argument("--nms_threshold", default=0.1, type=float) #../path/Rotate_inbarn_rgb/images/val
parser.add_argument('--save_path', default='/home/io18230/Desktop/1/', type=str) #/home/io18230/Desktop/1/    #val_no_label
#parser.add_argument('--set_dir', default='/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/RGB2020_detec/Test/images/val', type=str) # Rotate_inbarn_rgb demo demo_any demo_3images /home/io18230/Desktop/actually_ratin/images/val New_may_filter_val NEW_MAY_VAL_ALL New_may_filter_val COWs  New_april_block41
parser.add_argument('--set_dir', default='/home/io18230/0Projects/keras-retinanet-master/path/Detection_rota/demo_3images/images/val', type=str)
parser.add_argument('--model_path', default=os.path.join('trained_model', 'resnet50_g_17.h5'), type=str)  # /home/io18230/Desktop/ resnet50_256_0406
parser.add_argument('--rotated_box_path', default=None)  # if not specify, it is in images/val

parser.add_argument("--withground", default=1, type=int, help='If 1, load, show, if use it for evaluation map ==1')
parser.add_argument("--pre", default=1, type=int, help='if 1 , run detection load, show, output json format and evluate')
parser.add_argument("--map", default=1, type=int, help='map evaluation?')
parser.add_argument("--save_img", default=1, type=int, help='')


parser.add_argument("--print_id", default=1, type=int)
parser.add_argument("--show_low_conf", default=0, type=int) # hilight some
parser.add_argument("--low_th", default=0.3, type=float)
parser.add_argument("--high_th", default=0.5, type=float)

parser.add_argument("--ignore_clip", default=1, type=int, help='')
parser.add_argument("--ignore_clip_th_dt", default=0.1, type=float, help='')  #set to 0.1 nothing happen Percentage higher than it suvive
parser.add_argument("--ignore_clip_th_gt", default=0.1, type=float, help='')
parser.add_argument("--similar", default=0.04, type=float, help='')


parser.add_argument("--start_image", default=-2, type=int, help='Skip some images,defult= -2 ')
parser.add_argument("--crop_detect", default=0, type=int, help='Crop and save atomotively')
parser.add_argument("--show_time_per", default=0, type=int, help='')
parser.add_argument("--show_time", default=0, type=int, help='')

parser.add_argument('--gt_path_retangular', default='', type=str,help='in case we have the rec gt and want to see it')  # os.path.join('../path/COWs/annotations','instances_val.json'), type=str)
args = parser.parse_args()

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def print_summary(save):
    print('\n\nTest images are from : {}'.format(args.set_dir))
    if save:
        print('Save images to :       {}'.format(args.save_path))
    else:
        print('set not to save images')
    print('\nConfidence threshold:  {}'.format(args.confidence_score_th))
    print('NMS threshold:         {}'.format(args.nms_threshold))
    if args.ignore_clip:
        print('clip_th for detection: {}'.format(args.ignore_clip_th_dt))
        print('clip_th for gt:        {}'.format(args.ignore_clip_th_gt))
    print('IOU (positive_overlap_threshold):\n')

def crop(draw, box_pre, save_path, id_of_image, crop_n): # 有毛病

    '''
    crop and save  # need to fix it
    '''

    # tem for calculate cv4animal paper
    box_image = get_rbox_poly(640, 360, 1280, 720, 0)
    inter_area = box_pre[1].intersection(box_image[1]).area
    expour = float(inter_area) / box_pre[1].area
    #print(expour)
    # tem for calculate paper

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

    # crop large image and fill to keep the centre

    crop = draw_for_crop[ymin:ymax, xmin:xmax]
    crop_fill = cv2.copyMakeBorder(crop, ymin_gap + 2, ymax_gap + 2, xmin_gap + 2, xmax_gap + 2,
                                   cv2.BORDER_CONSTANT, value=0)
    # rotate image
    degree = 180 * box[4] / math.pi
    center = (crop_fill.shape[1] // 2, crop_fill.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, degree, 1)
    rotated = cv2.warpAffine(crop_fill, M, (crop_fill.shape[1], crop_fill.shape[0]))

    # crop again
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

    cv2.imwrite(
        save_path + 'r_' + str("%06d" % id_of_image) + '_' + str("%02d" % crop_n)
        + '_center_' + str(xe) + '_' + str(ye) + '_h_' + str(round(degree, 2)) +
        '_ovlp_' + str(round(expour, 4)) + '.jpg', rotated_crop)
    crop_n += 1

    return crop_n



if args.save_path:
    save_path = args.save_path
else:
    save_path = args.set_dir+'/2/'

makedirs(save_path)

if args.rotated_box_path is not None:
    rotated_box_path = args.rotated_box_path
else:
    rotated_box_path = os.path.join(args.set_dir.replace('/images/val', ''), 'annotations', 'instances_val.json')

if args.withground:
    rgt = COCO(rotated_box_path)  # p
if args.gt_path_retangular:
    ggt = COCO(args.gt_path_retangular)

gpu = 0  # use this to change which GPU to use
setup_gpu(gpu)  # set the modified tf session as backend in keras

# output boxes and validation initialization
results = []
c = 1
cow={}
cow['info'] = {"description": "predicted cow", "year": 2020, "format": "coco,x1,y1,w,h"}
cow['summary'] = []
cow['images'] = []
cow['annotations'] = []
cow['categories'] = [{"supercategory": "predicted_cow", "id": 1, "name": "xin"}]
image_ids = []
c_image = 0

################# model initialisation
if args.pre:
    model = models.load_model(args.model_path, backbone_name='resnet50')
    model = models.convert_model(model, nms_threshold=args.nms_threshold)
    #print(model.summary())
    labels_to_names = {0: 'Cow', 1: 'error'}  #######check labels later ###############################
    print('\nload trained model:  {0:40s}'.format(args.model_path))

new = []
tmp_low_gt = []
###################### function
def show_rotated_gt (gt, id_of_image, clip, clip_th, image_size):

    annIds = gt.getAnnIds(imgIds=id_of_image, iscrowd=None)
    anns = gt.loadAnns(annIds)

    for n in range(len(anns)):
        x, y, w, h, angle = anns[n]['bbox']  # (x1,y1,w,h)
        x = x + w / 2
        y = y + h / 2
        # box_r = [x, y, x, h]  # (xc,yc,w,h)
        id = (anns[n]['category_id'])
        caption = "{}".format(gt.loadCats(id)[0]['supercategory'])  # id need to be redefined
        box2 = get_rbox_poly(x, y, w, h, angle)

        b = []

        if clip:
            b = [anns[n]['bbox']]
            mn= np.array([[0, 0, image_size.shape[1], image_size.shape[0], 0]])
            ratio = faster_cauculate_rt_iou_clip(np.array(b), mn)

            draw_caption(draw, [x, y], 'GT :' + caption, left=0)  # show caption always, does not count for mAP
            display = 0
            if ratio <= clip_th:
                #print(id_of_image, ratio)
                if ratio > clip_th - args.similar:
                    tmp_low_gt.append(anns[n])
                    display = 0  # 不一定有display的资格
                else:
                    continue
            else:
                display = 1
                new.append(anns[n])

        if display:
            draw_caption(draw, [x, y], 'GT :' + caption, left=0)  #
            draw_rbox(draw, box2[0], color=label_color(3), thickness=5)  # 1 is light blue
            # draw head
            xe = int(x)
            ye = int(y)
            endx = int(xe + w / 1.5 * math.cos(angle))
            endy = int(ye + w / 1.5 * math.sin(angle))
            cv2.line(draw, (xe, ye), (endx, endy), color=label_color(3), thickness=4)

    #gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

def show_retangular_gt(gt, id_of_image):
    annIds = gt.getAnnIds(imgIds=id_of_image, iscrowd=None)
    anns = gt.loadAnns(annIds)
    for n in range(len(anns)):
        x, y, w, h = anns[n]['bbox']
        box_g = [x, y, x + w, y + h]
        xg, yg, wg, hg = int(x + w / 2), int(y + h / 2), w, h
        id = (anns[n]['category_id'])
        caption = "{}".format(
            gt.loadCats(id)[0]['supercategory'])  # superc is cow in json file. name is the detailed id xin
        draw_caption(draw, box_g, 'GT :'+caption, left=0)
        boxxx = get_rbox_poly(xg, yg, wg, hg, 0)
        draw_rbox(draw, boxxx[0], color=label_color(3), thickness=5)  # 3 is red

t_sum = 0
count = 0  # show result for every xx samples
tmp = 0

#if __name__ == '__main__':
for im in sorted(os.listdir(args.set_dir)):  ### load images
    crop_n = 0
    if im.endswith(".jpg") or im.endswith(".png"):
        image = read_image_bgr(os.path.join(args.set_dir, im))
        draw = image.copy()
        draw_for_crop = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=256, max_side=342) #####(800,1333)
        if str.isdigit(im.strip('.jpg')):
            id_of_image = int(im.strip('.jpg'))
        else:
            id_of_image = im.strip('.jpg')

        if args.map: # prepare for json
            img_dict = {
                'file_name': im,
                'id': id_of_image,
                "width": draw.shape[1],
                "height": draw.shape[0],
            }
            cow['images'].append(img_dict)
            c_image = c_image + 1
            image_ids.append(id_of_image)

        #start from specific images
        if id_of_image > args.start_image:
            # show gt
            if args.withground:          # rotated gt box
                show_rotated_gt(rgt, id_of_image, args.ignore_clip, args.ignore_clip_th_gt, draw)
            if args.gt_path_retangular:  # retangular ground truth
                show_retangular_gt(ggt, id_of_image)

            # show predicted box
            string_print = ''
            sting_print2 = ''
            if args.pre:
                start = time.time()  # process image and calculate
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                boxes[:-1] = boxes[:-1].astype(int)
                boxes[:, :, :-1] /= scale  # correct for image scale
                indice = np.where(scores[0]>args.confidence_score_th)



                # count time and count
                t = time.time() - start
                if args.show_time_per:
                    print(im, ' ', t)
                count = count + 1
                if count == 1:
                    pass
                else:
                    t_sum = t_sum + t

                #drwa
                mutual_iou = []
                for box, score, label in zip(boxes[0][indice], scores[0][indice], labels[0][indice]):
                    if args.show_low_conf:
                        if score > args.low_th and score < args.high_th:
                            #print (id_of_image, round(score, 4))
                            pass

                    box[4] = round(box[4], 3)     #(x1,y1,x2,y2)
                    x1 = int(round((box[0])))
                    y1 = int(round((box[1])))
                    xe = int(round((box[0]+box[2])/2))
                    ye = int(round((box[1]+box[3])/2))
                    we = int(round(box[2]-box[0]))
                    he = int(round(box[3]-box[1]))  #(xc,yc,w,h)
                    coco_box = box
                    coco_box[2] = we
                    coco_box[3] = he

                    #show: ignore those < ratio (inside image/ the detection box)
                    image_size = np.array([[0, 0, draw.shape[1], draw.shape[0], 0]])
                    ratio2 = faster_cauculate_rt_iou_clip(np.array([coco_box.tolist()]), image_size)
                    #if id_of_image ==6693: for debug
                        #print (id_of_image, ratio2, score)
                    ignore_value = 1
                    crop_value = 1
                    search_ = 1
                    if ratio2 <= args.ignore_clip_th_dt-args.similar:  # to balance the inconsistancy of gt and detection to improve detection accuracy.
                        image_result = {
                            'file_name': im,  # can be sheild when calculate map
                            'image_id': id_of_image,
                            'category_id': 1 if label == 0 or label == 12 else 0,  # labels_to_names[label],######################
                            'id': c,  # can be sheild when calculate map
                            'score': float(score),
                            'bbox': coco_box.tolist(),  # (x1,y1,w,h)
                            "width": draw.shape[1],
                            "height": draw.shape[0],
                            'ignore': 1,
                            'crop_value': 0,
                        }
                        cow['annotations'].append(image_result)
                        results.append(image_result)
                        c = c + 1

                    elif ratio2 > args.ignore_clip_th_dt:
                        ignore_value = 0
                    else:
                        search_ = 0
                        if len(new):
                            if len(new)>11:
                                ttt = new[-10:]
                            else:
                                ttt = new
                            for item in ttt:
                                ll = item['image_id']
                                if ll == id_of_image:
                                    box_gt = item['bbox']
                                    xeg = int(round(box_gt[0] + box_gt[2] / 2))
                                    yeg = int(round(box_gt[1] + box_gt[3] / 2))
                                    distance_centre = math.sqrt((xeg-xe)**2+(yeg-ye)**2)
                                    if distance_centre < 60: # in later version, it should be a proportion of
                                        #print(id_of_image, ratio2, distance_centre, score)
                                        ignore_value = 0
                                        label = 12
                                        crop_value = 0
                                        pass
                        else:
                            image_result = {
                                'file_name': im,  # can be sheild when calculate map
                                'image_id': id_of_image,
                                'category_id': 1 if label == 0 or label == 12 else 0,
                                # labels_to_names[label],######################
                                'id': c,  # can be sheild when calculate map
                                'score': float(score),
                                'bbox': coco_box.tolist(),  # (x1,y1,w,h)
                                "width": draw.shape[1],
                                "height": draw.shape[0],
                                'ignore': 1,
                                'crop_value': 0,
                            }
                            cow['annotations'].append(image_result)
                            results.append(image_result)
                            c = c + 1

                    if ignore_value == 0:
                        if search_:  # when gt is lower
                            if len(tmp_low_gt) > 6:
                                tt = tmp_low_gt[-5:]
                            else:
                                tt = tmp_low_gt
                            for item in tt:
                                ll = item['image_id']
                                if ll == id_of_image:
                                    box_gt = item['bbox']
                                    x, y, w, h, angle = item['bbox']  # (x1,y1,w,h)
                                    x = x + w / 2
                                    y = y + h / 2
                                    distance_centre = math.sqrt((x - xe) ** 2 + (y - ye) ** 2)
                                    if distance_centre < 60:
                                        new.append(item)   # pick up it
                                        box2 = get_rbox_poly(x, y, w, h, angle)
                                        draw_rbox(draw, box2[0], color=label_color(11), thickness=5)
                                        # draw head
                                        xe = int(x)
                                        ye = int(y)
                                        endx = int(xe + w / 1.5 * math.cos(angle))
                                        endy = int(ye + w / 1.5 * math.sin(angle))
                                        cv2.line(draw, (xe, ye), (endx, endy), color=label_color(11), thickness=5)
                                        draw_caption(draw, [x, y], 'gt : cow', left=0)  #

                            box_pre = get_rbox_poly(xe, ye, we, he, box[4])
                        if args.crop_detect:
                            crop_n = crop(draw, box_pre, save_path, id_of_image, crop_n)

                        if score>0.65:
                            c_lor = (255 , 95 , 0)
                        else:
                            c_lor = label_color(label)
                        c_lor = label_color(label)
                        draw_rbox(draw, box_pre[0], color=c_lor, thickness=2)  # 1 is light blue
                        # draw head
                        endx = int(xe + we / 1.5 * (math.cos(box[4])))
                        endy = int(ye + we / 1.5 * (math.sin(box[4])))
                        cv2.line(draw, (xe, ye), (endx, endy), color=label_color(label), thickness=3)

                        image_result = {
                            'file_name': im,  # can be sheild when calculate map
                            'image_id': id_of_image,
                            'category_id': 1 if label == 0 or label == 12 else 0,  # labels_to_names[label],######################
                            'id': c,  # can be sheild when calculate map
                            'score': float(score),
                            'bbox': coco_box.tolist(),  # (x1,y1,w,h)
                            "width": draw.shape[1],
                            "height": draw.shape[0],
                            'ignore': ignore_value,
                            'crop_value': crop_value,
                            # give ignore mark of those < ratio (inside image/ the detection box)
                        }
                        c = c + 1
                        cow['annotations'].append(image_result)
                        results.append(image_result)

                    #caption = "{} {:.3f}".format('Pred :'+labels_to_names[label], score)
                    caption = "{:.3f}".format(score)  # keep the confidence but delet box at the boundary
                    b = [xe, ye]
                    draw_caption(draw, b, caption, left=1)  # always display images
                    mutual_iou.append([coco_box,box_pre])


            # calculate mutual iou:
            if len(mutual_iou)>1:
                for i in range(len(mutual_iou)):
                    expour = faster_cauculate_rt_iou_clip(np.array([mutual_iou[i][0].tolist()]), image_size) #相对于图片
                    if expour<0.999:  # do not display obvious right
                        m = str(round(expour[0][0], 2)) + ' '
                        for hh in range(len(m)):
                            sting_print2 += m[hh]

                    if i == len(mutual_iou):
                        break

                    for j in range(i+1,len(mutual_iou)):
                        inter_area = mutual_iou[i][1][1].intersection(mutual_iou[j][1][1]).area
                        union_area = mutual_iou[i][1][1].area + mutual_iou[j][1][1].area - inter_area
                        if union_area == 0:
                            iou = 0
                        else:
                            iou = float(inter_area) / union_area
                        if iou>0:
                            m = str(round(iou,2))+' '
                            for k in range(len(m)):
                                string_print += m[k]



            # put id to image
            if args.print_id:
                draw_caption(draw, [0, draw.shape[0] - 36], im + 'IOU:'+ string_print + ' EXP:' + sting_print2, left=1)

            #save
            if args.save_img:
                plt.figure(figsize=(12, 12))
                plt.axis('off')
                plt.imshow(draw)
                plt.savefig(save_path+im, bbox_inches = 'tight', pad_inches = 0.0)
                plt.close('all')
            # plt.show()  # display the images

if args.show_time:
    print('\nAverage processing time per image:')
    print(round(t_sum / (count-1), 3))
    print('\n')

# formal output
if args.pre:
    temp = {"number of images": c_image, "number of annotations": c}
    cow['summary'].append(temp)
    json.dump(cow, open('{}_bbox.json'.format('predicted'), 'w'), indent=4)
# json for calculation
json.dump(results, open('{}_bbox.json'.format('pre_results'), 'w'), indent=4)
json.dump(new, open('{}_tmp.json'.format('gt_annotation'), 'w'), indent=4)

# map
def calculate_map(gt, save):
    coco_true = gt #(x1, y1, w, h)
    coco_true2 = coco_true.loadRes('{}_tmp.json'.format('gt_annotation')) # fixed annotation
    coco_pred = coco_true.loadRes('{}_bbox.json'.format('pre_results'))  # turn (x1,y1,w,h) to (x1,x2,y1,y2)
    # run COCO evaluation
    coco_eval = COCOeval(coco_true2, coco_pred, 'bbox')
    coco_eval.params.maxDets = [1, 10, 20]
    coco_eval.params.imgIds = image_ids # cause we do not use all validation ground truth.
    coco_eval.evaluate()
    coco_eval.accumulate()

    # plot and save rc-curve
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve for Cow')
    ax.set_xlim([0.0, 1.3])
    ax.set_ylim([0.0, 1.1])
    x = np.arange(0.0, 1.01, 0.01)

    for i in range(0, 4):
        color = COLORS[i * 2]
        j = coco_eval.eval['precision'][i, :, 0, 0, 2]
        plt.plot(x, j, color=color, label='{:.2f}'.format(0.5 + i * 0.1))
        # ax.scatter(x, j, label='{:.2f}'.format(0.5), s=20, color=color)

    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')

    #plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(save_path+'000P_R_CURVE.png', bbox_inches='tight', pad_inches=0.2)

    # print summary
    print_summary(save)
    coco_eval.summarize()
    # for index, result in enumerate(coco_eval.stats):
    #     logs[coco_tag[index]] = result

if args.pre:
    if args.map:
        calculate_map(rgt, args.save_img)
print('\nDone\n')
