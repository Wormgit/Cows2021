'''
Output detection images with BB and a json file for the box information
'''


import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, cv2
import matplotlib.pyplot as plt
import math
import argparse
import json

from keras_retinanet.utils.visualization import draw_caption, draw_rbox, get_rbox_poly
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.bin.cocoapi.PythonAPI.pycocotools.iou_clip import faster_cauculate_rt_iou_clip

gpu = 0  # use this to change which GPU to use
setup_gpu(gpu)  # set the modified tf session as backend in keras

parser = argparse.ArgumentParser()
parser.add_argument("--confidence_score_th", default=0.3, type=float)
parser.add_argument("--nms_threshold", default=0.28, type=float)
parser.add_argument('--frame_file', default='/home/io18230/Desktop/0Videos/', type=str)
parser.add_argument('--model_path', default=os.path.join('trained_model', 'resnet50_trained_144.h5'), type=str)  # /home/io18230/Desktop/ resnet50_256_0406
parser.add_argument("--crop_detect", default=1, type=int, help='crop and save') # save cropped images

parser.add_argument("--ignore_clip_th_dt", default=0.945, type=float, help='')  #set to 0.1 nothing happen Percentage higher than it suvive
parser.add_argument("--similar", default=0.04, type=float, help='')
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

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

model = models.load_model(args.model_path, backbone_name='resnet50')
model = models.convert_model(model, nms_threshold=args.nms_threshold)

for items in os.listdir(args.frame_file):
    print(items)
    File = args.frame_file + items + '/frames'
    save_path = args.frame_file + items + '/results/'
    makedirs(save_path)

    # output boxes and validation initialization
    results = []
    c = 1
    cow={}
    cow['info'] = {"description": "predicted cow", "year": 2021, "format": "coco, x1, y1, w, h", "BELONGS TO": items}
    cow['summary'] = []
    cow['images'] = []
    cow['annotations'] = []
    cow['categories'] = [{"supercategory": "predicted_cow", "id": 1, "name": "xin"}]
    image_ids = []
    c_image = 0

    new = []
    tmp_low_gt = [] # show result for every xx samples
    tmp = 0
    count_exit = 0  # 1 or 2

    for im in sorted(os.listdir(File)):  ### load images
        crop_n = 0
        if im.endswith(".jpg") or im.endswith(".png"):
            image = read_image_bgr(os.path.join(File, im))
            draw = image.copy()
            draw_for_crop = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image)
            image, scale = resize_image(image, min_side=256, max_side=342)  #####(800,1333)
            if str.isdigit(im.strip('.jpg')):
                id_of_image = int(im.strip('.jpg'))
            else:
                id_of_image = im.strip('.jpg')

            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes[:-1] = boxes[:-1].astype(int)
            boxes[:, :, :-1] /= scale  # correct for image scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < args.confidence_score_th:  # 0.5,0.7
                    break

                box[4] = round(box[4], 3)  # (x1,y1,x2,y2)
                x1 = int(round((box[0])))
                y1 = int(round((box[1])))
                xe = int(round((box[0] + box[2]) / 2))
                ye = int(round((box[1] + box[3]) / 2))
                we = int(round(box[2] - box[0]))
                he = int(round(box[3] - box[1]))  # (xc,yc,w,h)
                coco_box = box
                coco_box[2] = we
                coco_box[3] = he

                # show: ignore those < ratio (inside image/ the detection box)
                image_size = np.array([[0, 0, draw.shape[1], draw.shape[0], 0]])
                ratio2 = faster_cauculate_rt_iou_clip(np.array([coco_box.tolist()]), image_size)

                ignore_value = 1
                crop_value = 1
                search_ = 1
                # ignore poor cropped images
                if ratio2 <= args.ignore_clip_th_dt - args.similar:  # to balance the inconsistancy of gt and detection to improve detection accuracy.
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

                elif ratio2 > args.ignore_clip_th_dt:
                    ignore_value = 0
                else:
                    search_ = 0
                    if len(new):
                        if len(new) > 11:
                            ttt = new[-10:]
                        else:
                            ttt = new
                        for item in ttt:
                            ll = item['image_id']
                            if ll == id_of_image:
                                box_gt = item['bbox']
                                xeg = int(round(box_gt[0] + box_gt[2] / 2))
                                yeg = int(round(box_gt[1] + box_gt[3] / 2))
                                distance_centre = math.sqrt((xeg - xe) ** 2 + (yeg - ye) ** 2)
                                if distance_centre < 60:  # in later version, it should be a proportion of
                                    # print(id_of_image, ratio2, distance_centre, score)
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
                                    new.append(item)  # pick up it
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


                    draw_rbox(draw, box_pre[0], color=label_color(label), thickness=2)  # 1 is light blue
                    # draw head
                    endx = int(xe + we / 1.5 * (math.cos(box[4])))
                    endy = int(ye + we / 1.5 * (math.sin(box[4])))
                    cv2.line(draw, (xe, ye), (endx, endy), color=label_color(label), thickness=3)

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
                        'ignore': ignore_value,
                        'crop_value': crop_value,
                        # give ignore mark of those < ratio (inside image/ the detection box)
                    }
                    c = c + 1
                    cow['annotations'].append(image_result)
                    results.append(image_result)

                # caption = "{} {:.3f}".format('Pred :'+labels_to_names[label], score)
                caption = "{:.3f}".format(score)  # keep the confidence but delet box at the boundary
                b = [xe, ye]
                draw_caption(draw, b, caption, left=1)  # always display images

                # save
                plt.figure(figsize=(12, 12))
                plt.axis('off')
                plt.imshow(draw)
                plt.savefig(save_path + im, bbox_inches='tight', pad_inches=0.0)
                plt.close('all')


    save_label = args.frame_file + items + '/'
    temp = {"number of images": c_image, "number of annotations": c}
    cow['summary'].append(temp)
    json.dump(cow, open(save_label + '/{}_bbox{}.json'.format('predicted', items), 'w'), indent=4)

print('\ndone\n')
print('\nNext:Open 3crop and crop images\n')
