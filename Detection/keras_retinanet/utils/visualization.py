import cv2
import numpy as np
from .colors import label_color
import math
from shapely.geometry import Polygon

def get_rbox_poly(x, y, w, h, angle):    #(xc,yc,w,h)
    x0 = x
    y0 = y
    l = math.sqrt(pow(w/2, 2) + pow(h/2, 2))  # 即对角线的一半 n 次 mi
    # defult clockwise. angle is related to pi, where pi is 3.14 rather than 180.but tool outputs another direction
    a1 = angle + math.atan(h / float(w))
    a2 = angle - math.atan(h / float(w))

    pt1 = (x0 - l * math.cos(a1), y0 - l * math.sin(a1))
    pt2 = (x0 + l * math.cos(a2), y0 + l * math.sin(a2))
    pt3 = (x0 + l * math.cos(a1), y0 + l * math.sin(a1))
    pt4 = (x0 - l * math.cos(a2), y0 - l * math.sin(a2))
    line = [pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]
    line = np.array(line).reshape(4, 2)
    line = np.array(line).astype(int)
    poly = Polygon(line).convex_hull
    xx = list(poly.exterior.coords)
    B = np.array(xx).reshape(1, 10)

    return (B[0],poly)     # B[0]: 10 vertex   B[1]:poly for area calculation

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.
    # Arguments
        box       : A list of 4 elements (x1, y1, x2, y2).
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.
    # Arguments
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)

def draw_rbox(image, box, color, thickness=2):
    b = np.array(box).astype(int)
    cv2.line(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
    cv2.line(image, (b[2], b[3]), (b[4], b[5]), color, thickness, cv2.LINE_AA)
    cv2.line(image, (b[4], b[5]), (b[6], b[7]), color, thickness, cv2.LINE_AA)
    cv2.line(image, (b[6], b[7]), (b[0], b[1]), color, thickness, cv2.LINE_AA)

def draw_rboxes(image, boxes, color, thickness=2):
    for b in boxes:
        draw_rbox(image, b, color, thickness=thickness)

def draw_caption(image, box, caption, left=1):
    """caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    # if left:
    # 	kk = b[0]
    # else:
    #     kk = b[2] - 30
    # #cv2.putText(image, caption, (kk, b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    # cv2.putText(image, caption, (kk, b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    ((txt_w, txt_h), _) = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)
    if left:  # pr
        x1 = b[0]
        y1 = b[1]
        colour = label_color(0)
    else:  #gt
        x1 = b[0]-txt_w
        y1 = b[1]-int(txt_h*1.8)
        colour = label_color(3)

    cv2.rectangle(image, (x1, y1), (x1 + int(txt_w), y1 + int(txt_h*1.8)), colour, thickness=-1)
    cv2.putText(image, caption, (x1, y1 + int(txt_h*1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) , 2)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)