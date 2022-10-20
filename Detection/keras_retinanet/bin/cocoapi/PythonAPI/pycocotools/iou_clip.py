import cv2
import numpy as np

# Account for clip
def faster_cauculate_rt_iou_clip(boxes, query_boxes):
    N = boxes.shape[0]  # (x1, y1, w, h, angle)
    K = query_boxes.shape[0]
    ious = np.zeros((N, K), dtype=np.float64)

    for i in range(K):
        w2 = query_boxes[i][2]
        h2 = query_boxes[i][3]
        x20 = query_boxes[i][0] + query_boxes[i][2] / 2
        y20 = query_boxes[i][1] + query_boxes[i][3] / 2
        a2 = -(query_boxes[i][4] * 57.3)  # * 180/3.1416
        r2 = ((x20, y20), (w2, h2), a2)    # (xc, yc, w, h, angle)
        for n in range(N):
            w1 = boxes[n][2]
            h1 = boxes[n][3]
            area1 = w1 * h1
            x10 = boxes[n][0] + boxes[n][2] / 2
            y10 = boxes[n][1] + boxes[n][3] / 2
            a1 = -(boxes[n][4] * 57.3)
            r1 = ((x10, y10), (w1, h1), a1)
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                iou = int_area / (area1)  #####
            else:
                iou = 0
            ious[n, i] = iou
    return ious


