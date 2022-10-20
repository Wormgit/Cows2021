import cv2
import numpy as np

'''
def cauculate_rt_iou(boxes,query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ious = np.zeros((N, K), dtype=np.float64)
    for i in range(K):
        box2 = get_rbox_poly_corner(query_boxes[i][0],query_boxes[i][1],query_boxes[i][2],query_boxes[i][3],query_boxes[i][4])
        for n in range(N):
            box1 = get_rbox_poly_corner(boxes[n][0], boxes[n][1], boxes[n][2], boxes[n][3], boxes[n][4])
            iou = 0
            if not box1[1].intersects(box2[1]):  # 如果两四边形不相交
                iou = 0
            else:
                try:
                    inter_area = box1[1].intersection(box2[1]).area  # 相交面积
                    union_area = box1[1].area + box2[1].area - inter_area
                    if union_area == 0:
                        iou = 0
                    else:
                        iou = float(inter_area) / union_area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured, iou set to 0')
                    iou = 0
            ious[n, i] = iou
    return ious
'''

def faster_cauculate_rt_iou(boxes, query_boxes):
    N = boxes.shape[0]  # (x1, y1, x2, y2, angle)
    K = query_boxes.shape[0]
    ious = np.zeros((N, K), dtype=np.float64)

    for i in range(K):
        w2 = query_boxes[i][2] - query_boxes[i][0]
        h2 = query_boxes[i][3] - query_boxes[i][1]
        x20 = (query_boxes[i][0] + query_boxes[i][2]) / 2
        y20 = (query_boxes[i][1] + query_boxes[i][3]) / 2
        area2 = w2 * h2
        a2 = -(query_boxes[i][4] * 57.3)  # * 180/3.1416
        r2 = ((x20, y20), (w2, h2), a2)

        for n in range(N):
            a1 = -(boxes[n][4] * 57.3) # put it forward
            if 310> abs(a1 - a2) > 50 :
                ious[n, i] = 0
                continue

            # try:
            #     main_dist = (max((boxes[n + 24][0] - boxes[n][0]), (boxes[n + 24][1] - boxes[n][1])))
            #     dx = (boxes[n][2] + boxes[n][0])/2 - (query_boxes[i][2] + query_boxes[i][0])/2
            #     dy = (boxes[n][2] + boxes[n][0])/2 - (query_boxes[i][2] + query_boxes[i][0])/2
            #     dist = math.sqrt( dx ** 2 + dy ** 2)
            #     if dist > main_dist*1.5: # 128 * 1.5
            #         ious[n, i] = 0
            #         continue
            #     # if dist < main_dist/2:
            #     #     min = (abs(query_boxes[i][2] - 0),abs(query_boxes[i][2] - 0),abs(query_boxes[i][2] - 0),abs(query_boxes[i][2] - 0))
            #     #     ious[n,i]=1
            #     #     continue
            # except:  #through away some margin
            #     continue

            w1 = boxes[n][2] - boxes[n][0]
            h1 = boxes[n][3] - boxes[n][1]
            area1 = w1 * h1
            x10 = (boxes[n][0] + boxes[n][2]) / 2
            y10 = (boxes[n][1] + boxes[n][3]) / 2

            r1 = ((x10, y10), (w1, h1), a1)
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                iou = int_area / (area1 + area2 - int_area)
            else:
                iou = 0
            ious[n, i] = iou
    return ious

# box1 = np.array([[80, 80, 50, 28,  2.64],[80, 80, 50, 28,  2.5]])
# box2 = np.array([[80, 80, 50, 28, 2]])
# kkk = cauculate_rt_iou(box1,box2)
# print (kkk)


