import keras
from .. import backend
import numpy as np
import cv2
import tensorflow as tf

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]
    #print(order,num)

    suppressed = np.zeros((num), dtype=np.int)
    #print(suppressed)
    image_q = ((640, 360), (1280, 720), 0)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue

        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]

        tmp = []
        suppressed_former = 0
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            if np.sqrt((boxes[i, 0] - boxes[j, 0])**2 + (boxes[i, 1] - boxes[j, 1])**2) > (boxes[i, 2] + boxes[j, 2] + boxes[i, 3] + boxes[j, 3]):
                inter = 0.0
            else:
                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                try:
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)
                        int_area = cv2.contourArea(order_pts)
                        inter = int_area/ (area_r1 + area_r2 - int_area + 0.00001)
                except:
                    inter = 0.99999



            if inter >= iou_threshold:

                int_pts = cv2.rotatedRectangleIntersection(r1, image_q)[1]
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                outer_i = int_area / area_r1

                int_pts = cv2.rotatedRectangleIntersection(r2, image_q)[1]
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                outer_j = int_area / area_r2

                tmp.append(j)
                if inter > 0.3:   #original effect:
                    if outer_i < 0.976 or outer_j < 0.976:
                        if outer_j < outer_i:
                            #if scores[i] - scores[j] < 0.2:
                            if scores[j] > 0.5:
                                suppressed_former = 1       # 最后条件: 中心点附近有其他的牛 or
        if suppressed_former:
            suppressed[i] = 1
        else:
            suppressed[tmp] =1

    keep = 1 -suppressed
    keep = list(np.where(keep))
    print(keep)

    return np.array(keep[0], np.int64)

def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size,
               use_angle_condition=False, angle_threshold=0, use_gpu=0, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """
    m = decode_boxes[:, 4] * 180 / 3.14159
    m = tf.reshape(m, [-1, 1])

    # x1, y1, x2, y2 to xc,yc,w,h
    xc = (decode_boxes[:, 0] + decode_boxes[:, 2])/2
    xc = tf.reshape(xc, [-1, 1])
    yc = (decode_boxes[:, 1] + decode_boxes[:, 3]) / 2
    yc = tf.reshape(yc, [-1, 1])
    w = (decode_boxes[:, 2] - decode_boxes[:, 0])
    w = tf.reshape(w, [-1, 1])
    h = (decode_boxes[:, 3] - decode_boxes[:, 1])
    h = tf.reshape(h, [-1, 1])

    new_tensor = tf.concat([xc,yc,w,h,m], axis=1)
    keep = tf.py_func(nms_rotate_cpu,
                          inp=[new_tensor, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 20,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        score_threshold       : Threshold used to prefilter the boxes with.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = backend.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]
            # perform NMS
            #############original: get value##############################################################################################
            nms_indices = nms_rotate(filtered_boxes, filtered_scores, nms_threshold, max_detections)
            #nms_indices = backend.non_max_suppression(filtered_boxes[:,:4], filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)
            # filter indices based on NMS  #############original: get value the output shape is the same as nms_indices. find position's(nms_)value in  indices ###########
            indices = keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = backend.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = keras.backend.max(classification, axis    = 1)
        labels  = keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    labels              = keras.backend.gather(labels, top_indices)
    other_              = [keras.backend.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')
    other_   = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    #boxes.set_shape([max_detections, 4])
    boxes.set_shape([max_detections, 5])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    and selecting the top-k detections."""

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 20, #300
        parallel_iterations   = 32,
        **kwargs
    ):
        """
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : IoU : a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]
            return filter_detections(
                boxes,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )
        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            #(input_shape[0][0], self.max_detections, 4),
            (input_shape[0][0], self.max_detections, 5),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output."""
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })
        return config