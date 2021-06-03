import numpy as np
import keras

#from ..utils.compute_overlap import compute_overlap
from ..utils.iou_rt import faster_cauculate_rt_iou  #cauculate_rt_iou

class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : Each stride correspond to one feature level.

    """
    def __init__(self, sizes, strides, ratios, scales, angle):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales
        self.angle = angle

    def num_anchors(self):
        #return len(self.ratios) * len(self.scales)
        return len(self.ratios) * len(self.scales) * len(self.angle)

AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.45], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    angle   = np.array([0, -0.785, -1.571, -2.356, -3.142, 2.356, 1.571, 0.785], keras.backend.floatx())
    #angle =  np.array([0, -0.785, -1.571, -2.356, ], keras.backend.floatx())
    #angle = np.array([0, 0.78,0.39,1.17], keras.backend.floatx())#clockwise JIng
)

def anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    negative_overlap=0.4,
    positive_overlap=0.58
):
    """ Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap:
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."
    batch_size = len(image_group)

    regression_batch  = np.zeros((batch_size, anchors.shape[0], 5 +1), dtype=keras.backend.floatx())
    labels_batch      = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1]       = -1
            labels_batch[index, positive_indices, -1]     = 1

            regression_batch[index, ignore_indices, -1]   = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1
            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        #if image.shape:
           # anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
           # indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])
           # labels_batch[index, indices, -1]     = -1
           # regression_batch[index, indices, -2] = -1
    return regression_batch, labels_batch


def compute_gt_annotations(
    anchors,
    annotations,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """ Obtain indices of gt annotations with the greatest overlap.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    #overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    overlaps = faster_cauculate_rt_iou(anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    #a=np.arange(overlaps.shape[0]) #build index    max_overlaps get iou value
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)
    return shape

def make_shapes_callback(model):
    """ Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes

def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.
    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.
    Returns
        A list of image shapes at each pyramid level.
    """
    aaaaa = image_shape
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    anchor_params=None,
    shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]
    if anchor_params is None:
        anchor_params = AnchorParameters.default
    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 5))   #all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales,
            angle =anchor_params.angle
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
    return all_anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4) reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    #all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    #all_anchors = all_anchors.reshape((K * A, 4))
    tem = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    var = np.array(np.zeros((tem.shape[0],1,1)))
    all_anchors = (anchors.reshape((1, A, 5)) + np.concatenate((tem, var), axis=2))
    all_anchors = all_anchors.reshape((K * A, 5))

    # it seems that the code is slower if i remove margind anchors.
    # AAAA= 0
    # A_index=[]
    # for index in (all_anchors):
    #     if ((index[0] + index[2]) / 2 <= (index[3] - index[1]) / 4) or \
    #             (index[1] + index[3]) / 2 <= (index[3] - index[1]) / 4 or \
    #             (shape [0]*stride -(index[1] + index[3]) / 2 <=  (index[3] - index[1]) / 4) or \
    #             (shape [1]*stride -(index[0] + index[2]) / 2 <=  (index[3] - index[1]) / 4):
    #         A_index.append(AAAA)
    #     AAAA = AAAA + 1
    # all_anchors = np.delete(all_anchors, A_index, axis=0)
    return all_anchors

def generate_anchors(base_size=16, ratios=None, scales=None, angle=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios
    if scales is None:
        scales = AnchorParameters.default.scales
    #num_anchors = len(ratios) * len(scales)##############################################
    if angle is None:
        angle = AnchorParameters.default.angle
    num_anchors = len(ratios) * len(scales) * len(angle)
    # initialize output anchors
    #anchors = np.zeros((num_anchors, 4))
    anchors = np.zeros((num_anchors, 5)) # 12x5 or 9x4(original)
    # scale base_size
    # anchors[:, 2:-1] = base_size * np.tile(scales, (2, len(ratios))).T
    anchors[:, 2:-1] = base_size * np.tile(scales, (2, len(ratios) * len(angle))).T
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3] #LENGTH *HEIGHT

    # correct for ratios
    ratios_temp= np.repeat(ratios, len(scales)*len(angle))
    anchors[:, 2] = np.sqrt(areas / ratios_temp)
    anchors[:, 3] = anchors[:, 2] * ratios_temp

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0:4:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # correct for angle
    tem_an=np.repeat(angle, (len(scales)))
    anchors[:, 4] = np.tile(tem_an, (len(ratios))).T
    return anchors

def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths

    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
    TA= (gt_boxes[:,4] - anchors[:, 4])

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2,TA))
    targets = targets.T
    targets[:,:] = (targets[:,:] - mean) / std
    return targets
