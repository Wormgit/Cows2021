import keras
from . import backend
import tensorflow as tf
def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.
            y_true: Tensor of target data from the generator ()gt with shape (B, N, num_classes).
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer
    return _focal

def smooth_l1(sigma=3.0):
    """ sigma: defines the point where the loss changes from L2 to L1. """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4). jing now 5
        """
        # separate target and state
        regression        = y_pred

        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]   ########################## jing the last one is anchor state

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)

        # regression_diff = tf.reshape(regression_diff, [-1, 5])
        # dx, dy, dw, dh, dtheta = tf.unstack(regression_diff, axis=-1)
        # dtheta = tf.atan(dtheta)
        # regression_diff = tf.transpose(tf.stack([dx, dy, dw, dh, dtheta]))

        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared)

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer
    return _smooth_l1
