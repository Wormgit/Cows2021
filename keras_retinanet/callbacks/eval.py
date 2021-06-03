import keras
from ..utils.eval import evaluate

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets using a given model at the end of every epoch during training.
    """

    def __init__(
        self,
        generator,            # represents the dataset to evaluate.
        iou_threshold=0.7,    # The threshold used to consider when a detection is positive or negative.
        score_threshold=0.05, # score confidence threshold to use for detections.
        max_detections=20,    # The maximum number of detections to use per image.
        save_path=None,
        tensorboard=None,     #Instance of keras.callbacks.TensorBoard used to log the mAP value.
        weighted_average=False, #Compute the mAP using the weighted average of precisions among classes
        verbose=1
    ):
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average= weighted_average
        self.verbose         = verbose
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard:
            import tensorflow as tf
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_ap
                summary_value.tag = "mAP"
                self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))